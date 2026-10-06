import hashlib
import hmac
import secrets
import time
from queue import Full, Queue
from threading import Lock, Thread

from flask import Flask, jsonify, render_template, request, session

from .adapters import FakeMessageSender
from .models import CustomerReply
from .ports import MessageSender
from .runtime import AgentRuntime, log_event


class BackgroundReplies:
    """Bounded FIFO worker; a webhook request only verifies and enqueues."""

    def __init__(self, runtime: AgentRuntime, sender: MessageSender):
        self.runtime = runtime
        self.sender = sender
        self.queue = Queue(maxsize=64)
        self._seen = set()
        self._lock = Lock()
        self.worker = Thread(target=self._work, daemon=True, name="whatsapp-replies")
        self.worker.start()
        for job in self.runtime.operations.pending():
            self.queue.put(job)
            self._seen.add(job[0])

    def submit(self, message_id: str, sender_id: str, text: str) -> bool:
        with self._lock:
            if message_id in self._seen:
                return True
            if not self.runtime.operations.receive(message_id, sender_id, text):
                return True
            try:
                self.queue.put_nowait((message_id, sender_id, text))
            except Full:
                self.runtime.operations.forget(message_id)
                return False
            self._seen.add(message_id)
            return True

    def _work(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                message_id, sender_id, text = item
                cached = self.runtime.operations.cached_reply(message_id)
                if cached is None:
                    try:
                        self.sender.read_and_typing(sender_id, message_id)
                    except Exception as error:
                        log_event("typing_failed", sender_id, exception_type=type(error).__name__)
                    try:
                        reply = self.runtime.turn(sender_id, text)
                    except Exception as error:
                        log_event(
                            "background_turn_failed", sender_id, exception_type=type(error).__name__
                        )
                        reply = CustomerReply(
                            message="No pude completar la solicitud. Intenta de nuevo más tarde."
                        )
                    cached = reply.message
                    self.runtime.operations.save_reply(message_id, cached)
                try:
                    if cached:
                        self.sender.send(sender_id, cached, message_id)
                    self.runtime.operations.delivered(message_id)
                except Exception as error:
                    log_event(
                        "background_send_failed", sender_id, exception_type=type(error).__name__
                    )
                finally:
                    self._seen.discard(message_id)
            finally:
                self.queue.task_done()

    def close(self):
        if not self.worker.is_alive():
            return
        self.queue.join()
        self.queue.put(None)
        self.worker.join(timeout=5)


def valid_signature(body: bytes, signature: str, secret: str) -> bool:
    if not secret or not signature.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


def create_app(runtime: AgentRuntime, sender: MessageSender | None = None) -> Flask:
    app = Flask(__name__)
    app.secret_key = runtime.settings.session_secret or secrets.token_bytes(32)
    app.config.update(
        DEBUG=False,
        MAX_CONTENT_LENGTH=64 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
    )
    if sender is None and runtime.settings.whatsapp_sender == "cloud":
        from .whatsapp import CloudMessageSender

        sender = CloudMessageSender(runtime.settings)
    outgoing = sender or FakeMessageSender()
    background = BackgroundReplies(runtime, outgoing)
    app.extensions.update(agent_runtime=runtime, message_sender=outgoing, background=background)

    def web_sender():
        if "sender_id" not in session:
            session["sender_id"] = "web-" + secrets.token_hex(16)
        return session["sender_id"]

    @app.get("/")
    def home():
        return render_template("index.html", staff_panel=runtime.settings.staff_panel)

    @app.get("/health")
    def health():
        return jsonify(status="ok", simulated=True)

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok")

    @app.get("/readyz")
    def readyz():
        try:
            if runtime._closed or not background.worker.is_alive():
                return jsonify(status="not_ready"), 503
            with runtime.operations.lock:
                runtime.operations.connection.execute("SELECT 1").fetchone()
            return jsonify(status="ready", model=runtime.settings.model_backend, simulated=True)
        except Exception:
            return jsonify(status="not_ready"), 503

    @app.post("/api/chat")
    def chat():
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict) or not isinstance(body.get("message"), str):
            return jsonify(error="Mensaje inválido."), 400
        try:
            return jsonify(runtime.turn(web_sender(), body["message"]).model_dump())
        except ValueError:
            return jsonify(error="Mensaje inválido."), 400
        except Exception:
            return jsonify(error="No pude completar la solicitud. Intenta de nuevo más tarde."), 503

    @app.get("/api/staff")
    def staff():
        # This endpoint belongs exclusively to the local, explicitly labeled demo staff panel.
        if not runtime.settings.staff_panel or request.remote_addr not in {"127.0.0.1", "::1"}:
            return jsonify(error="Local demo only"), 403
        return jsonify(runtime.staff_snapshot(web_sender()))

    @app.get("/webhook/whatsapp")
    def verification():
        token = runtime.settings.whatsapp_verify_token
        if (
            token
            and request.args.get("hub.mode") == "subscribe"
            and hmac.compare_digest(request.args.get("hub.verify_token", ""), token)
        ):
            return request.args.get("hub.challenge", ""), 200, {"Content-Type": "text/plain"}
        return "", 403

    @app.post("/webhook/whatsapp")
    def webhook():
        if not valid_signature(
            request.get_data(),
            request.headers.get("X-Hub-Signature-256", ""),
            runtime.settings.whatsapp_app_secret,
        ):
            return "", 403
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return "", 400
        jobs = []
        try:
            for entry in data.get("entry", []):
                for change in entry["changes"]:
                    for message in change["value"].get("messages", []):
                        if message.get("type") != "text":
                            continue
                        mid, sid, text = message["id"], message["from"], message["text"]["body"]
                        if (
                            not all(isinstance(x, str) and x for x in (mid, sid, text))
                            or len(mid) > 256
                            or len(sid) > 128
                            or len(text) > 2000
                        ):
                            raise ValueError("Invalid message")
                        timestamp = int(message["timestamp"])
                        age = time.time() - timestamp
                        if age < -60 or age > runtime.settings.webhook_max_age:
                            continue  # Acknowledge stale deliveries without processing them.
                        jobs.append((mid, sid, text))
        except (KeyError, TypeError, AttributeError, ValueError):
            return "", 400
        for job in jobs:
            if not background.submit(*job):
                return (
                    "",
                    503,
                )  # Ask the upstream transport to retry; never silently drop queue overflow.
        return "", 200

    return app
