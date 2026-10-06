import hashlib
import hmac
import json
import time
from threading import Event

import pytest

from isp_support_agent.adapters import FakeMessageSender
from isp_support_agent.models import CustomerReply
from isp_support_agent.runtime import AgentRuntime
from isp_support_agent.settings import Settings
from isp_support_agent.web import create_app, valid_signature


@pytest.fixture
def web():
    runtime = AgentRuntime(
        Settings(
            staff_panel=True, whatsapp_verify_token="test-token", whatsapp_app_secret="test-secret"
        )
    )
    sender = FakeMessageSender()
    app = create_app(runtime, sender)
    yield app, app.test_client(), runtime, sender
    app.extensions["background"].close()
    runtime.close()


def sign(body):
    return "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()


def payload(message="cliente 12", mid="mid.001", sid="sender-1"):
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "id": mid,
                                    "from": sid,
                                    "type": "text",
                                    "timestamp": str(int(time.time())),
                                    "text": {"body": message},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }


def post(client, data):
    body = json.dumps(data).encode()
    return client.post(
        "/webhook/whatsapp",
        data=body,
        content_type="application/json",
        headers={"X-Hub-Signature-256": sign(body)},
    )


def test_webhook_verification_and_signature(web):
    app, client, runtime, sender = web
    response = client.get(
        "/webhook/whatsapp?hub.mode=subscribe&hub.verify_token=test-token&hub.challenge=123456"
    )
    assert response.status_code == 200 and response.text == "123456"
    assert client.get("/webhook/whatsapp?hub.verify_token=bad").status_code == 403
    assert client.post("/webhook/whatsapp", json=payload()).status_code == 403
    assert not valid_signature(b"{}", "sha256=x", "")
    body = json.dumps(payload()).encode()
    assert not valid_signature(body + b" ", sign(body), "test-secret")
    assert post(client, payload()).status_code == 200
    app.extensions["background"].queue.join()
    assert len(sender.messages) == 1
    assert "cuatro dígitos" in sender.messages[0]["message"]
    assert runtime.snapshot("sender-1").next == ("identity",)


def test_background_ack_does_not_wait_for_turn(web, monkeypatch):
    app, client, runtime, sender = web
    started, release = Event(), Event()

    def blocked(sender_id, text):
        started.set()
        assert release.wait(3)
        return CustomerReply(message="Respuesta de prueba.")

    monkeypatch.setattr(runtime, "turn", blocked)
    try:
        response = post(client, payload())
        assert response.status_code == 200
        assert started.wait(2)
        assert sender.messages == []  # Ack has already returned while turn remains blocked.
    finally:
        release.set()
        app.extensions["background"].queue.join()
    assert sender.messages[0]["message"] == "Respuesta de prueba."


def test_whatsapp_duplicate_delivery_and_resume(web):
    app, client, runtime, sender = web
    post(client, payload())
    post(client, payload())
    app.extensions["background"].queue.join()
    assert len(sender.messages) == 1
    post(client, payload("1234", "mid.002"))
    post(client, payload("Luz roja", "mid.003"))
    app.extensions["background"].queue.join()
    assert len(sender.messages) == 3
    assert runtime.tickets.tickets(12) == []
    post(client, payload("Sí", "mid.004"))
    app.extensions["background"].queue.join()
    assert len(runtime.tickets.tickets(12)) == 1
    assert all(set(m) == {"sender_id", "message", "key"} for m in sender.messages)
    assert all("192.0.2" not in m["message"] for m in sender.messages)


@pytest.mark.parametrize("data", [[], {"entry": [{}]}, payload("x" * 2001), payload(sid=3)])
def test_signed_bad_payloads(web, data):
    assert post(web[1], data).status_code == 400


def test_status_notifications_ignored(web):
    app, client, runtime, sender = web
    assert post(client, {"entry": [{"changes": [{"value": {"statuses": []}}]}]}).status_code == 200
    assert (
        post(
            client, {"entry": [{"changes": [{"value": {"messages": [{"type": "image"}]}}]}]}
        ).status_code
        == 200
    )
    assert sender.messages == []


def test_queue_backpressure(web, monkeypatch):
    monkeypatch.setattr(web[0].extensions["background"], "submit", lambda *a: False)
    assert post(web[1], payload()).status_code == 503


def test_background_failure_does_not_expose_exception(web, monkeypatch):
    app, client, runtime, sender = web

    def fail(*args):
        raise RuntimeError("sensitive error")

    monkeypatch.setattr(runtime, "turn", fail)
    assert post(client, payload()).status_code == 200
    app.extensions["background"].queue.join()
    assert len(sender.messages) == 1
    assert "sensitive error" not in sender.messages[0]["message"]


def test_customer_http_boundary_and_local_staff_panel(web):
    app, client, runtime, sender = web
    assert not app.debug
    assert client.get("/health").json == {"status": "ok", "simulated": True}
    assert "Staff panel" in client.get("/").text
    assert (
        client.post("/api/chat", json={"message": "cliente 12"}).json["awaiting"] == "verification"
    )
    assert client.get("/api/staff").json["staff_note"] == ""
    client.post("/api/chat", json={"message": "1234"})
    response = client.post("/api/chat", json={"message": "Lento"})
    assert set(response.json) == {"message", "awaiting"}
    assert "192.0.2" not in response.text
    staff = client.get("/api/staff").json
    assert "192.0.2.12" in staff["staff_note"]
    assert "diagnostics/prepare_probes" in staff["trace"]
    assert client.get("/api/staff", environ_base={"REMOTE_ADDR": "192.0.2.99"}).status_code == 403
    other = app.test_client()
    assert other.post("/api/chat", json={"message": "saldo"}).json["awaiting"] == "customer_id"


@pytest.mark.parametrize("data", [{}, [], {"message": 123}, {"message": "x" * 2001}])
def test_invalid_chat_payload(web, data):
    assert web[1].post("/api/chat", json=data).status_code == 400


def test_web_error_is_safe(web, monkeypatch):
    def fail(*args):
        raise RuntimeError("192.0.2.12 secret")

    monkeypatch.setattr(web[2], "turn", fail)
    response = web[1].post("/api/chat", json={"message": "hola"})
    assert response.status_code == 503
    assert "secret" not in response.text


def test_stale_and_future_messages_are_acknowledged_without_execution(web):
    for timestamp in (int(time.time()) - 601, int(time.time()) + 120):
        data = payload(mid=str(timestamp))
        data["entry"][0]["changes"][0]["value"]["messages"][0]["timestamp"] = str(timestamp)
        assert post(web[1], data).status_code == 200
    web[0].extensions["background"].queue.join()
    assert not web[3].messages


def test_replay_receipt_persists_across_restart(tmp_path):
    settings = Settings(db_path=tmp_path / "state.sqlite", whatsapp_app_secret="test-secret")
    data = payload()
    runtime = AgentRuntime.persistent(settings)
    sender = FakeMessageSender()
    app = create_app(runtime, sender)
    try:
        assert post(app.test_client(), data).status_code == 200
        app.extensions["background"].queue.join()
        assert len(sender.messages) == 1
    finally:
        app.extensions["background"].close()
        runtime.close()
    runtime = AgentRuntime.persistent(settings)
    sender = FakeMessageSender()
    app = create_app(runtime, sender)
    try:
        assert post(app.test_client(), data).status_code == 200
        app.extensions["background"].queue.join()
        assert sender.messages == []
    finally:
        app.extensions["background"].close()
        runtime.close()


def test_cached_reply_retries_after_restart_without_another_graph_turn(tmp_path, monkeypatch):
    settings = Settings(db_path=tmp_path / "state.sqlite", whatsapp_app_secret="test-secret")
    runtime = AgentRuntime.persistent(settings)
    sender = FakeMessageSender()

    def fail(*args):
        raise RuntimeError("transport")

    monkeypatch.setattr(sender, "send", fail)
    app = create_app(runtime, sender)
    try:
        post(app.test_client(), payload())
        app.extensions["background"].queue.join()
        assert runtime.operations.pending()
        assert runtime.snapshot("sender-1").next == ("identity",)
    finally:
        app.extensions["background"].close()
        runtime.close()
    runtime = AgentRuntime.persistent(settings)
    sender = FakeMessageSender()
    app = create_app(runtime, sender)
    try:
        app.extensions["background"].queue.join()
        assert len(sender.messages) == 1
        assert runtime.operations.pending() == []
        assert runtime.snapshot("sender-1").next == ("identity",)
    finally:
        app.extensions["background"].close()
        runtime.close()
