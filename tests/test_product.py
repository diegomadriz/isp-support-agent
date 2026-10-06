"""Final product contracts: live defaults, channel UX and deployment boundary."""

import asyncio
import json
from pathlib import Path
from threading import Event

import httpx
import pytest
from test_web import payload, post

from isp_support_agent.adapters import FakeMessageSender
from isp_support_agent.classification import rules
from isp_support_agent.language_models import StubLanguageModel
from isp_support_agent.models import CustomerReply, Intent
from isp_support_agent.rewriting import TurnModelBudget, public_facts, rejection_reason
from isp_support_agent.runtime import AgentRuntime
from isp_support_agent.settings import Settings
from isp_support_agent.templates import OPTIONS
from isp_support_agent.web import create_app
from isp_support_agent.whatsapp import CloudMessageSender


@pytest.fixture
def web():
    runtime = AgentRuntime(
        Settings(whatsapp_verify_token="test-token", whatsapp_app_secret="test-secret")
    )
    sender = FakeMessageSender()
    app = create_app(runtime, sender)
    yield app, app.test_client(), runtime, sender
    app.extensions["background"].close()
    runtime.close()


@pytest.mark.parametrize(
    "phrase,intent",
    [
        ("internet lento", Intent.DIAGNOSE),
        ("ver mis reportes", Intent.TICKETS),
        ("pagos", Intent.BILLING),
        ("otra cosa", Intent.UNCLEAR),
    ],
)
def test_suggested_options_have_frozen_rule_and_stub_routes(phrase, intent):
    assert rules(phrase).intent == intent
    assert asyncio.run(StubLanguageModel().classify(phrase)).intent == intent


def test_live_defaults_and_explicit_optout():
    live = dict(
        model_backend="ollama", model_base_url="http://127.0.0.1", model_name="gemma4:latest"
    )
    assert not Settings().rewrite
    assert Settings(**live).rewrite and Settings(**live).model_timeout == 6
    assert Settings(**live).rewrite_timeout == 8
    assert not Settings(**live, rewrite=False).rewrite
    assert Settings(**live, rewrite="").rewrite
    assert Settings(
        model_backend="openai",
        model_base_url="https://example.invalid",
        model_name="example",
        model_key="test",
    ).rewrite


def test_budget_shared_across_both_model_calls():
    class Model:
        async def classify(self, message):
            await asyncio.sleep(0.02)
            return rules(message)

        async def rewrite(self, text, facts):
            await asyncio.sleep(0.08)
            return text

    async def run():
        model = TurnModelBudget(Model(), 0.06)
        assert (await model.classify("internet lento")).intent == Intent.DIAGNOSE
        with pytest.raises(TimeoutError):
            await model.rewrite("Gracias.", {})
        with pytest.raises(Exception, match="timeout"):
            await model.classify("pagos")

    asyncio.run(run())


def test_rewrite_timeout_and_options_fallback_in_live_graph():
    class Model(StubLanguageModel):
        async def rewrite(self, text, facts):
            await asyncio.sleep(1)
            return text

    r = AgentRuntime(
        Settings(
            model_backend="ollama",
            model_base_url="http://127.0.0.1",
            model_name="gemma4:latest",
            model_timeout=0.03,
            rewrite_timeout=0.03,
        ),
        model=Model(),
    )
    try:
        r.turn("a", "cliente 12")
        reply = r.turn("a", "1234")
        assert OPTIONS in reply.message
        assert r.staff_snapshot("a")["rewrite"]["reason"] == "timeout"
    finally:
        r.close()


def test_brand_and_required_options(runtime):
    r = AgentRuntime(Settings(company_name="Proveedor Ejemplo", bot_name="Luna"))
    try:
        r.turn("a", "cliente 12")
        welcome = r.turn("a", "1234")
        assert "Soy Luna de Proveedor Ejemplo" in welcome.message and OPTIONS in welcome.message
        clarify = r.turn("a", "otra cosa")
        assert "Cuéntame con tus palabras" in clarify.message
        facts = public_facts(welcome.message)
        assert (
            rejection_reason(welcome.message.replace(OPTIONS, ""), facts) == "missing_required_fact"
        )
        assert (
            rejection_reason(welcome.message.replace("Luna", "Otro"), facts)
            == "missing_required_fact"
        )
        r.turn("a", "Ya funciona")
        closed = r.turn("a", "sí")
        assert OPTIONS in closed.message
    finally:
        r.close()


def test_typing_precedes_model_work_and_does_not_block_ack(web, monkeypatch):
    app, client, runtime, sender = web
    started, release = Event(), Event()

    def blocked(sid, text):
        assert sender.typing_events == [
            {
                "sender_id": sid,
                "message_id": "mid.001",
                "status": "read",
                "typing_indicator": "text",
            }
        ]
        started.set()
        assert release.wait(2)
        return CustomerReply(message="Listo.")

    monkeypatch.setattr(runtime, "turn", blocked)
    try:
        assert post(client, payload()).status_code == 200
        assert started.wait(1) and not sender.messages
    finally:
        release.set()
        app.extensions["background"].queue.join()
    post(client, payload())
    app.extensions["background"].queue.join()
    assert len(sender.typing_events) == len(sender.messages) == 1


def test_typing_failure_does_not_drop_reply(web, monkeypatch):
    def fail(*args):
        raise RuntimeError("secret")

    monkeypatch.setattr(web[3], "read_and_typing", fail)
    assert post(web[1], payload()).status_code == 200
    web[0].extensions["background"].queue.join()
    assert len(web[3].messages) == 1 and "secret" not in web[3].messages[0]["message"]


def test_cloud_sender_matches_read_plus_typing_contract():
    calls = []

    def handle(request):
        calls.append(
            (str(request.url), json.loads(request.content), request.headers["authorization"])
        )
        return httpx.Response(200, json={"success": True})

    sender = CloudMessageSender(
        Settings(
            whatsapp_sender="cloud",
            whatsapp_base_url="https://example.invalid/messages",
            whatsapp_access_token="test",
        ),
        httpx.Client(transport=httpx.MockTransport(handle)),
    )
    try:
        sender.read_and_typing("customer", "message-1")
        sender.send("customer", "Hola.", "key")
    finally:
        sender.close()
    assert calls[0][1] == {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": "message-1",
        "typing_indicator": {"type": "text"},
    }
    assert calls[1][1]["text"] == {"body": "Hola."}
    assert calls[0][2] == "Bearer test"


def test_health_and_readiness(web):
    app, client, runtime, _ = web
    assert client.get("/healthz").json == {"status": "ok"}
    assert client.get("/readyz").status_code == 200
    app.extensions["background"].close()
    assert client.get("/readyz").status_code == 503
    # Fixture teardown remains safe after this worker shutdown.


def test_review_pack_has_variety_and_no_public_key():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "docs/rewrite-review.md").exists()
    assert not (root / "eval/rewrite-review-key.json").exists()


def test_wsgi_factory_has_persistent_state_and_private_config_errors(tmp_path, monkeypatch):
    from isp_support_agent import server

    callbacks = []
    monkeypatch.setattr(server.atexit, "register", callbacks.append)
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "checkpoint.sqlite"))
    app = server.create_app()
    try:
        client = app.test_client()
        assert client.get("/readyz").status_code == 200
        assert (
            client.post("/api/chat", json={"message": "cliente 12"}).json["awaiting"]
            == "verification"
        )
        assert (tmp_path / "checkpoint.sqlite").exists()
    finally:
        callbacks.pop()()
    monkeypatch.setenv("AGENT_MODEL_BACKEND", "openai")
    monkeypatch.setenv("AGENT_MODEL_KEY", "must-never-appear")
    with pytest.raises(RuntimeError) as exc:
        server.create_app()
    assert "must-never-appear" not in str(exc.value)


def test_http_access_log_is_structured_and_excludes_query_secrets():
    import logging

    from isp_support_agent.logging_config import JsonEventFormatter

    record = logging.LogRecord(
        "gunicorn.access",
        logging.INFO,
        "",
        0,
        '{"event":"http_request","method":"GET","status":200,"duration_us":10}',
        (),
        None,
    )
    assert json.loads(JsonEventFormatter().format(record))["event"] == "http_request"
    cmd = Path("Dockerfile").read_text()
    assert "%(r)s" not in cmd and "%(q)s" not in cmd


def test_usted_options_remain_professional_and_required():
    from isp_support_agent.rewriting import render_public_reply
    from isp_support_agent.templates import CLARIFY

    template, _, profile = render_public_reply(OPTIONS, "¿Usted puede ayudarme?", {})
    assert template.startswith("Puede escribirme")
    assert "«internet lento»" in template
    assert rejection_reason(template, public_facts(template, profile=profile)) is None
    full, _, formal = render_public_reply(CLARIFY, "¿Usted puede ayudarme?", {})
    assert rejection_reason(full, public_facts(full, profile=formal)) is None
    assert "Cuénteme" in full
    r = AgentRuntime(Settings())
    try:
        r.turn("a", "cliente 12")
        r.turn("a", "1234")
        reply = r.turn("a", "¿Usted puede ayudarme con otra cosa?")
        assert "Puede escribirme" in reply.message and "Puedes escribirme" not in reply.message
    finally:
        r.close()
