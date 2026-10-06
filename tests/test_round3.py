"""Customer options, strict courtesy rules and production safety contracts."""

import asyncio

import pytest
from conftest import verified

from isp_support_agent.language_models import ModelFailure, StubLanguageModel
from isp_support_agent.rewriting import (
    TurnModelBudget,
    public_facts,
    rejection_reason,
    rewrite_reply,
)
from isp_support_agent.runtime import AgentRuntime
from isp_support_agent.settings import Settings
from isp_support_agent.web import create_app


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("internet lento", "módem"),
        ("ver mis reportes", "reportes"),
        ("pagos", "pago"),
        ("otra cosa", "Cuéntame con tus palabras"),
    ],
)
def test_every_suggested_phrase_completes_end_to_end(runtime, phrase, expected):
    sender = verified(runtime)
    reply = runtime.turn(sender, phrase)
    assert expected in reply.message and reply.awaiting is None
    assert not runtime.tickets.tickets(12)


@pytest.mark.parametrize("message", ["otra cosa", "algo raro pasa"])
@pytest.mark.parametrize("consent", ["sí", "no"])
def test_repeated_unclear_offers_human_handoff_with_consent(runtime, message, consent):
    sender = verified(runtime)
    first = runtime.turn(sender, message)
    second = runtime.turn(sender, message)
    assert first.message != second.message and second.awaiting == "confirmation"
    assert "una persona" in second.message and "Responde sí o no." in second.message
    assert not runtime.tickets.tickets(12)
    runtime.turn(sender, consent)
    tickets = runtime.tickets.tickets(12)
    assert len(tickets) == int(consent == "sí")
    if tickets:
        assert tickets[0].category == "general_support"


@pytest.mark.parametrize(
    "addition,code",
    [
        ("Hola. ", "tone_late_greeting"),
        ("Claro. Con gusto. ", "too_many_courtesies"),
        ("Lamento la molestia. Lamento la molestia. ", "tone_multiple_apologies"),
    ],
)
def test_courtesy_rejections_are_explicit(addition, code):
    text = "Registré el folio 1001."
    assert rejection_reason(addition + text, public_facts(text)) == code


@pytest.mark.parametrize(
    "candidate,code",
    [
        ("Registré el folio 1001. ¿Algo más? Claro.", "courtesy_out_of_place"),
        ("Registré el folio 1001. Entiendo. ¿Algo más?", "courtesy_out_of_place"),
        ("Claro. Registré el folio 1001. ¿Algo más?", None),
        ("Registré el folio 1001. ¿Algo más? Gracias por escribirnos.", None),
    ],
)
def test_acknowledgements_open_the_reply(candidate, code):
    text = "Registré el folio 1001. ¿Algo más?"
    assert rejection_reason(candidate, public_facts(text)) == code


@pytest.mark.parametrize("addition", ["Hola. ", "Claro. ", "Con gusto. "])
def test_angry_messages_only_allow_calm_courtesy(addition):
    text = "Registré el folio 1001."
    facts = public_facts(text, "ESTOY HARTO")
    assert rejection_reason(addition + text, facts) is not None
    assert rejection_reason("Entiendo. " + text, facts) is None


def test_greeting_only_allowed_in_first_reply_and_survives_restart(tmp_path):
    class Model(StubLanguageModel):
        async def rewrite(self, text, facts):
            return "Hola. " + text

    settings = Settings(db_path=tmp_path / "state.sqlite", rewrite=True)
    agent = AgentRuntime.persistent(settings)
    from dataclasses import replace

    agent.model = Model()
    agent.context = replace(agent.context, model=agent.model)
    try:
        agent.turn("a", "cliente 12")
        reply = agent.turn("a", "1234")
        assert "Hola." not in reply.message
        assert agent.staff_snapshot("a")["rewrite"]["reason"] == "tone_late_greeting"
    finally:
        agent.close()
    agent = AgentRuntime.persistent(settings)
    from dataclasses import replace

    agent.model = Model()
    agent.context = replace(agent.context, model=agent.model)
    try:
        reply = agent.turn("a", "pagos")
        assert "Hola." not in reply.message
        assert agent.staff_snapshot("a")["rewrite"]["reason"] == "tone_late_greeting"
    finally:
        agent.close()


def test_first_reply_can_contain_one_greeting():
    text = "Registré el folio 1001."
    facts = public_facts(text)
    facts["tone_profile"]["greeting_allowed"] = True
    facts = public_facts(text, profile=facts["tone_profile"])
    assert rejection_reason("Hola. " + text, facts) is None


def test_actions_questions_and_options_keep_their_order():
    text = "Prueba acercarte al módem. ¿Quieres que abra un reporte? Responde sí o no."
    candidate = "¿Quieres que abra un reporte? Prueba acercarte al módem. Responde sí o no."
    assert rejection_reason(candidate, public_facts(text)) == "action_question_order"


def test_rewrite_skips_when_less_than_1500ms_remain():
    class Model:
        async def rewrite(self, *args):
            raise AssertionError("must not run")

    def clock():
        return 6.6

    model = TurnModelBudget(Model(), 8, clock=clock, started_at=0)
    sent, result = asyncio.run(
        rewrite_reply("Registré el folio 1001.", "hola", model, enabled=True, verified=True)
    )
    assert sent == "Registré el folio 1001." and not result["attempted"]
    assert result["reason"] == "insufficient_turn_budget"


@pytest.mark.parametrize("reason", ["timeout", "http_error"])
@pytest.mark.parametrize("stage", ["fresh", "verification", "confirmation"])
def test_model_unavailable_unclassified_messages_alert_staff(reason, stage):
    class Model(StubLanguageModel):
        async def classify(self, message):
            raise ModelFailure(reason)

    agent = AgentRuntime(model=Model())
    try:
        if stage == "verification":
            agent.turn("a", "cliente 12")
        else:
            verified(agent, "a")
            if stage == "confirmation":
                agent.turn("a", "no tengo internet")
        before = len(agent.operations.alerts())
        reply = agent.turn("a", "la antena se cayó con el aire")
        assert len(agent.operations.alerts()) == before + 1
        assert agent.operations.alerts()[-1]["kind"] == "unclassified message, model unavailable"
        assert reply.message and not agent.tickets.tickets(12)
    finally:
        agent.close()


def test_staff_panel_is_off_by_default_and_explicitly_opt_in():
    for enabled in (False, True):
        agent = AgentRuntime(Settings(staff_panel=enabled))
        app = create_app(agent)
        try:
            client = app.test_client()
            assert client.get("/api/staff").status_code == (200 if enabled else 403)
            assert ('id="trace"' in client.get("/").text) == enabled
        finally:
            app.extensions["background"].close()
            agent.close()
