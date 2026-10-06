import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from conftest import verified

from isp_support_agent.adapters import SimulatedNetwork, scenarios
from isp_support_agent.classification import prediction
from isp_support_agent.graph import confirmation_answer
from isp_support_agent.language_models import StubLanguageModel
from isp_support_agent.models import Intent
from isp_support_agent.runtime import AgentRuntime
from isp_support_agent.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "message,intent,awaiting",
    [
        ("Está lento", "diagnose", None),
        ("Luz roja", "escalate", "confirmation"),
        ("Consultar saldo", "billing", None),
        ("Cambiar mi plan", "admin", "confirmation"),
        ("Ver tickets", "tickets", None),
        ("Ya quedó", "resolved", "confirmation"),
        ("Volver al menú", "reset", None),
        ("Eso", "unclear", None),
        ("gracias", "thanks", None),
    ],
)
def test_routes(runtime, message, intent, awaiting):
    s = verified(runtime)
    reply = runtime.turn(s, message)
    assert reply.awaiting == awaiting
    state = runtime.snapshot(s).values
    if intent != "reset":
        assert state["turn"]["intent"] == intent
    assert reply.message


def test_identity_loops_invalid_id_and_trims_factor(runtime):
    assert runtime.turn("a", "hola").awaiting == "customer_id"
    assert runtime.turn("a", "no me acuerdo").awaiting == "customer_id"
    assert runtime.turn("a", "cliente 12").awaiting == "verification"
    assert "verificada" in runtime.turn("a", " 1234 ").message
    assert runtime.operations.budget(12) == (0, False)


def test_change_id_during_verification_is_not_guess(runtime):
    runtime.turn("a", "cliente 12")
    assert runtime.turn("a", "cliente 24").awaiting == "verification"
    assert runtime.operations.budget(12) == (0, False)
    assert "verificada" in runtime.turn("a", "5678").message
    assert "pendiente" in runtime.turn("a", "saldo").message


def test_unknown_and_known_failure_feedback_is_identical(runtime):
    assert runtime.turn("known", "cliente 12") == runtime.turn("unknown", "cliente 999")
    for attempt in range(3):
        known = runtime.turn("known", "0000")
        unknown = runtime.turn("unknown", "0000")
        assert known == unknown
        if attempt < 2:
            assert f"{2 - attempt} intento" in known.message
    assert "Comunícate con nosotros" in known.message


def test_shared_budget_cannot_reset_or_change_sender(runtime):
    runtime.turn("a", "cliente 12")
    for m in ("0000", "1111", "2222"):
        runtime.turn("a", m)
    runtime.turn("a", "reiniciar")
    assert "bloqueados" in runtime.turn("a", "cliente 12").message
    assert "bloqueados" in runtime.turn("b", "cliente 12").message
    assert "pendiente" not in runtime.turn("b", "1234").message


def test_lockout_and_interrupt_survive_restart(tmp_path):
    settings = Settings(db_path=tmp_path / "state.sqlite")
    a = AgentRuntime.persistent(settings)
    try:
        a.turn("a", "cliente 12")
        a.turn("a", "0000")
    finally:
        a.close()
    b = AgentRuntime.persistent(settings)
    try:
        assert "1 intento" in b.turn("a", "1111").message
        assert "bloqueados" in b.turn("a", "2222").message
    finally:
        b.close()
    c = AgentRuntime.persistent(settings)
    try:
        assert "bloqueados" in c.turn("b", "cliente 12").message
    finally:
        c.close()


def test_thread_isolation_and_account_switch(runtime):
    verified(runtime, "a")
    assert runtime.turn("b", "saldo").awaiting == "customer_id"
    assert not runtime._session(runtime.snapshot("b")).get("verified")
    assert runtime.turn("a", "cliente 24").awaiting == "verification"
    runtime.turn("a", "5678")
    assert "pendiente" in runtime.turn("a", "saldo").message


def test_original_issue_after_verification(runtime):
    assert runtime.turn("a", "No tengo internet").awaiting == "customer_id"
    runtime.turn("a", "cliente 12")
    assert runtime.turn("a", "1234").awaiting == "confirmation"
    assert runtime.operations.alerts()


@pytest.mark.parametrize("at", ["customer_id", "verification", "confirmation"])
def test_critical_resume_always_alerts(runtime, at):
    if at == "customer_id":
        runtime.turn("a", "hola")
    elif at == "verification":
        runtime.turn("a", "cliente 12")
    else:
        verified(runtime, "a")
        runtime.turn("a", "Cambiar mi plan")
    reply = runtime.turn("a", "ahora tengo luz roja y se cortó la fibra")
    assert runtime.operations.alerts()
    assert reply.awaiting == at
    assert runtime.tickets.tickets(12) == []
    if at == "confirmation":
        assert "Ya avisé" in reply.message
        runtime.turn("a", "no")
        assert runtime.tickets.tickets(12) == []


def test_model_critical_cannot_be_downgraded_or_bypassed_in_confirmation():
    r = AgentRuntime(model=StubLanguageModel(prediction(Intent.ADMIN)))
    try:
        verified(r, "a")
        r.turn("a", "Solicito revisión")
        r.model.fixed = prediction(Intent.BILLING, 0.1, critical=True)
        assert "Ya avisé" in r.turn("a", "Mi solicitud necesita atención urgente").message
        assert r.operations.alerts()
        assert r.tickets.tickets(12) == []
    finally:
        r.close()


@pytest.mark.parametrize(
    "answer",
    [
        "Sí, por favor",
        "sí.",
        "si!",
        "sí claro",
        "ok",
        "dale",
        "va",
        "sale",
        "órale",
        "ándale pues",
        "sip",
        "por favor",
    ],
)
def test_natural_confirmations(answer):
    assert confirmation_answer(answer) == "yes"


@pytest.mark.parametrize("answer", ["no", "nel", "mejor no", "cancela", "no gracias"])
def test_negative_confirmations(answer):
    assert confirmation_answer(answer) == "no"


def test_ambiguous_confirmation_keeps_question_and_context():
    r = AgentRuntime(Settings(scenario="weak_radio_signal"))
    try:
        s = verified(r)
        question = r.turn(s, "Está lento").message
        assert r.turn(s, "quizá").message == question
        assert r.turn(s, "tal vez").message == question
        final = r.turn(s, "no sé")
        assert "No pude confirmar" in final.message
        assert not r.tickets.tickets(12)
    finally:
        r.close()


def test_ticket_confirmation_dedupe_and_resolution():
    r = AgentRuntime(Settings(scenario="existing_open_ticket"))
    try:
        s = verified(r)
        original = r.tickets.tickets(12)[0]
        question = r.turn(s, "Mi internet sigue lento")
        assert str(original.id) in question.message
        assert r.tickets.tickets(12)[0] == original
        r.turn(s, "Sí, por favor")
        assert len(r.tickets.tickets(12)) == 1
        assert len(r.tickets.tickets(12)[0].notes) == len(original.notes) + 1
        r.turn(s, "¿Ya quedó mi reporte?")
        assert r.tickets.tickets(12)[0].status == "open"
        assert r.turn(s, "Ya funciona pero sigue lento").awaiting == "confirmation"
        r.turn(s, "no")
        assert r.tickets.tickets(12)[0].status == "open"
        assert r.turn(s, "Ya funciona bien").awaiting == "confirmation"
        assert r.tickets.tickets(12)[0].status == "open"
        r.turn(s, "no")
        assert r.tickets.tickets(12)[0].status == "open"
        r.turn(s, "Ya funciona bien")
        r.turn(s, "sí")
        assert r.tickets.tickets(12)[0].status == "resolved"
    finally:
        r.close()


def test_model_selected_resolution_still_requires_consent():
    r = AgentRuntime(
        Settings(scenario="existing_open_ticket"),
        model=StubLanguageModel(prediction(Intent.RESOLVED)),
    )
    try:
        s = verified(r)
        assert r.turn(s, "Haz algo").awaiting == "confirmation"
        assert r.tickets.tickets(12)[0].status == "open"
    finally:
        r.close()


def test_low_confidence_and_model_primary():
    r = AgentRuntime(model=StubLanguageModel(prediction(Intent.TICKETS, 0.2)))
    try:
        s = verified(r)
        assert "Cuéntame" in r.turn(s, "saldo").message
        assert r.snapshot(s).values["turn"]["classification"]["fallback_reason"] == "low_confidence"
        r.model.fixed = prediction(Intent.TICKETS, 0.95)
        assert "reportes" in r.turn(s, "saldo").message
        assert r.snapshot(s).values["turn"]["intent"] == "tickets"
    finally:
        r.close()


def test_mixed_billing_tickets(runtime):
    s = verified(runtime, customer=24, factor="5678")
    reply = runtime.turn(s, "Pago y tickets")
    assert "$499" in reply.message and "reportes" in reply.message


def test_parallel_probes_partial_timeout_and_runtime_gate():
    ready = Event()

    class Network(SimulatedNetwork):
        def __init__(self):
            super().__init__(scenarios()["healthy"]["network"])
            self.started = set()
            self.all_started = asyncio.Event()

        async def _run(self, kind, model, service):
            self.started.add(kind)
            if len(self.started) == 4:
                self.all_started.set()
                ready.set()
            await asyncio.wait_for(self.all_started.wait(), 0.4)
            if kind == "cpe":
                await asyncio.sleep(10)
            await asyncio.sleep(0.04)
            return await super()._run(kind, model, service)

    network = Network()
    r = AgentRuntime(Settings(probe_timeout=0.15), network=network)
    try:
        verified(r, "a")
        verified(r, "b")
        with ThreadPoolExecutor() as pool:
            future = pool.submit(r.turn, "a", "Está lento")
            assert ready.wait(1)
            assert "en curso" in r.turn("b", "Está lento").message
            assert future.result(timeout=2).awaiting == "confirmation"
        note = json.loads(r.staff_snapshot("a")["staff_note"])
        assert note["probes"]["cpe"]["error"] == "timeout"
        assert len(note["probes"]) == 4
        assert not r._customer_owners
        assert not r._locks
    finally:
        r.close()


def test_trace_from_langgraph_stream_not_state(runtime):
    s = verified(runtime)
    runtime.turn(s, "Está lento")
    staff = runtime.staff_snapshot(s)
    assert any("diagnostics/probe" in node for node in staff["trace"])
    assert "trace" not in runtime.snapshot(s).values
    assert set(runtime.snapshot(s).values) == {"session", "turn"}


def test_diagram_is_current(runtime):
    diagram = runtime.graph.get_graph(xray=True).draw_mermaid() + "\n"
    assert (ROOT / "docs/graph.mmd").read_text() == diagram
    for subgraph in ("identity", "diagnostics", "ticket_confirmation"):
        assert f"subgraph {subgraph}" in diagram


@pytest.mark.parametrize("message", [None, 12, {}, [], True, "x" * 2001, ""])
def test_typed_resume_rejected_without_spending_attempt(runtime, message):
    runtime.turn("a", "cliente 12")
    with pytest.raises(ValueError):
        runtime.turn("a", message)
    assert runtime.operations.budget(12) == (0, False)


def test_readme_overview_is_generated(runtime):
    diagram = runtime.graph.get_graph().draw_mermaid()
    readme = (ROOT / "README.md").read_text()
    assert readme.split("```mermaid\n", 1)[1].split("\n```", 1)[0] == diagram


def test_invalid_id_cap_and_reset_from_pending_identity(runtime):
    runtime.turn("a", "hola")
    for text in ("no recuerdo", "un recibo", "no sé"):
        reply = runtime.turn("a", text)
    assert reply.awaiting is None and "comunícate con nosotros" in reply.message
    runtime.turn("b", "cliente 12")
    assert runtime.turn("b", "reiniciar").awaiting is None
    assert runtime.turn("b", "cliente 12").awaiting == "verification"


def test_service_specific_complaint_and_greeting(runtime):
    s = verified(runtime)
    assert "¿En qué puedo ayudarte" in runtime.turn(s, "Hola, buenas tardes").message
    reply = runtime.turn(s, "se traba mucho netflix")
    assert "módem" in reply.message
    assert runtime.snapshot(s).values["turn"]["intent"] == "diagnose"


def test_model_secondary_intent_is_served_even_without_keyword():
    model = StubLanguageModel(prediction(Intent.BILLING, secondary_intents=(Intent.TICKETS,)))
    r = AgentRuntime(model=model)
    try:
        s = verified(r)
        reply = r.turn(s, "Quisiera revisar mi cuenta y lo que he levantado antes")
        assert "pago" in reply.message and "reportes" in reply.message
    finally:
        r.close()


def test_resolution_does_not_close_unrelated_admin_ticket(runtime):
    runtime.tickets.create_ticket(12, "administrative", "Pending cancellation", "admin")
    s = verified(runtime)
    reply = runtime.turn(s, "Ya funciona bien")
    assert reply.awaiting == "confirmation" and "folio" not in reply.message
    runtime.turn(s, "sí")
    assert runtime.tickets.tickets(12)[0].status == "open"


def test_oversized_numeric_id_stays_in_identity_prompt(runtime):
    assert runtime.turn("a", "cliente " + "9" * 200).awaiting == "customer_id"
    assert runtime.turn("a", "9" * 200).awaiting == "customer_id"
    assert runtime.turn("a", "cliente 12").awaiting == "verification"


def test_account_switch_drops_previous_issue_key_and_staff_evidence(runtime):
    s = verified(runtime)
    runtime.turn(s, "Está lento")
    assert runtime.staff_snapshot(s)["staff_note"]
    runtime.turn(s, "Luz roja")
    runtime.turn(s, "sí")
    original = runtime.tickets.tickets(12)[0]
    assert runtime.turn(s, "cliente 24").awaiting == "verification"
    assert runtime.staff_snapshot(s)["staff_note"] == ""
    runtime.turn(s, "5678")
    assert runtime.staff_snapshot(s)["staff_note"] == ""
    runtime.turn(s, "Luz roja")
    runtime.turn(s, "sí")
    new = runtime.tickets.tickets(24)[0]
    assert original.id != new.id and new.customer_id == 24
    assert runtime.tickets.tickets(12)[0] == original
