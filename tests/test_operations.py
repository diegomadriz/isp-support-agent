import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from conftest import verified
from pydantic import ValidationError

from isp_support_agent.adapters import MockTicketSystem, scenarios
from isp_support_agent.cli import main
from isp_support_agent.demo import run_demo
from isp_support_agent.evaluation import gate
from isp_support_agent.logging_config import JsonEventFormatter
from isp_support_agent.operations import OperationStore
from isp_support_agent.runtime import AgentRuntime, RateLimiter
from isp_support_agent.safety import ADDRESS, HOST, NUMBER, TECHNICAL
from isp_support_agent.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", list(scenarios()))
def test_customer_output_property_across_all_scenarios(name):
    fixture = scenarios()[name]
    r = AgentRuntime(Settings(scenario=name, rewrite=True))
    try:
        allowed = {"12", "24", "4", "3", "2", "1", "2026", "10", "15", "499"}
        for message in fixture["conversation"]:
            reply = r.turn("scenario", message)
            allowed.update(str(t.id) for c in (12, 24) for t in r.tickets.tickets(c))
            assert (
                not ADDRESS.search(reply.message)
                and not HOST.search(reply.message)
                and not TECHNICAL.search(reply.message)
            )
            assert set(NUMBER.findall(reply.message)) <= allowed
            assert set(reply.model_dump()) == {"message", "awaiting"}
            assert "simulación" not in reply.message
        assert "phone_last4" not in json.dumps(r.snapshot("scenario").values)
        if name == "area_outage":
            assert r.tickets.tickets(12) == [] and r.operations.alerts()
        if name == "probe_timeout":
            note = json.loads(r.tickets.tickets(12)[0].notes[0])
            assert any(p["error"] == "timeout" for p in note["probes"].values())
    finally:
        r.close()


def test_demo_reproducible_and_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_BACKEND", "openai")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    rows = run_demo(tmp_path)
    assert len(rows) == 16
    assert rows == json.loads((ROOT / "docs/demo-results.json").read_text())
    for path in (ROOT / "docs/transcripts").glob("*.md"):
        assert (tmp_path / "transcripts" / path.name).read_text() == path.read_text()


def test_sqlite_ticket_confirmation_and_idempotency_survive_restart(tmp_path):
    s = Settings(db_path=tmp_path / "state.sqlite")
    a = AgentRuntime.persistent(s)
    try:
        verified(a, "a")
        a.turn("a", "Luz roja")
    finally:
        a.close()
    b = AgentRuntime.persistent(s)
    try:
        b.turn("a", "Sí, por favor")
        notes = b.tickets.tickets(12)[0].notes
    finally:
        b.close()
    c = AgentRuntime.persistent(s)
    try:
        assert c.tickets.tickets(12)[0].notes == notes
        c.turn("a", "Luz roja")
        c.turn("a", "sí")
        assert c.tickets.tickets(12)[0].notes == notes
        c.turn("a", "Ya funciona")
        assert c.tickets.tickets(12)[0].status == "open"
    finally:
        c.close()
    d = AgentRuntime.persistent(s)
    try:
        d.turn("a", "sí")
        assert d.tickets.tickets(12)[0].status == "resolved"
    finally:
        d.close()


def test_ticket_atomic_dedupe_and_idempotency():
    store = MockTicketSystem(scenarios()["healthy"])
    with ThreadPoolExecutor(4) as pool:
        tickets = list(
            pool.map(
                lambda n: store.create_ticket(12, "technical", f"note{n}", f"key{n}"), range(4)
            )
        )
    assert len({t.id for t in tickets}) == 1
    assert len(store.tickets(12)[0].notes) == 4
    original = store.tickets(12)[0]
    assert store.create_ticket(12, "technical", "duplicate", "key0") == original
    assert store.add_note(original.id, "duplicate", "key1") == original
    with pytest.raises(ValueError):
        store.create_ticket(999, "technical", "bad", "bad")


def test_lockout_cooldown_and_atomic_budget():
    now = [0]
    store = OperationStore(clock=lambda: now[0], cooldown=60)
    try:
        with ThreadPoolExecutor(4) as pool:
            result = list(pool.map(lambda n: store.verify(12, False), range(4)))
        assert sum(remaining == 0 for ok, remaining in result) == 2
        assert store.budget(12)[1]
        assert store.verify(12, True) == (False, 0)
        now[0] = 61
        assert store.verify(12, True) == (True, 3)
    finally:
        store.close()


def test_rate_limit_expiry_and_one_reply_per_window():
    now = [0]
    limiter = RateLimiter(2, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    assert limiter.allow("b")
    now[0] = 61
    assert limiter.allow("a") and "b" not in limiter._times
    r = AgentRuntime(Settings(rate_limit=1))
    try:
        r.turn("a", "cliente 12")
        assert "muchos mensajes" in r.turn("a", "1234").message
        assert r.turn("a", "otra cosa").message == ""
        r.turn("a", "luz roja")
        assert not r.operations.alerts()
    finally:
        r.close()


def test_logs_hide_factors_messages_and_diagnostics(runtime, caplog):
    with caplog.at_level(logging.INFO, logger="isp_support_agent"):
        s = verified(runtime)
        runtime.turn(s, "Luz roja")
    for text in ("1234", "192.0.2", "Luz roja", "customer-a"):
        assert text not in caplog.text
    for record in caplog.records:
        assert "sender_hash" in json.loads(record.message)


def test_logging_records_exception_type_and_logger_only():
    formatter = JsonEventFormatter()
    try:
        raise RuntimeError("private-token")
    except RuntimeError:
        import sys

        record = logging.LogRecord(
            "httpx", logging.ERROR, "x", 1, "private-token", (), sys.exc_info()
        )
    rendered = formatter.format(record)
    assert "private-token" not in rendered
    assert json.loads(rendered)["exception_type"] == "RuntimeError"
    assert json.loads(rendered)["logger"] == "httpx"


def test_settings_once_and_secrets_hidden(monkeypatch):
    monkeypatch.setenv("AGENT_RATE_LIMIT", "7")
    monkeypatch.setenv("AGENT_WHATSAPP_APP_SECRET", "hidden")
    settings = Settings.from_env()
    monkeypatch.setenv("AGENT_RATE_LIMIT", "2")
    assert settings.rate_limit == 7 and "hidden" not in repr(settings)
    with pytest.raises(ValidationError):
        Settings(model_backend="openai")


def test_graph_failure_logged_and_not_silently_fallback(runtime, monkeypatch, caplog):
    async def fail(*args, **kwargs):
        raise RuntimeError("secret detail")
        yield

    monkeypatch.setattr(runtime.graph, "astream", fail)
    with caplog.at_level(logging.INFO, logger="isp_support_agent"), pytest.raises(RuntimeError):
        runtime.turn("a", "cliente 12")
    assert "RuntimeError" in caplog.text and "secret detail" not in caplog.text
    assert not runtime._customer_owners


def test_cli_demo_diagram_chat_and_server(tmp_path, monkeypatch, capsys):
    main(["demo", "--output", str(tmp_path / "demo")])
    main(["diagram", "--output", str(tmp_path / "graph.mmd")])
    assert (tmp_path / "graph.mmd").read_text() == (ROOT / "docs/graph.mmd").read_text()
    messages = iter(["cliente 12", "1234", "saldo"])

    def input_fake(prompt):
        try:
            return next(messages)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", input_fake)
    main(["chat", "--scenario", "healthy"])
    assert "al corriente" in capsys.readouterr().out
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "app.sqlite"))

    def fake_run(self, **kwargs):
        assert kwargs == {"host": "127.0.0.1", "port": 5000, "debug": False, "use_reloader": False}
        assert self.test_client().get("/health").status_code == 200

    monkeypatch.setattr("flask.Flask.run", fake_run)
    main(["serve"])


def test_eval_gate_rejects_missing_critical():
    with pytest.raises(SystemExit, match="graph-replay"):
        gate({"splits": {"dev": {"modes": {"rules-only": {"critical_recall": 0}}}}})


def test_cli_eval_suite_replay_and_gate(tmp_path, capsys, pre_blind_root):
    output = tmp_path / "eval.json"
    main(
        [
            "eval",
            "--directory",
            str(pre_blind_root / "eval"),
            "--output",
            str(output),
            "--calibrate",
        ]
    )
    result = json.loads(output.read_text())
    assert result["metadata"]["backend"] == "stub"
    assert set(result["splits"]) == {"dev", "test", "known-cases"}
    assert result["dev_calibration"] == []
    assert all(set(s["modes"]) == {"rules-only"} for s in result["splits"].values())
    replayed = tmp_path / "replayed.json"
    main(["eval", "--replay", str(output), "--output", str(replayed)])
    assert output.read_bytes() == replayed.read_bytes()
    assert output.with_suffix(".md").read_bytes() == replayed.with_suffix(".md").read_bytes()
    with pytest.raises(SystemExit):
        main(["eval", "--replay", str(output), "--output", str(replayed), "--gate"])
    assert "graph-replay" in capsys.readouterr().err


def test_studio_factory_loads_and_exposes_subgraphs():
    from isp_support_agent.studio import graph

    compiled = graph({})
    assert {"identity", "diagnostics", "ticket_confirmation"} <= set(compiled.nodes)
    compiled.get_input_jsonschema()
    compiled.get_output_jsonschema()
    assert "scenario" in compiled.get_context_jsonschema()["properties"]


def test_reset_does_not_reuse_a_closed_issue_key(runtime):
    s = verified(runtime)
    runtime.turn(s, "Luz roja")
    runtime.turn(s, "sí")
    first = runtime.tickets.tickets(12)[0]
    runtime.turn(s, "Ya funciona")
    runtime.turn(s, "sí")
    runtime.turn(s, "reiniciar")
    verified(runtime, s)
    runtime.turn(s, "Luz roja")
    runtime.turn(s, "sí")
    assert len(runtime.tickets.tickets(12)) == 2
    assert runtime.tickets.tickets(12)[0].status == "resolved"
    assert runtime.tickets.tickets(12)[1].id != first.id


def test_idempotency_key_cannot_return_another_customers_ticket():
    store = MockTicketSystem(scenarios()["healthy"])
    first = store.create_ticket(12, "technical", "note", "same-key")
    second = store.create_ticket(24, "technical", "note", "other-key")
    with pytest.raises(ValueError):
        store.create_ticket(24, "technical", "note", "same-key")
    with pytest.raises(ValueError):
        store.add_note(second.id, "note", "same-key")
    assert store.tickets(12) == [first] and store.tickets(24) == [second]
