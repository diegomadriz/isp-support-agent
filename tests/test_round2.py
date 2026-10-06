"""Regression checks for review-round-2 behavior, provenance and blind recording."""

import asyncio
import inspect
import json
from pathlib import Path

import httpx
import pytest
from conftest import verified
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command

from isp_support_agent import nodes, recordings
from isp_support_agent.classification import prediction
from isp_support_agent.cli import main
from isp_support_agent.eval_data import CORE_SPLITS
from isp_support_agent.graph import build_graph
from isp_support_agent.models import Intent
from isp_support_agent.operations import OperationStore
from isp_support_agent.runtime import AgentRuntime
from isp_support_agent.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def test_verification_only_charges_four_digit_guesses(runtime):
    runtime.turn("a", "cliente 12")
    for reply in ("¿cuánto debo?", "no recuerdo", "123", "abcd", "12345"):
        answer = runtime.turn("a", reply)
        assert answer.awaiting == "verification"
        assert "no cuenta como intento" in answer.message
        assert runtime.operations.budget(12) == (0, False)
    assert "2 intentos" in runtime.turn("a", "0000").message
    assert "queda 1 intento" in runtime.turn("a", "1111").message
    assert "1 intentos" not in runtime.turn("a", "abcd").message
    assert runtime.operations.budget(12) == (2, False)
    assert "verificada" in runtime.turn("a", " 1234 ").message


def test_suffix_after_lockout_is_not_a_customer_id(runtime):
    runtime.turn("a", "cliente 12")
    for suffix in ("0000", "1111", "2222"):
        runtime.turn("a", suffix)
    answer = runtime.turn("a", "1234")
    assert answer.awaiting is None and "bloqueados" in answer.message
    assert runtime.snapshot("a").values["session"]["customer_id"] == 12
    assert runtime.operations.budget(1234) == (0, False)


def test_raw_message_not_rewritten_for_welcome(runtime):
    runtime.turn("a", "Hola, buenas tardes")
    runtime.turn("a", "12")
    reply = runtime.turn("a", "1234")
    assert "Tu identidad quedó verificada" in reply.message
    turn = runtime.snapshot("a").values["turn"]
    assert turn["message"] == "Hola, buenas tardes" and turn["intent"] == "welcome"
    assert "classify" not in runtime.staff_snapshot("a")["trace"]


@pytest.mark.parametrize("kind", ["customer_id", "verification", "confirmation"])
def test_critical_resume_writes_one_alert_and_acknowledges(kind):
    r = AgentRuntime()
    try:
        if kind == "customer_id":
            r.turn("a", "hola")
        elif kind == "verification":
            r.turn("a", "cliente 12")
        else:
            verified(r, "a")
            r.turn("a", "Quiero cambiar mi plan")
        answer = r.turn("a", "Se cortó la fibra y hay luz roja")
        assert "Ya avisé al equipo técnico" in answer.message
        assert len(r.operations.alerts()) == 1
        assert r.operations.alerts()[0]["customer"] == (12 if kind == "confirmation" else None)
    finally:
        r.close()


def test_anonymous_alerts_collapse_per_sender_and_window():
    now = [100]
    store = OperationStore(clock=lambda: now[0])
    r = AgentRuntime(operations=store)
    try:
        for _ in range(5):
            r.turn("unverified", "luz roja")
        assert len(store.alerts()) == 1
        assert store.alerts()[0]["customer"] is None
        assert "unverified" not in json.dumps(store.alerts())
        r.turn("other", "luz roja")
        assert len(store.alerts()) == 2
        now[0] += 60
        r.turn("unverified", "luz roja")
        assert len(store.alerts()) == 3
    finally:
        r.close()


def test_rate_limited_critical_does_not_alert():
    r = AgentRuntime(Settings(rate_limit=1))
    try:
        r.turn("a", "hola")
        r.turn("a", "luz roja")
        assert not r.operations.alerts()
    finally:
        r.close()


@pytest.mark.parametrize("kind", ["customer_id", "verification", "confirmation"])
@pytest.mark.parametrize("answer", [None, 1234, {}, "", "x" * 2001])
def test_raw_graph_resumes_validate_inside_nodes(runtime, kind, answer):
    async def run():
        g = build_graph(MemorySaver())
        config = {"configurable": {"thread_id": "raw"}}
        message = "hola" if kind == "customer_id" else "cliente 12"
        await g.ainvoke(
            {"session": {}, "turn": {"message": message, "sender_id": "raw", "request_id": "raw"}},
            config,
            context=runtime.context,
        )
        if kind == "confirmation":
            await g.ainvoke(Command(resume="1234"), config, context=runtime.context)
            await g.ainvoke(
                {
                    "turn": {
                        "message": "Quiero cambiar mi plan",
                        "sender_id": "raw",
                        "request_id": "raw-2",
                    }
                },
                config,
                context=runtime.context,
            )
        with pytest.raises(ValueError, match="Resume must"):
            pending = await g.aget_state(config)
            interrupt_id = next(i.id for task in pending.tasks for i in task.interrupts)
            await g.ainvoke(Command(resume={interrupt_id: answer}), config, context=runtime.context)
        assert runtime.operations.budget(12) == (0, False)

    asyncio.run(run())


def test_module_level_nodes_take_runtime_context(runtime):
    state = {"session": {"customer_id": 24, "verified": True}, "turn": {"message": "saldo"}}
    result = asyncio.run(nodes.billing(state, Runtime(context=runtime.context)))
    assert "$499" in result["turn"]["response"]
    result = asyncio.run(nodes.classify_node(state, Runtime(context=runtime.context)))
    assert result["turn"]["intent"] == "billing"
    assert all(
        "<locals>" not in node.__qualname__
        for name, node in inspect.getmembers(nodes, inspect.iscoroutinefunction)
    )


@pytest.mark.parametrize("answer", ["simón", "por fa", "ándale sí", "sí porfa"])
def test_extra_mexican_yeses(answer):
    from isp_support_agent.routing import confirmation_answer

    assert confirmation_answer(answer) == "yes"


def test_studio_context_and_lockout_survive_factory_calls():
    from isp_support_agent.studio import StudioContext, graph, resources, studio_operations

    first, second = graph({}), graph({})
    assert "scenario" in first.get_context_jsonschema()["properties"]
    ctx = StudioContext(scenario="weak_radio_signal")
    assert ctx.operations is StudioContext(scenario="weak_radio_signal").operations
    assert ctx.operations is StudioContext(scenario="healthy").operations
    for _ in range(3):
        ctx.operations.verify(12, False)

    async def run():
        output = await second.ainvoke(
            {
                "session": {},
                "turn": {"message": "cliente 12", "sender_id": "studio", "request_id": "studio"},
            },
            context={"scenario": "weak_radio_signal"},
        )
        assert "bloqueados" in output["turn"]["response"]

    try:
        asyncio.run(run())
    finally:
        ctx.operations.close()
        resources.cache_clear()
        studio_operations.cache_clear()


def test_public_artifacts_disclose_split_roles_and_stub():
    for name in (
        "README.md",
        "docs/architecture.md",
        "docs/eval-ollama.md",
        "docs/eval-rules.md",
        "docs/eval-replay.md",
    ):
        text = (ROOT / name).read_text()
        assert "not independent" in text and "optimistic" in text
        assert "Gemini" in text and "Grok" in text
        assert "fresh chats" in text
        assert "no blind set has been evaluated yet" not in text.casefold()
    for name in ("docs/eval-ollama.json", "docs/eval-rules.json", "docs/eval-replay.json"):
        data = json.loads((ROOT / name).read_text())
        metadata = data.get("metadata", data)
        assert "not independent" in metadata["split_provenance"]["test"]
        assert "not held out" in metadata["split_provenance"]["known-cases"].casefold()
    demo = json.loads((ROOT / "docs/demo-results.json").read_text())
    assert any(row["runtime_paths"].get("stub", 0) for row in demo)
    assert all("model" not in row["runtime_paths"] for row in demo)
    assert "¿Sucede en todos" not in (ROOT / "docs/transcripts/healthy.md").read_text()


def test_gate_pins_graph_baseline_and_detects_second_routing_loss():
    b = json.loads((ROOT / "eval/recordings.json").read_text())
    assert b["baseline_kind"] == "graph-replay"
    assert b["baseline"]["dev"]["correct"] == 98
    assert b["baseline"]["test"]["correct"] == 100
    result = json.loads((ROOT / "docs/eval-replay.json").read_text())
    result["splits"]["test"]["accuracy"] = 99 / 102
    recordings.gate_graph(result, b["baseline"])
    result["splits"]["test"]["accuracy"] = 98 / 102
    with pytest.raises(SystemExit, match="required at least"):
        recordings.gate_graph(result, b["baseline"])


def test_empty_blind_cli_does_not_make_inference(pre_blind_root, monkeypatch):
    monkeypatch.chdir(pre_blind_root)

    def forbidden(*args):
        pytest.fail("Empty blind data must not instantiate a live model")

    monkeypatch.setattr(recordings, "RecordingModel", forbidden)
    with pytest.raises(SystemExit, match="at least 30"):
        main(["eval", "--record-blind"])


@pytest.mark.parametrize("critical_miss", [False, True])
def test_blind_live_capture_appends_only_new_rows_and_enters_gate(
    pre_blind_root, monkeypatch, critical_miss
):
    # Synthetic local fake inputs here exercise the command; they are not the public blind set.
    tmp_path = pre_blind_root
    path = tmp_path / "eval/recordings.json"
    original = json.loads(path.read_text())
    rows = [
        {
            "id": f"fake{i}",
            "message": ("Una situación urgente" if critical_miss else "No tengo internet")
            if i < 5
            else f"Necesito revisar mi pago {i}",
            "intent": "escalate" if i < 5 else "billing",
            "escalation": i < 5,
        }
        for i in range(30)
    ]
    (tmp_path / "eval/blind.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    config = original["config"]

    class Fake(recordings.RecordingModel):
        def __init__(self, settings):
            super().__init__(settings)
            self.client = httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        200,
                        json={
                            "models": [{"name": config["model"], "digest": config["model_digest"]}]
                        },
                    )
                )
            )

        async def _request(self, system, payload, schema=None):
            self.raw = json.dumps(
                prediction(
                    Intent.ESCALATE if payload["message"] == "No tengo internet" else Intent.BILLING
                ).model_dump(mode="json")
            )
            return self.raw

    monkeypatch.setattr(recordings, "RecordingModel", Fake)
    settings = Settings(
        model_backend="ollama",
        model_name=config["model"],
        model_base_url="http://127.0.0.1",
        model_timeout=config["request_timeout_seconds"],
    )
    output = tmp_path / "docs/eval-ollama.json"
    if critical_miss:
        with pytest.raises(SystemExit, match="blind/graph critical recall"):
            asyncio.run(recordings.record_blind(tmp_path, settings, path, output))
        result = json.loads(output.read_text())
    else:
        result = asyncio.run(recordings.record_blind(tmp_path, settings, path, output))
    saved = json.loads(path.read_text())
    assert len(saved["records"]) == len(original["records"]) + len(rows)
    assert saved["records"][: len(original["records"])] == original["records"]
    assert all(
        saved["result"]["splits"][name] == original["result"]["splits"][name]
        for name in CORE_SPLITS
    )
    assert saved["baseline"]["dev"] == original["baseline"]["dev"]
    assert result["splits"]["blind"]["modes"]["combined"]["critical_hits"] == (
        0 if critical_miss else 5
    )
    replay = asyncio.run(recordings.replay_graph(saved, tmp_path))
    if critical_miss:
        with pytest.raises(SystemExit, match="blind/graph critical recall"):
            recordings.gate_graph(replay, saved["baseline"])
    else:
        recordings.gate_graph(replay, saved["baseline"])
    replay["splits"]["blind"]["critical_recall"] = 0.9
    with pytest.raises(SystemExit, match="blind/graph critical recall"):
        recordings.gate_graph(replay, saved["baseline"])
    with pytest.raises(SystemExit, match="already recorded"):
        asyncio.run(recordings.record_blind(tmp_path, settings, path, output))


def test_blind_capture_refuses_wrong_labels_and_changed_existing_policy(pre_blind_root):
    tmp_path = pre_blind_root
    path = tmp_path / "eval/recordings.json"
    original_count = len(json.loads(path.read_text())["records"])
    rows = [
        {"id": "duplicate", "message": "hello", "intent": "unknown", "escalation": False}
        for _ in range(30)
    ]
    (tmp_path / "eval/blind.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(SystemExit, match="Invalid blind"):
        asyncio.run(recordings.record_blind(tmp_path, Settings(), path, tmp_path / "out.json"))
    policy = tmp_path / "src/isp_support_agent/classification.py"
    policy.write_text(policy.read_text() + "\n")
    with pytest.raises(SystemExit, match="Stale or invalid"):
        asyncio.run(recordings.record_blind(tmp_path, Settings(), path, tmp_path / "out.json"))
    assert len(json.loads(path.read_text())["records"]) == original_count


def test_ledger_write_does_not_block_checkpoint_event_loop(runtime):
    from threading import Event

    entered, release = Event(), Event()
    original = runtime.context.operations.critical_alert

    def waiting_write(*args):
        entered.set()
        assert release.wait(1), "Event loop blocked by synchronous ledger write"
        original(*args)

    runtime.context.operations.critical_alert = waiting_write

    async def run():
        state = {
            "session": {"customer_id": 12, "verified": True},
            "turn": {"request_id": "io", "sender_id": "io", "message": "luz roja"},
        }
        task = asyncio.create_task(nodes.escalate(state, Runtime(context=runtime.context)))
        try:
            while not entered.is_set():
                await asyncio.sleep(0)
            release.set()  # Models the checkpoint commit needing this same event loop.
            await asyncio.wait_for(task, 2)
        finally:
            release.set()
        assert len(runtime.operations.alerts()) == 1

    asyncio.run(run())
