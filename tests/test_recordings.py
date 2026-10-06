import asyncio
import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from conftest import verified

from isp_support_agent import recordings
from isp_support_agent.classification import critical_signal
from isp_support_agent.cli import main
from isp_support_agent.eval_data import blind_status, provenance
from isp_support_agent.evaluation import load_rows, replay
from isp_support_agent.language_models import ModelFailure, StubLanguageModel
from isp_support_agent.recordings import (
    POLICY_FILES,
    RecordedModel,
    RecordingModel,
    canonical,
    digest,
    fingerprints,
    gate_graph,
    replay_graph,
    validate_recordings,
    write_graph_results,
)
from isp_support_agent.runtime import AgentRuntime
from isp_support_agent.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle():
    return json.loads((ROOT / "eval/recordings.json").read_text())


def test_full_recorded_graph_replay_offline_and_reproducible(bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    result = asyncio.run(replay_graph(bundle, ROOT))
    gate_graph(result, bundle["baseline"])
    assert result["splits"]["test"]["critical_hits"] == 30
    assert result["splits"]["test"]["model_calls"] > 0
    assert result["splits"]["test"]["paths"]["critical_rule"] == 17
    output = tmp_path / "replay.json"
    write_graph_results(result, output)
    assert output.read_bytes() == (ROOT / "docs/eval-replay.json").read_bytes()
    assert output.with_suffix(".md").read_bytes() == (ROOT / "docs/eval-replay.md").read_bytes()
    saved = bundle["result"]
    expected = deepcopy(saved)
    expected["metadata"].update(
        split_provenance=provenance(saved["splits"]), blind_status=blind_status(saved["splits"])
    )
    assert replay(deepcopy(saved)) == expected
    assert len(bundle["records"]) == sum(len(s["rows"]) for s in saved["splits"].values())
    assert all(isinstance(r["raw_output"], str) or r["error"] for r in bundle["records"])


@pytest.mark.parametrize("name", POLICY_FILES)
def test_changed_policy_data_or_config_requires_live_recording(bundle, tmp_path, name):
    for path in POLICY_FILES:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / path).read_bytes())
    with (tmp_path / name).open("ab") as target:
        target.write(b"\n")
    with pytest.raises(SystemExit, match="re-record.*live model|Configure the live model"):
        validate_recordings(bundle, tmp_path)


def test_changed_prompt_or_schema_requires_recording(bundle, monkeypatch):
    monkeypatch.setattr(recordings, "classification_prompt", lambda: "changed prompt")
    with pytest.raises(SystemExit, match="prompt"):
        validate_recordings(bundle, ROOT)


@pytest.mark.parametrize(
    "mutation", ["config", "metadata", "raw", "missing", "duplicate", "row", "hash", "no_raw"]
)
def test_corrupt_or_incomplete_capture_fails_closed(bundle, mutation):
    if mutation == "config":
        bundle["config"]["confidence_threshold"] = 0.1
    elif mutation == "metadata":
        bundle["result"]["metadata"]["confidence_threshold"] = 0.1
    elif mutation == "raw":
        bundle["records"][0]["raw_output"] += "changed"
    elif mutation == "missing":
        bundle["records"].pop()
    elif mutation == "duplicate":
        bundle["records"][1] = bundle["records"][0]
    elif mutation == "row":
        bundle["result"]["splits"]["dev"]["rows"][0]["message"] = "changed"
    elif mutation == "hash":
        bundle["records"][0]["config_hash"] = "changed"
    else:
        bundle["records"][0]["raw_output"] = None
    if mutation not in {"raw", "config"}:
        bundle["records_sha256"] = digest(canonical(bundle["records"]))
    with pytest.raises(SystemExit, match="Configure the live model"):
        validate_recordings(bundle, ROOT)


def test_parsed_prediction_cannot_replace_raw_capture(bundle):
    bundle["result"]["splits"]["dev"]["samples"][0]["prediction"]["reason"] = "edited"
    with pytest.raises(SystemExit, match="raw/parsed"):
        asyncio.run(replay_graph(bundle, ROOT))


def test_gate_checks_current_graph_routing_not_only_policy(bundle, monkeypatch):
    import isp_support_agent.graph as graph_module

    original = graph_module.intent_route
    monkeypatch.setattr(
        graph_module,
        "intent_route",
        lambda state: "unclear" if state["turn"].get("intent") == "escalate" else original(state),
    )
    result = asyncio.run(replay_graph(bundle, ROOT))
    with pytest.raises(SystemExit, match="critical recall"):
        gate_graph(result, bundle["baseline"])


def test_escalation_must_actually_alert_staff(bundle, monkeypatch):
    monkeypatch.setattr(recordings.OperationStore, "alert", lambda *args: None)
    with pytest.raises(SystemExit, match="did not create staff alert"):
        asyncio.run(replay_graph(bundle, ROOT))


@pytest.mark.parametrize("split", ["dev", "test"])
def test_accuracy_floor_is_one_percentage_point_not_one_percent(split):
    baseline = {"dev": {"accuracy": 97 / 104}, "test": {"accuracy": 99 / 102}}
    result = {
        "splits": {
            k: {"critical_recall": 1, "accuracy": v["accuracy"]} for k, v in baseline.items()
        }
    }
    result["splits"][split]["accuracy"] = baseline[split]["accuracy"] - 0.01
    gate_graph(result, baseline)
    result["splits"][split]["accuracy"] -= 0.000001
    with pytest.raises(SystemExit, match="1 percentage point"):
        gate_graph(result, baseline)
    result["splits"][split]["accuracy"] = baseline[split]["accuracy"]
    result["splits"][split]["critical_recall"] = 0.999
    with pytest.raises(SystemExit, match="required 100%"):
        gate_graph(result, baseline)


@pytest.mark.parametrize("reason", ["timeout", "http_error"])
def test_provider_failure_clarifies_but_critical_guard_still_alerts(reason):
    class UnavailableModel(StubLanguageModel):
        async def classify(self, message):
            raise ModelFailure(reason)

    agent = AgentRuntime(Settings(), model=UnavailableModel())
    try:
        sender = verified(agent)
        reply = agent.turn(sender, "Mi conexión necesita revisión")
        classification = agent.snapshot(sender).values["turn"]["classification"]
        assert classification["prediction"]["intent"] == "unclear"
        assert classification["fallback_reason"] == "model_error"
        assert classification["model_error"] == reason
        assert "Cuéntame" in reply.message
        assert agent.operations.alerts()[-1]["kind"] == "unclassified message, model unavailable"
        assert not agent.tickets.tickets(12)
        # Exercise the disclosed safety gap, without editing rules or test labels.
        unmatched_critical = next(
            row["message"]
            for row in load_rows(ROOT / "eval/test.jsonl")
            if row["escalation"] and not critical_signal(row["message"])
        )
        missed = agent.turn(sender, unmatched_critical)
        assert "Cuéntame" in missed.message
        assert agent.operations.alerts()[-1]["kind"] == "unclassified message, model unavailable"
        assert agent.snapshot(sender).values["turn"]["classification"]["model_error"] == reason
        critical = agent.turn(sender, "No tengo internet")
        assert critical.awaiting == "confirmation"
        assert agent.operations.alerts() and not agent.tickets.tickets(12)
        assert agent.snapshot(sender).values["turn"]["classification"]["path"] == "critical_rule"
    finally:
        agent.close()


def test_raw_capture_retains_unmodified_provider_content_and_failure_reason():
    raw = '{ "intent": "billing", "confidence": 0.95, "reason": "¿Cuánto?" }'

    async def run():
        settings = Settings(
            model_backend="ollama", model_name="fake", model_base_url="http://localhost"
        )
        model = RecordingModel(settings)
        await model.client.aclose()
        model.client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"message": {"content": raw}})
            )
        )
        try:
            assert (await model.classify("¿Cuánto debo?")).intent.value == "billing"
            assert model.records[0]["raw_output"] == raw
            await model.client.aclose()
            model.client = httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json={"message": {"content": "invalid"}})
                )
            )
            with pytest.raises(ModelFailure, match="invalid_json"):
                await model.classify("¿Cuánto debo?")
            assert model.records[-1]["raw_output"] == "invalid"
            assert model.records[-1]["error"] == "invalid_json"
        finally:
            await model.close()

    asyncio.run(run())


def test_exact_recording_refuses_unrecorded_message():
    model = RecordedModel()
    model.record = {"message": "exact", "raw_output": "{}", "error": None}
    with pytest.raises(SystemExit, match="outside its exact recording"):
        asyncio.run(model.classify("different"))


def test_cli_record_rejects_stub_and_recalibration():
    with pytest.raises(SystemExit, match="live model"):
        main(["eval", "--mode", "suite", "--record", "unused.json"])
    with pytest.raises(SystemExit):
        main(["eval", "--mode", "suite", "--record", "unused.json", "--calibrate"])


def test_recording_fingerprints_match_frozen_policy(bundle):
    assert bundle["fingerprints"] == fingerprints(ROOT)


def test_live_record_command_emits_raw_bundle_and_reproducible_metrics(pre_blind_root, monkeypatch):
    from isp_support_agent.classification import rules

    tmp_path = pre_blind_root
    output = tmp_path / "results.json"
    previous = json.loads((tmp_path / "eval/recordings.json").read_text())
    output.write_text(json.dumps(previous["result"]))
    config = json.loads((ROOT / "eval/model-config.json").read_text())

    class FakeLiveModel(RecordingModel):
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
                rules(payload["message"]).model_dump(mode="json"), ensure_ascii=False
            )
            return self.raw

    monkeypatch.setattr(recordings, "RecordingModel", FakeLiveModel)
    settings = Settings(
        model_backend="ollama",
        model_name=config["model"],
        model_base_url="http://localhost",
        model_timeout=config["request_timeout_seconds"],
    )
    path = tmp_path / "captures.json"
    result = asyncio.run(recordings.record_suite(tmp_path, settings, path, output))
    recorded = json.loads(path.read_text())
    assert recorded["result"] == result
    assert len(recorded["records"]) == len(previous["records"])
    assert all(r["config_hash"] == recorded["config_hash"] for r in recorded["records"])
    validate_recordings(recorded, tmp_path)
    assert result == replay(deepcopy(result))
    assert recorded["baseline_kind"] == "graph-replay"
    assert (
        recorded["baseline"]["test"]["accuracy"]
        == asyncio.run(replay_graph(recorded, tmp_path))["splits"]["test"]["accuracy"]
    )
    wrong = settings.model_copy(update={"model_timeout": 1})
    with pytest.raises(SystemExit, match="settings differ"):
        asyncio.run(recordings.record_suite(tmp_path, wrong, path, output))
    wrong = settings.model_copy(update={"model_name": "wrong"})
    with pytest.raises(SystemExit, match="provider must match"):
        asyncio.run(recordings.record_suite(tmp_path, wrong, path, output))


def test_missing_input_and_unknown_format_explain_live_rerecord(bundle, tmp_path):
    with pytest.raises(SystemExit, match="Configure the live model"):
        validate_recordings(bundle, tmp_path)
    bundle["version"] = 999
    with pytest.raises(SystemExit, match="unsupported recording format"):
        validate_recordings(bundle, ROOT)


def test_cli_runs_offline_graph_gate(bundle, tmp_path, capsys):
    main(
        [
            "eval",
            "--graph-replay",
            str(ROOT / "eval/recordings.json"),
            "--gate",
            "--output",
            str(tmp_path / "graph.json"),
        ]
    )
    assert "test/graph:" in capsys.readouterr().out
    assert (
        json.loads((tmp_path / "graph.json").read_text())["splits"]["test"]["critical_recall"] == 1
    )


def test_changed_schema_or_baseline_is_rejected(bundle, monkeypatch):
    bundle["baseline"]["test"]["accuracy"] = 0
    with pytest.raises(SystemExit, match="baseline counts"):
        validate_recordings(bundle, ROOT)
    monkeypatch.setattr(
        recordings.IntentPrediction, "model_json_schema", classmethod(lambda cls: {"changed": True})
    )
    with pytest.raises(SystemExit, match="schema"):
        validate_recordings(bundle, ROOT)


@pytest.mark.parametrize("reason", ["timeout", "http_error"])
def test_raw_capture_records_provider_failure_without_fabricating_output(reason):
    async def run():
        settings = Settings(
            model_backend="ollama", model_name="fake", model_base_url="http://localhost"
        )
        model = RecordingModel(settings)
        await model.client.aclose()

        def fail(request):
            if reason == "timeout":
                raise httpx.ReadTimeout("not logged", request=request)
            raise httpx.ConnectError("not logged", request=request)

        model.client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
        try:
            with pytest.raises(ModelFailure, match=reason):
                await model.classify("¿Cuánto debo?")
            assert model.records == [
                {"message": "¿Cuánto debo?", "raw_output": None, "error": reason}
            ]
            recorded = RecordedModel()
            recorded.record = model.records[0]
            with pytest.raises(ModelFailure, match=reason):
                await recorded.classify("¿Cuánto debo?")
        finally:
            await model.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "extra", [["--calibrate"], ["--record", "unused.json"], ["--directory", "different"]]
)
def test_cli_replay_rejects_conflicting_or_unfrozen_options(extra):
    with pytest.raises(SystemExit):
        main(["eval", "--graph-replay", "unused.json", *extra])


@pytest.mark.parametrize("metric", ["critical_recall", "accuracy"])
def test_populated_blind_gate_rejects_regressions(bundle, metric):
    # Exercise the committed blind baseline, rather than substituting stub accuracy.
    result = json.loads((ROOT / "docs/eval-replay.json").read_text())
    assert result["splits"]["blind"]["count"] == len(bundle["result"]["splits"]["blind"]["rows"])
    gate_graph(result, bundle["baseline"])
    result["splits"]["blind"][metric] = (
        16 / 17
        if metric == "critical_recall"
        else bundle["baseline"]["blind"]["accuracy"] - 0.010001
    )
    with pytest.raises(SystemExit, match=f"blind/graph {metric.replace('_', ' ')}"):
        gate_graph(result, bundle["baseline"])
