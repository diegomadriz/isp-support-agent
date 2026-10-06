"""Live raw-response recording and fail-closed, offline graph regression replay."""

import hashlib
import json
from collections import Counter

from langgraph.checkpoint.memory import MemorySaver
from langsmith import tracing_context

from .adapters import MockTicketSystem, SimulatedNetwork, scenarios
from .eval_data import CORE_SPLITS, PROVENANCE_TEXT, blind_status, display_splits, provenance
from .evaluation import evaluate_suite, load_rows, score, write_results
from .graph import AgentContext, build_graph
from .language_models import JsonLanguageModel, ModelFailure, classification_prompt
from .models import Intent, IntentPrediction
from .operations import OperationStore
from .settings import Settings

RECORD_COMMAND = (
    "Configure the live model, then re-record with "
    "isp-agent eval --mode suite --record eval/recordings.json "
    "--output docs/eval-ollama.json (do not tune on held-out data)."
)
POLICY_FILES = (
    "src/isp_support_agent/classification.py",
    "src/isp_support_agent/language_models.py",
    "src/isp_support_agent/models.py",
    "src/isp_support_agent/settings.py",
    "src/isp_support_agent/fixtures/dev_examples.json",
    "eval/model-config.json",
    "eval/frozen.json",
    "eval/dev.jsonl",
    "eval/test.jsonl",
    "eval/known-cases.jsonl",
    "eval/blind.jsonl",
    "eval/options.jsonl",
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def fingerprints(root):
    try:
        source = {name: digest((root / name).read_bytes()) for name in POLICY_FILES}
    except OSError as error:
        stale(f"fingerprinted input is missing or unreadable: {error.filename}")
    return {
        "files": source,
        "prompt": digest(classification_prompt().encode()),
        "schema": digest(canonical(IntentPrediction.model_json_schema())),
    }


def stale(reason):
    raise SystemExit(f"Stale or invalid model recordings: {reason}. {RECORD_COMMAND}")


class RecordingModel(JsonLanguageModel):
    """Capture provider content verbatim before the existing parser validates it."""

    def __init__(self, settings):
        super().__init__(settings)
        self.records = []
        self.raw = None

    async def _request(self, system, payload, schema=None):
        self.raw = await super()._request(system, payload, schema)
        return self.raw

    async def classify(self, message):
        self.raw = None
        error = None
        try:
            return await super().classify(message)
        except Exception as exc:
            error = getattr(exc, "reason", type(exc).__name__)
            raise
        finally:
            self.records.append({"message": message, "raw_output": self.raw, "error": error})


def model_config(settings, metadata):
    # Credentials and endpoint location are intentionally excluded from public receipts.
    return {
        "backend": settings.model_backend,
        "model": settings.model_name,
        "confidence_threshold": settings.confidence_threshold,
        "request_timeout_seconds": settings.model_timeout,
        "ollama_options": metadata["ollama_options"],
    }


async def record_suite(root, settings, path, output):
    if settings.model_backend == "stub":
        raise SystemExit("Recordings require a live model; the stub is only a deterministic demo.")
    config_path = root / "eval/model-config.json"
    configured = json.loads(config_path.read_text())
    if (
        settings.model_name != configured["model"]
        or settings.model_backend != configured["backend"]
    ):
        raise SystemExit("Live provider must match eval/model-config.json before recording.")
    selected = model_config(settings, {"ollama_options": configured["ollama_options"]})
    if selected != {k: configured[k] for k in selected}:
        raise SystemExit("Live model settings differ from eval/model-config.json.")
    before = fingerprints(root)
    model = RecordingModel(settings)
    try:
        if settings.model_backend == "ollama":
            tags = await model.client.get(settings.model_base_url.rstrip("/") + "/api/tags")
            tags.raise_for_status()
            found = next(m for m in tags.json()["models"] if m["name"] == settings.model_name)
            if found["digest"] != configured["model_digest"]:
                raise SystemExit("Live model digest differs from eval/model-config.json.")
        result = await evaluate_suite(root / "eval", model, settings)
    finally:
        await model.close()
    actual = model_config(settings, result["metadata"])
    if actual != {k: configured[k] for k in actual}:
        raise SystemExit("Live model settings differ from eval/model-config.json.")
    if fingerprints(root) != before:
        stale("policy or data changed during live inference")
    # Provisional baseline from this run's combined results; replaced below by graph replay.
    baseline = {
        name: {key: split["modes"]["combined"][key] for key in ("correct", "count", "accuracy")}
        for name, split in result["splits"].items()
    }
    config_hash = digest(canonical(configured))
    records = []
    captured = iter(model.records)
    for name, split in result["splits"].items():
        for row, sample in zip(split["rows"], split["samples"], strict=True):
            raw = next(captured)
            if raw["message"] != row["message"] or raw["error"] != sample["error"]:
                raise RuntimeError("Inference capture order mismatch")
            records.append(dict(raw, split=name, id=row["id"], config_hash=config_hash))
    bundle = {
        "version": 1,
        "fingerprints": before,
        "config": configured,
        "config_hash": config_hash,
        "baseline": baseline,
        "baseline_source_sha256": digest(output.read_bytes()),
        "records": records,
        "records_sha256": digest(canonical(records)),
        "result": result,
    }
    graph_result = await replay_graph(bundle, root)
    bundle["baseline"] = {
        name: {key: split[key] for key in ("correct", "count", "accuracy")}
        for name, split in graph_result["splits"].items()
    }
    bundle["baseline_kind"] = "graph-replay"
    bundle["baseline_source_sha256"] = digest(canonical(bundle["baseline"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    write_results(result, output)
    return result


async def record_blind(root, settings, path, output):
    """Append new blind inferences without re-running or changing any existing response."""
    from .evaluation import collect, derive

    bundle = json.loads(path.read_text())
    validate_recordings(bundle, root, pending_blind=True)
    if "blind" in bundle["result"]["splits"]:
        raise SystemExit(
            "Blind responses already recorded; use the full live recording command to refresh."
        )
    blind_path = root / "eval/blind.jsonl"
    rows = load_rows(blind_path)
    if len(rows) < 30:
        raise SystemExit(
            "Supply at least 30 labeled blind messages before recording; no blind set has been evaluated yet."
        )
    ids = set()
    for row in rows:
        if (
            not isinstance(row.get("id"), str)
            or row["id"] in ids
            or not isinstance(row.get("message"), str)
            or not row["message"].strip()
            or len(row["message"]) > 2000
            or row.get("intent") not in {i.value for i in Intent}
            or type(row.get("escalation")) is not bool
            or row["escalation"] != (row["intent"] == "escalate")
        ):
            raise SystemExit("Invalid blind label, message, or duplicate id")
        ids.add(row["id"])
    if not any(row["escalation"] for row in rows):
        raise SystemExit("Blind messages must include critical reports to measure recall.")
    configured = bundle["config"]
    actual = model_config(settings, {"ollama_options": configured["ollama_options"]})
    if actual != {key: configured[key] for key in actual}:
        raise SystemExit("Live model settings differ from eval/model-config.json.")
    before = fingerprints(root)
    model = RecordingModel(settings)
    try:
        if settings.model_backend == "ollama":
            tags = await model.client.get(settings.model_base_url.rstrip("/") + "/api/tags")
            tags.raise_for_status()
            found = next(m for m in tags.json()["models"] if m["name"] == settings.model_name)
            if found["digest"] != configured["model_digest"]:
                raise SystemExit("Live model digest differs from eval/model-config.json.")
        samples = await collect(rows, model)
    finally:
        await model.close()
    if fingerprints(root) != before:
        stale("policy or data changed during blind inference")
    for row, raw in zip(rows, model.records, strict=True):
        if raw["message"] != row["message"]:
            raise RuntimeError("Inference capture order mismatch")
        bundle["records"].append(
            dict(raw, split="blind", id=row["id"], config_hash=bundle["config_hash"])
        )
    bundle["records_sha256"] = digest(canonical(bundle["records"]))
    threshold = settings.confidence_threshold
    bundle["result"]["splits"]["blind"] = {
        "sha256": digest(blind_path.read_bytes()),
        "rows": rows,
        "samples": samples,
        "modes": {
            m: derive(rows, samples, m, threshold) for m in ("rules-only", "model-only", "combined")
        },
    }
    blind_score = bundle["result"]["splits"]["blind"]["modes"]["combined"]
    bundle["baseline"]["blind"] = {k: blind_score[k] for k in ("correct", "count", "accuracy")}
    manifest_path = root / "eval/frozen.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"]["blind.jsonl"] = digest(blind_path.read_bytes())
    manifest["counts"]["blind"] = len(rows)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    bundle["fingerprints"] = fingerprints(root)
    metadata = bundle["result"]["metadata"]
    metadata.update(
        split_provenance=provenance(bundle["result"]["splits"]),
        blind_status=blind_status(bundle["result"]["splits"]),
        benchmark_inferences=len(bundle["records"]),
    )
    graph_result = await replay_graph(bundle, root)
    bundle["baseline"]["blind"] = {
        k: graph_result["splits"]["blind"][k] for k in ("correct", "count", "accuracy")
    }
    bundle["baseline_source_sha256"] = digest(canonical(bundle["baseline"]))
    # Preserve even a failing first blind result, so the score cannot disappear from view.
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    write_results(bundle["result"], output)
    graph_result = await replay_graph(bundle, root)
    write_graph_results(graph_result, root / "docs/eval-replay.json")
    gate_graph(graph_result, bundle["baseline"])
    return bundle["result"]


def validate_recordings(bundle, root, *, pending_blind=False):
    if bundle.get("version") != 1:
        stale("unsupported recording format")
    actual = fingerprints(root)
    if pending_blind:
        actual["files"]["eval/blind.jsonl"] = bundle["fingerprints"]["files"]["eval/blind.jsonl"]
    if bundle["fingerprints"] != actual:
        stale("guardrail code, prompt, schema, model config or eval data changed")
    configured = json.loads((root / "eval/model-config.json").read_text())
    config_hash = digest(canonical(configured))
    if bundle["config"] != configured or bundle["config_hash"] != config_hash:
        stale("model config hash mismatch")
    metadata = bundle["result"]["metadata"]
    if any(
        metadata[key] != configured[key]
        for key in (
            "backend",
            "model",
            "confidence_threshold",
            "request_timeout_seconds",
            "ollama_options",
        )
    ):
        stale("benchmark metadata differs from pinned model settings")
    if digest(canonical(bundle["records"])) != bundle["records_sha256"]:
        stale("raw response checksum mismatch")
    expected = []
    names = (
        *CORE_SPLITS,
        *(("blind",) if "blind" in bundle["result"]["splits"] else ()),
        *(("options",) if "options" in bundle["result"]["splits"] else ()),
    )
    if set(bundle["result"]["splits"]) != set(names):
        stale("unexpected evaluation split")
    for name in names:
        split = bundle["result"]["splits"][name]
        rows = load_rows(root / f"eval/{name}.jsonl")
        if rows != split["rows"] or len(split["samples"]) != len(rows):
            stale(f"{name} rows or samples do not match the frozen dataset")
        baseline = bundle["baseline"][name]
        if (
            baseline["count"] != len(rows)
            or not 0 <= baseline["correct"] <= len(rows)
            or baseline["accuracy"] != baseline["correct"] / len(rows)
        ):
            stale(f"{name} baseline counts/accuracy are inconsistent")
        expected.extend((name, row["id"], row["message"]) for row in rows)
    observed = [(r["split"], r["id"], r["message"]) for r in bundle["records"]]
    if observed != expected:
        stale("missing, duplicate, reordered or mismatched recording")
    for record in bundle["records"]:
        if record["config_hash"] != config_hash:
            stale("per-inference config hash mismatch")
        if not isinstance(record["raw_output"], str) and not record["error"]:
            stale("successful inference has no raw output")
    return configured


class RecordedModel(JsonLanguageModel):
    """Use the current JSON parser, with an exact raw response and no HTTP client."""

    def __init__(self):
        self.record = None
        self.calls = 0

    async def _request(self, system, payload, schema=None):
        if payload != {"message": self.record["message"]} or system != classification_prompt():
            stale("graph requested a message or prompt outside its exact recording")
        self.calls += 1
        if self.record["raw_output"] is None:
            raise ModelFailure(self.record["error"])
        return self.record["raw_output"]

    async def close(self):
        pass


async def replay_graph(bundle, root):
    # Offline replay must stay offline even with tracing enabled in the caller's env.
    with tracing_context(enabled=False):
        return await _replay_graph(bundle, root)


async def _replay_graph(bundle, root):
    """Run every recorded request through current parsing, policy and compiled subgraphs.

    Each row starts with an isolated, already-verified fixture identity. This is an
    intent/routing regression benchmark; identity and resume flows have separate tests.
    Ticket consent is never supplied. Escalation must reach the staff-alert node.
    """
    configured = validate_recordings(bundle, root)
    # Retain the recorded provider settings in graph context. The local placeholder
    # only satisfies URL validation; RecordedModel creates no HTTP client.
    settings = Settings(
        model_backend=configured["backend"],
        model_name=configured["model"],
        model_base_url="http://127.0.0.1",
        model_key="offline-replay" if configured["backend"] == "openai" else "",
        model_timeout=configured["request_timeout_seconds"],
        confidence_threshold=configured["confidence_threshold"],
        probe_latency=0,
        rewrite=False,
    )
    fixture = scenarios()["healthy"]
    operations = OperationStore()
    model = RecordedModel()
    context = AgentContext(
        MockTicketSystem(fixture),
        SimulatedNetwork(fixture["network"], latency=0),
        model,
        settings,
        operations,
    )
    graph = build_graph(MemorySaver())
    splits = {}
    try:
        # Validate all captures, including those skipped by guards or control routes.
        for name, split in bundle["result"]["splits"].items():
            records = [r for r in bundle["records"] if r["split"] == name]
            for record, sample in zip(records, split["samples"], strict=True):
                model.record = record
                error, parsed = None, None
                try:
                    parsed = (await model.classify(record["message"])).model_dump(mode="json")
                except Exception as exc:
                    error = getattr(exc, "reason", type(exc).__name__)
                if (
                    parsed != sample["prediction"]
                    or error != sample["error"]
                    or error != record["error"]
                ):
                    stale(f"raw/parsed inference mismatch for {name}/{record['id']}")
        model.calls = 0
        for name, split in bundle["result"]["splits"].items():
            rows, outputs, paths = split["rows"], [], Counter()
            records = [r for r in bundle["records"] if r["split"] == name]
            calls_before = model.calls
            for row, record in zip(rows, records, strict=True):
                model.record = record
                key = f"replay:{name}:{row['id']}"
                config = {"configurable": {"thread_id": key}}
                visited = []
                turn = {"sender_id": key, "request_id": key, "message": row["message"]}
                async for update in graph.astream(
                    {"session": {"customer_id": 12, "verified": True}, "turn": turn},
                    config,
                    context=context,
                    stream_mode="updates",
                ):
                    visited.extend(node for node in update if node != "__interrupt__")
                route = next(
                    (
                        node
                        for node in visited
                        if node
                        in {
                            "escalate",
                            "diagnostics",
                            "admin",
                            "billing",
                            "tickets",
                            "resolved",
                            "reset",
                            "thanks",
                            "unclear",
                            "welcome",
                        }
                    ),
                    None,
                )
                if route is None:
                    raise SystemExit(f"Graph did not route {key}")
                intent = {"diagnostics": "diagnose", "welcome": "unclear"}.get(route, route)
                if intent == "escalate" and not any(a["key"] == key for a in operations.alerts()):
                    raise SystemExit(f"Graph escalation did not create staff alert for {key}")
                snapshot = await graph.aget_state(config)
                classification = snapshot.values["turn"].get("classification", {})
                paths[classification.get("path", f"control:{route}")] += 1
                staff_alert = any(a["key"] == key for a in operations.alerts())
                if classification.get("model_error") and not staff_alert:
                    raise SystemExit(f"Unavailable classifier did not create staff alert for {key}")
                outputs.append({"intent": intent, "staff_alert": staff_alert})
            metrics = score(rows, outputs)
            metrics.update(
                paths=dict(paths),
                model_calls=model.calls - calls_before,
                critical_intent_hits=sum(
                    r["escalation"] and p["intent"] == "escalate"
                    for r, p in zip(rows, outputs, strict=True)
                ),
                unclassified_alerts=sum(
                    p["staff_alert"] and p["intent"] == "unclear" for p in outputs
                ),
            )
            splits[name] = metrics
    finally:
        operations.close()
    return {
        "config_hash": bundle["config_hash"],
        "baseline": bundle["baseline"],
        "split_provenance": provenance(splits),
        "blind_status": blind_status(splits),
        "splits": splits,
    }


def gate_graph(result, baseline):
    for name in ("dev", "test", *(("blind",) if "blind" in result["splits"] else ())):
        metrics = result["splits"][name]
        if metrics["critical_recall"] < 1:
            raise SystemExit(
                f"{name}/graph critical recall {metrics['critical_recall']:.1%}; required 100%"
            )
        floor = baseline[name]["accuracy"] - 0.01
        if metrics["accuracy"] + 1e-12 < floor:
            raise SystemExit(
                f"{name}/graph accuracy {metrics['accuracy']:.1%}; required at least {floor:.2%} "
                "(committed baseline minus 1 percentage point)"
            )


def write_graph_results(result, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Offline recorded-model graph replay",
        "",
        "Raw live responses are parsed and routed by the current compiled graph, with no model or network. Each message starts with a verified fixture identity; ticket consent is not supplied. Identity/resume flows are tested separately.",
        "",
        blind_status(result["splits"]) + " " + PROVENANCE_TEXT,
        "",
        f"Model config SHA-256: `{result['config_hash']}`.",
        "",
        "| Split | Graph accuracy | Critical recall | Model calls | Accuracy floor |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, r in display_splits(result["splits"]):
        floor = (
            f"{result['baseline'][name]['accuracy'] - 0.01:.2%}"
            if name in {"dev", "test", "blind"}
            else "not gated"
        )
        lines.append(
            f"| {name} | {r['accuracy']:.1%} ({r['correct']}/{r['count']}) | {r['critical_recall']:.1%} ({r['critical_hits']}/{r['critical_count']}) | {r['model_calls']} | {floor} |"
        )
    for name, r in display_splits(result["splits"]):
        lines.extend(["", f"{name} paths: `{json.dumps(r['paths'], sort_keys=True)}`.", ""])
    lines += [
        "The CI gate requires 100% critical recall on dev, synthetic test, and blind when recorded, and accuracy on each within one percentage point of its pinned graph-replay baseline. Known-cases are development design cases and are not gated. Checksums reject stale policy, prompt, schema, model settings or datasets; re-record with a live model after such changes. This is a regression gate, not a guarantee of future live inference or arbitrary-message safety.",
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
