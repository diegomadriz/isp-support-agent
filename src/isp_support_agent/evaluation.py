"""Reproducible metrics from recorded inference, with immutable split fingerprints."""

import hashlib
import json
import math
import time
from collections import Counter

from .classification import apply_policy, prediction, rules
from .eval_data import CORE_SPLITS, PROVENANCE_TEXT, blind_status, display_splits, provenance
from .language_models import StubLanguageModel, classification_prompt
from .models import Intent, IntentPrediction


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values, quantile):
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * quantile) - 1)], 3) if ordered else 0


def score(rows, outputs):
    confusion = Counter((r["intent"], p["intent"]) for r, p in zip(rows, outputs, strict=True))
    labels = [i.value for i in Intent]
    critical = sum(r["escalation"] for r in rows)
    hits = sum(
        r["escalation"] and (p["intent"] == "escalate" or p.get("staff_alert", False))
        for r, p in zip(rows, outputs, strict=True)
    )
    correct = sum(r["intent"] == p["intent"] for r, p in zip(rows, outputs, strict=True))
    metrics = {}
    for label in labels:
        tp = confusion[label, label]
        predicted = sum(confusion[a, label] for a in labels)
        expected = sum(confusion[label, b] for b in labels)
        metrics[label] = {
            "precision": tp / predicted if predicted else 0,
            "recall": tp / expected if expected else 0,
            "support": expected,
        }
    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "critical_count": critical,
        "critical_hits": hits,
        "critical_recall": hits / critical if critical else 0,
        "per_intent": metrics,
        "confusion_matrix": {a: {b: confusion[a, b] for b in labels} for a in labels},
        "misses": [
            {
                "id": r["id"],
                "message": r["message"],
                "expected": r["intent"],
                "predicted": p["intent"],
            }
            for r, p in zip(rows, outputs, strict=True)
            if r["intent"] != p["intent"]
        ],
    }


def derive(rows, samples, mode, threshold):
    outputs, paths, fallback, errors, latency = [], Counter(), Counter(), Counter(), []
    for row, sample in zip(rows, samples, strict=True):
        if mode == "rules-only":
            result = rules(row["message"])
            output = result.model_dump(mode="json")
            paths["reference_rules"] += 1
        else:
            result = (
                IntentPrediction.model_validate(sample["prediction"])
                if sample["prediction"]
                else prediction(Intent.UNCLEAR, 0, "Model request failed")
            )
            policy = apply_policy(row["message"], result, threshold, guards=mode == "combined")
            output = policy.prediction.model_dump(mode="json")
            paths[policy.path] += 1
            if policy.path == "model":
                latency.append(sample["latency_ms"])
                if sample["error"]:
                    errors[sample["error"]] += 1
                    fallback["model_error"] += 1
                elif policy.fallback_reason:
                    fallback[policy.fallback_reason] += 1
        outputs.append(output)
    result = score(rows, outputs)
    result.update(
        paths=dict(paths),
        fallback_reasons=dict(fallback),
        model_errors=dict(errors),
        clarification_rate=sum(p["intent"] == "unclear" for p in outputs) / len(outputs),
        latency_ms={"p50": percentile(latency, 0.5), "p95": percentile(latency, 0.95)},
        model_calls=paths["model"],
    )
    return result


async def collect(rows, model):
    samples = []
    for index, row in enumerate(rows, 1):
        start = time.perf_counter()
        try:
            output = await model.classify(row["message"])
            sample = {"prediction": output.model_dump(mode="json"), "error": None}
        except Exception as error:
            sample = {"prediction": None, "error": getattr(error, "reason", type(error).__name__)}
        sample["latency_ms"] = (
            0
            if isinstance(model, StubLanguageModel)
            else round((time.perf_counter() - start) * 1000, 3)
        )
        samples.append(sample)
        if index % 20 == 0 and not isinstance(model, StubLanguageModel):
            print(f"  recorded {index}/{len(rows)} inferences", flush=True)
    return samples


async def evaluate_suite(directory, model, settings, *, calibrate=False):
    manifest = json.loads((directory / "frozen.json").read_text())
    splits = {}
    threshold = settings.confidence_threshold
    calibration = []
    for name in (*CORE_SPLITS, "blind", "options"):
        path = directory / f"{name}.jsonl"
        if name == "options" and not path.exists():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # Data provenance check before any inference; test is never edited to repair a score.
        if name != "options":
            assert digest == manifest["sha256"][f"{name}.jsonl"], f"Frozen {name} changed"
        rows = load_rows(path)
        if not rows:
            continue
        print(f"Evaluating {name}: {len(rows)} messages", flush=True)
        samples = await collect(rows, model)
        if name == "dev":
            for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                r = derive(rows, samples, "combined", t)
                calibration.append(
                    {
                        "threshold": t,
                        "accuracy": r["accuracy"],
                        "clarification_rate": r["clarification_rate"],
                    }
                )
            if calibrate:
                best = max(
                    calibration,
                    key=lambda r: (
                        r["accuracy"],
                        -r["clarification_rate"],
                        -abs(r["threshold"] - 0.7),
                    ),
                )
                threshold = best["threshold"]
        splits[name] = {
            "sha256": digest,
            "rows": rows,
            "samples": samples,
            "modes": {
                m: derive(rows, samples, m, threshold)
                for m in ("rules-only", "model-only", "combined")
            },
        }
    return {
        "metadata": {
            "backend": settings.model_backend,
            "model": settings.model_name or "deterministic-stub",
            "confidence_threshold": threshold,
            "request_timeout_seconds": settings.model_timeout,
            "prompt_sha256": hashlib.sha256(classification_prompt().encode()).hexdigest(),
            "ollama_options": {
                "think": False,
                "temperature": 0,
                "seed": 7,
                "num_ctx": 4096,
                "num_predict": 512,
            }
            if settings.model_backend == "ollama"
            else None,
            "authorship": PROVENANCE_TEXT,
            "split_provenance": provenance(splits),
            "blind_status": blind_status(splits),
            "benchmark_inferences": sum(len(s["samples"]) for s in splits.values()),
            "calibration_used": calibrate,
            "latency_mode": "not_measured_stub"
            if isinstance(model, StubLanguageModel)
            else "wall_clock",
        },
        "dev_calibration": calibration,
        "splits": splits,
    }


def replay(result):
    result["metadata"].update(
        split_provenance=provenance(result["splits"]),
        blind_status=blind_status(result["splits"]),
    )
    threshold = result["metadata"]["confidence_threshold"]
    for split in result["splits"].values():
        split["modes"] = {
            m: derive(split["rows"], split["samples"], m, threshold) for m in split["modes"]
        }
    return result


def render_results(result, label=None):
    meta = result["metadata"]
    if meta["backend"] == "stub":
        lines = [
            "# Reference rules evaluation",
            "",
            "These scores measure reference rules only. The offline stub is a deterministic demo fixture, not a learned classifier.",
            "",
            blind_status(result["splits"]) + " " + PROVENANCE_TEXT,
            "",
            "| Split | Rules accuracy | Critical recall |",
            "| --- | --- | --- |",
        ]
        for name, split in display_splits(result["splits"]):
            r = split["modes"]["rules-only"]
            lines.append(
                f"| {name} | {r['accuracy']:.1%} ({r['correct']}/{r['count']}) | {r['critical_recall']:.1%} ({r['critical_hits']}/{r['critical_count']}) |"
            )
        lines += [
            "",
            "Full confusion matrices and per-intent metrics are in the adjacent JSON. CI gates recorded real-model graph replay instead of this reference baseline.",
        ]
        return "\n".join(lines) + "\n"
    lines = [
        f"# Evaluation: {meta['backend']}/{meta['model']}",
        "",
        f"Threshold: {meta['confidence_threshold']}; timeout: {meta['request_timeout_seconds']} seconds. Prompt SHA-256: `{meta['prompt_sha256']}`.",
        "",
        blind_status(result["splits"]) + " " + PROVENANCE_TEXT,
        "",
        "Model-only uses the structured model output and confidence threshold without safety rules. Combined uses safety rules OR the model critical flag, then the confidence threshold. The deterministic stub is a rules stand-in, not learned inference.",
        "",
        "| Split | Mode | Accuracy | Critical recall | Model calls | Clarify | p50 / p95 ms |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, split in display_splits(result["splits"]):
        for mode, r in split["modes"].items():
            lines.append(
                f"| {name} | {mode} | {r['accuracy']:.1%} ({r['correct']}/{r['count']}) | {r['critical_recall']:.1%} ({r['critical_hits']}/{r['critical_count']}) | {r['model_calls']} | {r['clarification_rate']:.1%} | {r['latency_ms']['p50']} / {r['latency_ms']['p95']} |"
            )
    lines += [
        "",
        f"Benchmark ran {meta['benchmark_inferences']} inferences to obtain model-only scores; combined would call the model only for its counted model path. No noncritical cheap rule path is enabled.",
        "",
        "## Dev threshold sensitivity",
        "",
        "| Threshold | Accuracy | Clarification |",
        "| --- | --- | --- |",
    ]
    for c in result["dev_calibration"]:
        lines.append(f"| {c['threshold']} | {c['accuracy']:.1%} | {c['clarification_rate']:.1%} |")
    if len({(c["accuracy"], c["clarification_rate"]) for c in result["dev_calibration"]}) == 1:
        lines += [
            "",
            "The dev curve is flat from 0.3 through 0.9. Confidence supplies no observed threshold-selection signal on these messages; 0.7 is retained as the configured default, not an empirically optimal threshold.",
        ]
    for name, split in display_splits(result["splits"]):
        for mode, r in split["modes"].items():
            lines += [
                "",
                f"## {name}: {mode}",
                "",
                f"Paths: `{json.dumps(r['paths'], sort_keys=True)}`. Fallbacks: `{json.dumps(r['fallback_reasons'], sort_keys=True)}`. Model errors: `{json.dumps(r['model_errors'], sort_keys=True)}`.",
                "",
                "| Intent | Precision | Recall | Support |",
                "| --- | --- | --- | --- |",
            ]
            for intent, stats in r["per_intent"].items():
                lines.append(
                    f"| {intent} | {stats['precision']:.1%} | {stats['recall']:.1%} | {stats['support']} |"
                )
            cols = list(r["confusion_matrix"])
            lines += [
                "",
                "Confusion matrix: expected rows, predicted columns.",
                "",
                "| Expected | " + " | ".join(cols) + " |",
                "| --- | " + " | ".join(["---"] * len(cols)) + " |",
            ]
            lines += [
                "| " + a + " | " + " | ".join(str(counts[b]) for b in cols) + " |"
                for a, counts in r["confusion_matrix"].items()
            ]
    return "\n".join(lines) + "\n"


def write_results(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    path.with_suffix(".md").write_text(render_results(result))


def gate(result):
    raise SystemExit(
        "Use --graph-replay eval/recordings.json --gate; rules/stub is not the safety gate"
    )
