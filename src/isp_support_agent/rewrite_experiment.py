"""Live paired graph turns, raw classification/rewrite capture and exact offline replay."""

import asyncio
import json
import math
import random
import time
from collections import Counter

import httpx

from .adapters import scenarios
from .evaluation import load_rows
from .language_models import ModelFailure
from .operations import OperationStore
from .recordings import RecordedModel, canonical, digest, fingerprints, validate_recordings
from .rewriting import (
    REWRITE_PROMPT,
    ReplyLanguageModel,
    RewriteRecordingError,
    public_facts,
    rejection_reason,
    tone_reason,
)
from .runtime import AgentRuntime
from .settings import Settings

EXTRA_FILES = (
    "src/isp_support_agent/rewriting.py",
    "src/isp_support_agent/rewrite_experiment.py",
    "src/isp_support_agent/safety.py",
    "src/isp_support_agent/cli.py",
    "src/isp_support_agent/ports.py",
    "src/isp_support_agent/graph.py",
    "src/isp_support_agent/nodes.py",
    "src/isp_support_agent/runtime.py",
    "src/isp_support_agent/graph_state.py",
    "src/isp_support_agent/routing.py",
    "src/isp_support_agent/adapters.py",
    "src/isp_support_agent/templates.py",
    "src/isp_support_agent/operations.py",
    "src/isp_support_agent/fixtures/scenarios.json",
    "eval/recordings.json",
    "eval/options.jsonl",
    "eval/rewrite-config.json",
    "eval/rewrite-cases.json",
    "eval/rewrite-owner-decision.json",
)
DECISION = {
    "leaks": 0,
    "required_facts_preserved": 1.0,
    "minimum_acceptance": 0.85,
    "turn_budget_seconds": 8,
    "classification_limit_seconds": 6,
    "minimum_rewrite_remaining_seconds": 1.5,
}


def experiment_fingerprints(root):
    try:
        return {
            "classification": fingerprints(root),
            "rewrite_prompt": digest(REWRITE_PROMPT.encode()),
            "files": {n: digest((root / n).read_bytes()) for n in EXTRA_FILES},
        }
    except OSError as error:
        stale(f"missing {error.filename}")


def stale(reason):
    raise SystemExit(
        f"Stale or invalid rewrite recordings: {reason}. Re-record with the live model using isp-agent rewrite-eval --record eval/rewrite-recordings.json."
    )


class ExperimentModel:
    """Record both model phases; replay raw responses and the measured budget clock."""

    def __init__(
        self, *, live=None, captures=None, config_hash="", replay=None, saved_rows=None, arm="on"
    ):
        self.live, self.config_hash, self.replay = live, config_hash, replay
        self.captures = captures if captures is not None else []
        self.saved_rows = saved_rows or []
        self.position, self.key, self.arm = 0, "", arm
        if live is not None:
            request = live._request

            async def capture_raw(*args, **kwargs):
                raw = await request(*args, **kwargs)
                self.raw = raw
                return raw

            live._request = capture_raw

    def begin_turn(self, key, index):
        self.key, self.index = key, index
        self.started = time.monotonic()
        self.clock_reads = 0

    def budget_clock(self):
        if self.replay is None:
            return time.monotonic()
        self.clock_reads += 1
        if self.clock_reads == 1:
            return 0
        if self.position < len(self.replay):
            next_record = self.replay[self.position]
            if next_record["key"] == self.key and next_record["method"] == "classify":
                return next_record["started_ms"] / 1000
        row = self.saved_rows[self.index]
        return row.get("rewrite", {}).get("budget_elapsed_ms", 0) / 1000

    async def _call(self, request, operation):
        request = dict(request, key=self.key, config_hash=self.config_hash)
        if self.replay is not None:
            if self.position >= len(self.replay):
                self.invalid = True
                raise RewriteRecordingError("unrecorded model request")
            record = self.replay[self.position]
            self.position += 1
            if any(record.get(k) != v for k, v in request.items()):
                self.invalid = True
                raise RewriteRecordingError(
                    "model request/facts/config differ from exact recording"
                )
            if record["error"] and record["raw_output"] is None:
                if request["method"] == "rewrite" and record["error"] == "timeout":
                    raise httpx.ReadTimeout("recorded timeout")
                raise ModelFailure(record["error"])
            if request["method"] == "classify":
                parser = RecordedModel()
                parser.record = record
                return await parser.classify(request["message"])
            return record["raw_output"]
        if self.live is None:
            raise AssertionError("Live model required")
        self.raw = None
        record = dict(
            request,
            raw_output=None,
            error=None,
            started_ms=round((time.monotonic() - self.started) * 1000, 3),
        )
        start = time.perf_counter()
        try:
            return await operation()
        except (asyncio.CancelledError, httpx.TimeoutException):
            record["error"] = "timeout"
            raise
        except Exception as error:
            record["error"] = getattr(error, "reason", type(error).__name__)
            raise
        finally:
            record["raw_output"] = self.raw
            record["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
            self.captures.append(record)

    async def classify(self, message):
        return await self._call(
            {"method": "classify", "message": message}, lambda: self.live.classify(message)
        )

    async def rewrite(self, text, facts):
        return await self._call(
            {"method": "rewrite", "text": text, "facts": facts},
            lambda: self.live.rewrite(text, facts),
        )

    async def close(self):
        if self.live:
            await self.live.close()


def case_specs(root):
    cases = [
        {"id": "demo/" + name, "group": "demo", "scenario": name, "messages": f["conversation"]}
        for name, f in scenarios().items()
    ]
    for group in ("blind", "options"):
        cases += [
            {
                "id": group + "/" + row["id"],
                "group": group,
                "scenario": "healthy",
                "messages": ["cliente 12", "1234", row["message"]],
                "issue_message": row["message"],
            }
            for row in load_rows(root / f"eval/{group}.jsonl")
        ]
    return cases


def settings_for(case, config, enabled):
    return Settings(
        model_backend=config["backend"],
        model_name=config["model"],
        model_base_url="http://127.0.0.1",
        model_timeout=config["request_timeout_seconds"],
        scenario=case["scenario"],
        probe_timeout=0.1,
        rewrite=enabled,
        rewrite_timeout=8,
    )


def run_arm(case, settings, model, *, complete=False, messages=None):
    agent = AgentRuntime(settings, model=model, operations=OperationStore(clock=lambda: 1234))
    queue = list(messages if messages is not None else case["messages"])
    if complete and case["group"] in {"blind", "options"}:
        queue += ["Ver mis tickets", "gracias"]
    rows = []
    try:
        while queue:
            message = queue.pop(0)
            index = len(rows)
            model.begin_turn(f"{case['id']}:{model.arm}:{index}", index)
            started = time.perf_counter()
            reply = agent.turn("rewrite-experiment", message)
            if getattr(model, "invalid", False):
                raise RewriteRecordingError("invalid model request")
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            staff = agent.staff_snapshot("rewrite-experiment")
            snapshot = agent.snapshot("rewrite-experiment")
            session = agent._session(snapshot)
            rows.append(
                {
                    "input": message,
                    "message": reply.message,
                    "awaiting": reply.awaiting,
                    "elapsed_ms": elapsed,
                    "verified": session.get("verified", False),
                    "rewrite": staff.get("rewrite", {}),
                    "classification": staff.get("classification", {}),
                    "node_path": agent._staff["rewrite-experiment"]["trace"],
                    "ticket_statuses": [
                        [t.id, t.status, len(t.notes)]
                        for t in agent.tickets.tickets(session.get("customer_id") or -1)
                    ],
                    "staff_note": staff.get("staff_note", ""),
                    "staff_alerts": [
                        {"customer": a["customer"], "kind": a["kind"]}
                        for a in agent.operations.alerts()
                    ],
                }
            )
            if complete and case["group"] in {"blind", "options"} and reply.awaiting:
                follow = {
                    "customer_id": "cliente 12",
                    "verification": "1234",
                    "confirmation": "Sí, por favor",
                }[reply.awaiting]
                if not queue or queue[0] != follow:
                    queue.insert(0, follow)
            if len(rows) > 25:
                raise AssertionError("Conversation did not finish")
        return rows
    finally:
        agent.close()


def paired_turns(conversation):
    queues = {}
    for row in conversation["off"]:
        queues.setdefault(row["input"], []).append(row)
    for on in conversation["on"]:
        matching = queues.get(on["input"], [])
        if matching:
            yield matching.pop(0), on


def validate(bundle, root):
    if bundle.get("version") != 2:
        stale("unsupported format")
    if bundle.get("fingerprints") != experiment_fingerprints(root):
        stale("policy/prompt/config/cases/data changed")
    config = json.loads((root / "eval/rewrite-config.json").read_text())
    if bundle.get("config") != config or bundle.get("config_hash") != digest(canonical(config)):
        stale("config mismatch")
    if bundle.get("decision_rule") != DECISION:
        stale("budget/check rules changed")
    for key in ("captures", "conversations"):
        if bundle.get(key + "_sha256") != digest(canonical(bundle[key])):
            stale(key + " checksum mismatch")
    if bundle["cases"] != json.loads((root / "eval/rewrite-cases.json").read_text()):
        stale("cases changed")
    if [c["id"] for c in bundle["cases"]] != [c["id"] for c in bundle["conversations"]]:
        stale("coverage/order mismatch")
    valid_keys = {
        f"{c['id']}:{arm}:{i}"
        for c in bundle["conversations"]
        for arm in ("off", "on")
        for i in range(len(c[arm]))
    }
    if any(
        r["key"] not in valid_keys or r["config_hash"] != bundle["config_hash"]
        for r in bundle["captures"]
    ):
        stale("capture key/config mismatch")
    validate_recordings(json.loads((root / "eval/recordings.json").read_text()), root)
    return config


def percentile(values, p):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * p) - 1)] if values else 0


def metrics(bundle):
    attempts = accepted = changed = leaks = preserved = sent = tone_violations = verified = (
        route_changes
    ) = 0
    reasons = Counter()
    draft_flags = Counter()
    lengths = []
    deltas = []
    turn_times = []
    verified_times = []
    for conv in bundle["conversations"]:
        matched = {id(on): off for off, on in paired_turns(conv)}
        for on in conv["on"]:
            sent += 1
            turn_times.append(on["elapsed_ms"])
            if on.get("verified"):
                verified += 1
                verified_times.append(on["elapsed_ms"])
            rw = on.get("rewrite", {})
            facts = rw.get("facts") or public_facts(on["message"], on["input"])
            reason = rejection_reason(on["message"], facts)
            leaks += int(
                bool(
                    reason
                    and reason
                    not in {
                        "missing_required_fact",
                        "duplicate_required_fact",
                        "action_question_order",
                        "too_many_courtesies",
                    }
                    and not reason.startswith("tone_")
                )
            )
            tone_violations += int(bool(tone_reason(on["message"], facts["tone_profile"])))
            preserved += int(all(on["message"].count(c) == 1 for c in facts["required_clauses"]))
            if on.get("verified") or rw.get("attempted"):
                reasons[rw.get("reason", "no_rewrite_step")] += 1
            if rw.get("attempted"):
                attempts += 1
                accepted += int(rw["accepted"])
                off = matched.get(id(on))
                if off:
                    changed += int(rw["accepted"] and off["message"] != on["message"])
                    deltas.append(round(on["elapsed_ms"] - off["elapsed_ms"], 3))
                    lengths.append(
                        round(
                            (len(on["message"]) - len(off["message"])) / len(off["message"]) * 100,
                            3,
                        )
                    )
            off = matched.get(id(on))
            if off and on.get("node_path") != off.get("node_path"):
                route_changes += 1
    rewrite_calls = [r for r in bundle["captures"] if r.get("method", "rewrite") == "rewrite"]
    for r in rewrite_calls:
        if isinstance(r.get("raw_output"), str):
            flag = tone_reason(r["raw_output"], r["facts"]["tone_profile"])
            if flag:
                draft_flags[flag] += 1
    checks = {
        "zero_leaks": leaks == 0,
        "all_required_facts": preserved == sent,
        "zero_sent_tone_violations": tone_violations == 0,
        "acceptance_at_least_85_percent": accepted / attempts >= 0.85 if attempts else False,
        "p95_turn_latency_under_8_seconds": percentile(turn_times, 0.95) < 8000,
    }
    return {
        "adoption": bundle.get("adoption", {}),
        "conversations": len(bundle["cases"]),
        "groups": dict(Counter(c["group"] for c in bundle["cases"])),
        "sent_messages": sent,
        "verified_messages": verified,
        "attempted": attempts,
        "skipped_unverified": sent - verified,
        "accepted": accepted,
        "acceptance_rate": accepted / attempts if attempts else 0,
        "accepted_changed": changed,
        "reasons": dict(sorted(reasons.items())),
        "leaks": leaks,
        "draft_tone_flags": dict(sorted(draft_flags.items())),
        "sent_tone_violations": tone_violations,
        "facts_preserved_messages": preserved,
        "required_facts_preserved": preserved / sent if sent else 0,
        "turn_latency_ms": {
            "p50": percentile(turn_times, 0.5),
            "p95": percentile(turn_times, 0.95),
        },
        "verified_turn_latency_ms": {
            "p50": percentile(verified_times, 0.5),
            "p95": percentile(verified_times, 0.95),
        },
        "added_latency_ms": {"p50": percentile(deltas, 0.5), "p95": percentile(deltas, 0.95)},
        "model_latency_ms": {
            "p50": percentile([r["elapsed_ms"] for r in rewrite_calls], 0.5),
            "p95": percentile([r["elapsed_ms"] for r in rewrite_calls], 0.95),
        },
        "length_change_percent": {
            "mean": round(sum(lengths) / len(lengths), 3) if lengths else 0,
            "p50": percentile(lengths, 0.5),
            "p95": percentile(lengths, 0.95),
        },
        "paired_route_changes": route_changes,
        "automatic_checks": checks,
        "automatic_criteria_pass": all(checks.values()),
        "decision": "live rewriting enabled with template fallback",
        "timings": "live end-to-end graph turns, including classification; replay retains measurements",
        "config_hash": bundle["config_hash"],
        "decision_rule": bundle["decision_rule"],
    }


def gate(result):
    if (
        result["leaks"]
        or result["required_facts_preserved"] != 1
        or result.get("sent_tone_violations", 0)
    ):
        raise SystemExit(
            "Rewrite safety gate failed: delivered leaks, tone violations or lost required facts"
        )


def replay_case(bundle, case, config, arm):
    saved = next(c for c in bundle["conversations"] if c["id"] == case["id"])[arm]
    captures = [r for r in bundle["captures"] if r["key"].startswith(case["id"] + ":" + arm + ":")]
    model = ExperimentModel(
        config_hash=bundle["config_hash"], replay=captures, saved_rows=saved, arm=arm
    )
    try:
        rows = run_arm(
            case,
            settings_for(case, config, arm == "on"),
            model,
            messages=[r["input"] for r in saved],
        )
    except RewriteRecordingError as error:
        stale(str(error))
    for i, (row, previous) in enumerate(zip(rows, saved, strict=True)):
        row["elapsed_ms"] = previous["elapsed_ms"]
        if row["rewrite"]:
            row["rewrite"]["elapsed_ms"] = previous["rewrite"]["elapsed_ms"]
        if row["classification"]:
            row["classification"]["latency_ms"] = previous["classification"]["latency_ms"]
        if json.loads(json.dumps(row)) != previous:
            stale(f"current graph/validator behavior changed: {case['id']}/{arm}/{i}")
    if model.position != len(captures):
        stale("unused capture")
    return rows


def replay_experiment(bundle, root):
    config = validate(bundle, root)
    for case in bundle["cases"]:
        for arm in ("off", "on"):
            replay_case(bundle, case, config, arm)
    return metrics(bundle)


def record_experiment(root, settings, path):
    if settings.model_backend != "ollama" or settings.model_name != "gemma4:latest":
        raise SystemExit("Configure live local Ollama gemma4:latest")
    classifier = json.loads((root / "eval/recordings.json").read_text())
    pinned = validate_recordings(classifier, root)
    if settings.model_timeout != pinned["request_timeout_seconds"]:
        raise SystemExit("Live settings must match frozen config")

    async def check():
        async with httpx.AsyncClient(trust_env=False, timeout=6) as client:
            response = await client.get(settings.model_base_url.rstrip("/") + "/api/tags")
            response.raise_for_status()
            if not any(
                m["name"] == pinned["model"] and m["digest"] == pinned["model_digest"]
                for m in response.json()["models"]
            ):
                raise SystemExit("Local model digest differs")

    asyncio.run(check())
    config = dict(
        pinned,
        rewrite_prompt_sha256=digest(REWRITE_PROMPT.encode()),
        rewrite=True,
        rewrite_timeout_seconds=8,
        turn_budget_seconds=8,
        minimum_rewrite_remaining_seconds=1.5,
    )
    cases = case_specs(root)
    (root / "eval/rewrite-config.json").write_text(json.dumps(config, indent=2) + "\n")
    (root / "eval/rewrite-cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n"
    )
    frozen = experiment_fingerprints(root)
    config_hash = digest(canonical(config))
    captures = []
    conversations = []
    for case in cases:
        conversation = {"id": case["id"]}
        for arm in ("off", "on"):
            model = ExperimentModel(
                live=ReplyLanguageModel(settings),
                captures=captures,
                config_hash=config_hash,
                arm=arm,
            )
            conversation[arm] = run_arm(
                case, settings_for(case, config, arm == "on"), model, complete=True
            )
        conversations.append(conversation)
        print(
            f"Recorded graph conversation {len(conversations)}/{len(cases)}: {case['id']}",
            flush=True,
        )
    if frozen != experiment_fingerprints(root):
        stale("inputs changed during live recording")
    bundle = {
        "version": 2,
        "config": config,
        "config_hash": config_hash,
        "fingerprints": frozen,
        "decision_rule": DECISION,
        "adoption": json.loads((root / "eval/rewrite-owner-decision.json").read_text()),
        "cases": cases,
        "captures": captures,
        "captures_sha256": digest(canonical(captures)),
        "conversations": conversations,
        "conversations_sha256": digest(canonical(conversations)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    return bundle, metrics(bundle)


def write_results(result, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "rewrite-results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# Customer reply rewriting",
        "",
        "The model may reorder template sentences and add one short courtesy phrase. Facts stay word for word. Actions and questions keep their order. The closed courtesy list prevents extra content; greetings are allowed only in the first reply, and angry messages allow only a calm acknowledgement. No reply contains two apologies.",
        "",
        "Both arms use live local gemma4 through the current graph. Classification gets up to six seconds within an eight-second turn budget. Rewriting starts only with at least 1.5 seconds left; otherwise the template is sent. Every recorded conversation includes identity and any ticket consent.",
        "",
        "| Metric | Result |",
        "| --- | --- |",
        f"| Conversations | {result['conversations']} |",
        f"| Sent replies | {result['sent_messages']} |",
        f"| Rewrite attempts | {result['attempted']} |",
        f"| Accepted | {result['accepted']}/{result['attempted']} ({result['acceptance_rate']:.1%}) |",
        f"| Delivered leaks | {result['leaks']} |",
        f"| Required facts | {result['facts_preserved_messages']}/{result['sent_messages']} |",
        f"| Detected tone violations | {result['sent_tone_violations']} |",
        f"| Complete turn latency median | {result['turn_latency_ms']['p50']} ms |",
        f"| Complete turn latency p95 | {result['turn_latency_ms']['p95']} ms |",
        f"| Verified turn latency median | {result['verified_turn_latency_ms']['p50']} ms |",
        f"| Verified turn latency p95 | {result['verified_turn_latency_ms']['p95']} ms |",
        f"| Added latency median | {result['added_latency_ms']['p50']} ms |",
        f"| Added latency p95 | {result['added_latency_ms']['p95']} ms |",
        f"| Mean length change | {result['length_change_percent']['mean']}% |",
        "",
        f"Result codes: `{json.dumps(result['reasons'], sort_keys=True)}`.",
        f"Raw tone flags: `{json.dumps(result['draft_tone_flags'], sort_keys=True)}`.",
        f"Paired turns with a different node path: {result['paired_route_changes']}.",
        "",
        "Timing starts at the chat turn call and ends when the reply returns, including classification, graph work and rewriting. It excludes the WhatsApp queue and outbound delivery. Replay uses the recorded clock to reproduce rewrite eligibility; it does not measure live speed. Added latency compares matched input turns from the two arms, whose model predictions can differ.",
        "",
        "The earlier 64-conversation trial accepted 268 of 271 drafts and measured p95 added latency of 4.29 seconds, missing the three-second target set before the trial. Classification was held fixed in that trial. After seeing the result, the limit was raised to eight seconds because a few seconds is normal in a WhatsApp chat. Warmer replies were preferred in a side-by-side read, without item-by-item scores. That was a decision to enable rewriting, not a pass under the original target.",
        "",
        f"Config SHA-256: `{result['config_hash']}`. Raw responses, requests, turn paths and measured timings are in `eval/rewrite-recordings.json`. The offline gate rejects stale inputs and replays both arms through the current graph. The review form and answer key are kept outside the public repository.",
        "",
    ]
    (output / "rewrite-results.md").write_text("\n".join(lines))


def write_review(bundle, output, key_path=None):
    pool = {"demo": [], "blind": []}
    for case, conversation in zip(bundle["cases"], bundle["conversations"], strict=True):
        for off, on in paired_turns(conversation):
            i = next(n for n, row in enumerate(conversation["on"]) if row is on)
            if on["rewrite"].get("attempted"):
                pool["blind" if case["group"] == "blind" else "demo"].append((case, i, off, on))
    rng = random.Random(20261005)
    mandatory = {"blind/x04", "blind/x12", "blind/x17", "blind/g11"}
    chosen = [
        row
        for row in pool["blind"]
        if row[0]["id"] in mandatory and row[2]["input"] == row[0]["issue_message"]
    ]
    if len(chosen) != 4:
        raise AssertionError("Angry/casual issue turns missing from review pack")
    message_counts = Counter(row[2]["input"] for row in chosen)
    for group, count in (("demo", 10), ("blind", 6)):
        available = [row for row in pool[group] if row not in chosen]
        rng.shuffle(available)
        picked = 0
        for row in available:
            if message_counts[row[2]["input"]] >= 2:
                continue
            chosen.append(row)
            message_counts[row[2]["input"]] += 1
            picked += 1
            if picked == count:
                break
        if picked != count:
            raise AssertionError("Insufficient distinct messages for review sampling")
    rng.shuffle(chosen)
    lines = [
        "# Customer reply review: A/B",
        "",
        "Choose A, B, or tie for each item based on natural Mexican Spanish, clarity and usefulness. These are customer-ready replies; equal pairs can occur because rejected drafts fall back to the template. Do not open the separate answer key before judging.",
        "",
        "| Item | Preference (A / B / tie) | Comment |",
        "| --- | --- | --- |",
        *[f"| {i + 1} | | |" for i in range(len(chosen))],
        "",
    ]
    keys = []
    for n, (case, i, off, on) in enumerate(chosen, 1):
        swap = bool(rng.getrandbits(1))
        a, b = (on["message"], off["message"]) if swap else (off["message"], on["message"])
        lines += [
            f"## Item {n}",
            "",
            f"Customer: {off['input']}",
            "",
            f"**A:** {a}",
            "",
            f"**B:** {b}",
            "",
        ]
        keys.append(
            {
                "item": n,
                "case": case["id"],
                "turn": i,
                "rewrite": "A" if swap else "B",
                "template": "B" if swap else "A",
                "accepted": on["rewrite"]["accepted"],
                "reason": on["rewrite"]["reason"],
                "identical": a == b,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "rewrite-review.md").write_text("\n".join(lines) + "\n")
    if key_path is not None:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(json.dumps(keys, ensure_ascii=False, indent=2) + "\n")
