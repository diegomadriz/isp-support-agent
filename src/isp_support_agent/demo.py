import json
from collections import Counter
from pathlib import Path

from .adapters import scenarios
from .runtime import AgentRuntime
from .settings import Settings


def run_demo(output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    (output / "transcripts").mkdir(exist_ok=True)
    records = []
    for name, fixture in scenarios().items():
        runtime = AgentRuntime(Settings(scenario=name, model_backend="stub"))
        sender = "demo-" + name
        lines = [
            f"# {name}",
            "",
            "Simulated customer conversation; no real account or network.",
            "",
        ]
        responses, last_verdict = [], None
        paths = Counter()
        try:
            for message in fixture["conversation"]:
                reply = runtime.turn(sender, message)
                lines.extend([f"**Customer:** {message}", "", f"**Agent:** {reply.message}", ""])
                responses.append(reply.message)
                path = runtime._staff[sender]["runtime_path"]
                paths["stub" if path == "model" else path] += 1
                current = runtime.snapshot(sender).values
                last_verdict = current.get("turn", {}).get("verdict") or last_verdict
            session = runtime._session(runtime.snapshot(sender))
            row = {
                "scenario": name,
                "verified": session.get("verified", False),
                "verdict": (last_verdict or {}).get("code"),
                "tickets": len(runtime.tickets.tickets(session.get("customer_id") or -1)),
                "awaiting": reply.awaiting,
                "staff_alerts": len(runtime.operations.alerts()),
                "runtime_paths": dict(paths),
                "customer_messages": responses,
            }
            records.append(row)
            (output / "transcripts" / f"{name}.md").write_text("\n".join(lines))
        finally:
            runtime.close()
    table = [
        "# Offline demo results",
        "",
        "| Scenario | Verified | Verdict | Tickets | Staff alerts |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in records:
        table.append(
            f"| {row['scenario']} | {row['verified']} | {row['verdict'] or '—'} | {row['tickets']} | {row['staff_alerts']} |"
        )
    totals = Counter()
    for row in records:
        totals.update(row["runtime_paths"])
    table += [
        "",
        "Runtime paths across all demo turns: `" + json.dumps(dict(totals), sort_keys=True) + "`.",
        "Identity/welcome and explicit reset are graph control paths. Other noncritical classification uses the deterministic stub. These are behavior transcripts, not model accuracy measurements.",
    ]
    (output / "demo-summary.md").write_text("\n".join(table) + "\n")
    (output / "demo-results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    )
    return records
