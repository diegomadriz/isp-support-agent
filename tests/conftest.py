import pytest

from isp_support_agent.runtime import AgentRuntime
from isp_support_agent.settings import Settings


@pytest.fixture
def runtime():
    agent = AgentRuntime(Settings())
    yield agent
    agent.close()


def verified(agent, sender="customer-a", customer=12, factor="1234"):
    assert agent.turn(sender, f"cliente {customer}").awaiting == "verification"
    assert agent.turn(sender, factor).awaiting is None
    return sender


@pytest.fixture
def pre_blind_root(tmp_path):
    """An isolated valid capture before blind recording, regardless of repo data."""
    import json
    from pathlib import Path

    from isp_support_agent.eval_data import CORE_SPLITS
    from isp_support_agent.recordings import (
        POLICY_FILES,
        canonical,
        digest,
        fingerprints,
        validate_recordings,
    )

    source = Path(__file__).resolve().parents[1]
    root = tmp_path / "pre-blind"
    for name in POLICY_FILES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / name).read_bytes())
    (root / "eval/blind.jsonl").write_bytes(b"")
    (root / "eval/options.jsonl").write_bytes(b"")
    manifest_path = root / "eval/frozen.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["counts"]["blind"] = 0
    manifest["sha256"]["blind.jsonl"] = digest(b"")
    manifest_path.write_text(json.dumps(manifest))
    bundle = json.loads((source / "eval/recordings.json").read_text())
    bundle["result"]["splits"] = {name: bundle["result"]["splits"][name] for name in CORE_SPLITS}
    bundle["baseline"] = {name: bundle["baseline"][name] for name in CORE_SPLITS}
    bundle["records"] = [r for r in bundle["records"] if r["split"] in CORE_SPLITS]
    bundle["result"]["metadata"]["benchmark_inferences"] = len(bundle["records"])
    bundle["fingerprints"] = fingerprints(root)
    bundle["records_sha256"] = digest(canonical(bundle["records"]))
    bundle["baseline_source_sha256"] = digest(canonical(bundle["baseline"]))
    (root / "eval/recordings.json").write_text(json.dumps(bundle))
    validate_recordings(bundle, root)
    return root
