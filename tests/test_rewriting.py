import asyncio

import httpx
import pytest
from conftest import verified
from langgraph.runtime import Runtime

from isp_support_agent import nodes
from isp_support_agent.language_models import StubLanguageModel
from isp_support_agent.rewriting import (
    REWRITE_PROMPT,
    ReplyLanguageModel,
    public_facts,
    rejection_reason,
    rewrite_reply,
)
from isp_support_agent.runtime import AgentRuntime
from isp_support_agent.settings import Settings
from isp_support_agent.templates import VERDICT_MESSAGES


@pytest.mark.parametrize(
    "template",
    list(VERDICT_MESSAGES.values())
    + [
        "Tu solicitud quedó registrada con el folio 1001. El equipo de soporte le dará seguimiento.",
        "Tu pago está pendiente. Tienes un saldo de $499 MXN, con vencimiento el 2026-10-15.",
        "Ya tienes el folio 1001 abierto. ¿Agrego esta revisión al reporte? Responde sí o no.",
    ],
)
def test_exact_public_facts_and_courtesy_are_valid(template):
    facts = public_facts(template)
    assert rejection_reason(template, facts) is None
    assert rejection_reason("Entiendo. " + template, facts) is None
    for clause in facts["required_clauses"]:
        assert rejection_reason(template.replace(clause, "", 1), facts) is not None


@pytest.mark.parametrize(
    "suffix,reason",
    [
        (" IP 192.0.2.3", "infrastructure_or_raw_output"),
        (" router.example.invalid", "infrastructure_or_raw_output"),
        (" 3 hops", "infrastructure_or_raw_output"),
        (" RTT: 9 ms", "infrastructure_or_raw_output"),
        (" router-demo", "unsupported_identifier"),
        (" Vuelve en 20 minutos.", "added_number"),
        (" Juan va a resolverlo.", "unsupported_entity_or_content"),
        (" No es cierto lo anterior.", "unsupported_entity_or_content"),
        (" ping statistics: packets transmitted", "unsupported_entity_or_content"),
    ],
)
def test_rejection_reason_codes(suffix, reason):
    template = "Registré el folio 1001."
    assert rejection_reason(template + suffix, public_facts(template)) == reason


@pytest.mark.parametrize("missing", ["folio", "verdict", "action", "question"])
def test_required_ticket_verdict_action_and_question_survive(missing):
    clauses = {
        "folio": "Ya tienes el folio 1001 abierto.",
        "verdict": "Revisé tu enlace y la señal llega débil.",
        "action": "Hace falta que el equipo técnico lo revise.",
        "question": "¿Agrego esta revisión al reporte?",
    }
    template = " ".join(clauses.values())
    assert (
        rejection_reason(template.replace(clauses[missing], ""), public_facts(template))
        == "missing_required_fact"
    )


@pytest.mark.parametrize("candidate", [None, "", "x" * 2001, "hola\x00"])
def test_bad_model_output(candidate):
    assert rejection_reason(candidate, public_facts("Gracias.")) == "malformed_output"


def test_tone_facts_never_include_customer_text_or_staff_evidence():
    for message, address, calm in [
        ("Usted puede ayudarme", "usted", False),
        ("Estoy harto", "tu", True),
        ("qué onda", "tu", False),
    ]:
        facts = public_facts("Gracias.", message + " cliente 12 192.0.2.3")
        assert facts["tone_profile"]["address_form"] == address
        assert facts["tone_profile"]["calm_required"] == calm
        assert message not in str(facts) and "192.0.2" not in str(facts)


@pytest.mark.parametrize("failure", ["timeout", "error", "rejection"])
def test_fallback_and_reason(failure):
    class Model:
        async def rewrite(self, text, facts):
            if failure == "timeout":
                await asyncio.sleep(10)
            if failure == "error":
                raise RuntimeError("secret")
            return "Tu folio es 9999."

    async def run():
        template = "Tu folio es 1001."
        sent, result = await rewrite_reply(
            template, "hola", Model(), enabled=True, verified=True, timeout=0.001
        )
        assert sent == template and result["attempted"] and not result["accepted"]
        assert (
            result["reason"]
            == {"timeout": "timeout", "error": "model_error", "rejection": "added_number"}[failure]
        )
        assert "secret" not in str(result)

    asyncio.run(run())


def test_graph_step_is_individual_and_default_stays_off(runtime):
    assert not Settings().rewrite
    state = {"session": {"verified": True}, "turn": {"message": "gracias", "response": "Gracias."}}
    result = asyncio.run(nodes.rewrite(state, Runtime(context=runtime.context)))
    assert result["turn"]["response"] == "Gracias."
    assert result["turn"]["rewrite"]["reason"] == "disabled"


def test_verified_confirmation_prompt_is_rewritten_once_before_interrupt():
    class Model(StubLanguageModel):
        def __init__(self):
            super().__init__()
            self.requests = []

        async def rewrite(self, text, facts):
            self.requests.append((text, facts))
            return "Entiendo. " + text

    model = Model()
    r = AgentRuntime(Settings(rewrite=True), model=model)
    try:
        assert r.turn("a", "cliente 12").awaiting == "verification"
        assert not model.requests
        r.turn("a", "1234")
        assert len(model.requests) == 1
        prompt = r.turn("a", "Quiero cambiar mi plan")
        assert prompt.awaiting == "confirmation" and prompt.message.startswith("Entiendo.")
        assert "?" in prompt.message and not r.tickets.tickets(12)
        staff = r.staff_snapshot("a")
        assert "ticket_confirmation/rewrite" in staff["trace"] and staff["rewrite"]["accepted"]
        count = len(model.requests)
        r.turn("a", "Sí, por favor")
        assert len(model.requests) == count + 1  # final receipt only; no interrupt replay HTTP call
        assert len(r.tickets.tickets(12)) == 1
        assert "Entiendo" not in str(r.tickets.tickets(12)[0].notes)
        assert set(prompt.model_dump()) == {"message", "awaiting"}
    finally:
        r.close()


def test_rejected_confirmation_keeps_original_question_and_dedupe():
    class Model(StubLanguageModel):
        async def rewrite(self, text, facts):
            return "¿Quieres comprar un paquete por 999 pesos?"

    r = AgentRuntime(Settings(rewrite=True, scenario="existing_open_ticket"), model=Model())
    try:
        verified(r)
        reply = r.turn("customer-a", "Está lento")
        assert "folio 1001" in reply.message and "¿Agrego esta revisión" in reply.message
        assert r.staff_snapshot("customer-a")["rewrite"]["reason"] == "added_number"
        r.turn("customer-a", "no")
        assert len(r.tickets.tickets(12)) == 1
    finally:
        r.close()


def test_reply_adapter_uses_new_prompt_without_modifying_classification_prompt():
    payloads = []

    def handle(request):
        import json

        body = json.loads(request.content)
        payloads.append(body)
        return httpx.Response(200, json={"message": {"content": "Gracias."}})

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        model = ReplyLanguageModel(
            Settings(
                model_backend="ollama",
                model_base_url="http://127.0.0.1",
                model_name="gemma4:latest",
            ),
            client,
        )
        try:
            assert await model.rewrite("Gracias.", public_facts("Gracias.")) == "Gracias."
        finally:
            await model.close()

    asyncio.run(run())
    assert payloads[0]["messages"][0]["content"] == REWRITE_PROMPT


def test_case_preparation_includes_full_blind_and_suggested_phrases():
    from pathlib import Path

    from isp_support_agent.rewrite_experiment import case_specs

    cases = case_specs(Path(__file__).resolve().parents[1])
    assert len(cases) == 70
    assert sum(c["group"] == "blind" for c in cases) == 50
    assert sum(c["group"] == "options" for c in cases) == 4
    assert all(c["messages"][:2] == ["cliente 12", "1234"] for c in cases if c["group"] == "blind")


@pytest.fixture(scope="module")
def shared_fake_rewrite_capture(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("reply-recording")
    monkeypatch = pytest.MonkeyPatch()
    import json
    from pathlib import Path

    from isp_support_agent import rewrite_experiment as experiment
    from isp_support_agent.recordings import POLICY_FILES

    source = Path(__file__).resolve().parents[1]
    for name in set(POLICY_FILES + experiment.EXTRA_FILES) - {
        "eval/rewrite-config.json",
        "eval/rewrite-cases.json",
    }:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / name).read_bytes())
    config = json.loads((source / "eval/model-config.json").read_text())

    def handler(request):
        from isp_support_agent.classification import rules

        if request.method == "GET":
            return httpx.Response(
                200, json={"models": [{"name": config["model"], "digest": config["model_digest"]}]}
            )
        body = json.loads(request.content)
        public = json.loads(body["messages"][1]["content"])
        if "message" in public:
            prediction = rules(public["message"])
            if public["message"] == "se corta todas las noches como a las 9":
                prediction = prediction.model_copy(update={"intent": "diagnose", "confidence": 1})
            raw = prediction.model_dump_json()
        else:
            raw = public["text"]
            # Only one permissible courtesy; no extra courtesy if the template has one.
            if not any(c in raw for c in public["facts"]["allowed_courtesies"]):
                raw = "Entiendo. " + raw
        return httpx.Response(200, json={"message": {"content": raw}})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        experiment.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    settings = Settings(
        model_backend="ollama",
        model_name=config["model"],
        model_base_url="http://127.0.0.1",
        model_timeout=config["request_timeout_seconds"],
    )
    bundle, result = experiment.record_experiment(
        tmp_path, settings, tmp_path / "eval/rewrite-recordings.json"
    )
    monkeypatch.undo()
    return tmp_path, bundle, result


@pytest.fixture
def fake_rewrite_capture(shared_fake_rewrite_capture, tmp_path):
    import shutil
    from copy import deepcopy

    source, bundle, result = shared_fake_rewrite_capture
    root = tmp_path / "inputs"
    shutil.copytree(source, root)
    return root, deepcopy(bundle), deepcopy(result)


def test_exact_rewrite_replays_actual_graph_and_review_pack(fake_rewrite_capture):
    import json

    from isp_support_agent import rewrite_experiment as experiment

    root, bundle, result = fake_rewrite_capture
    assert result["leaks"] == 0 and result["required_facts_preserved"] == 1
    assert result["acceptance_rate"] == 1 and result["attempted"] == sum(
        r["method"] == "rewrite" for r in bundle["captures"]
    )
    assert result["skipped_unverified"] > 0 and 0 < result["accepted_changed"] <= result["accepted"]
    assert experiment.replay_experiment(bundle, root) == result
    experiment.gate(result)
    for output in (root / "first", root / "second"):
        experiment.write_results(result, output)
        experiment.write_review(bundle, output, output / "key.json")
    for name in ("rewrite-results.json", "rewrite-results.md", "rewrite-review.md", "key.json"):
        assert (root / "first" / name).read_bytes() == (root / "second" / name).read_bytes()
    key = json.loads((root / "first/key.json").read_text())
    assert len(key) == 20 and {k["rewrite"] for k in key} == {"A", "B"}
    assert "accepted" not in (root / "first/rewrite-review.md").read_text()
    for c in bundle["conversations"]:
        assert [r["ticket_statuses"] for r in c["off"]] == [r["ticket_statuses"] for r in c["on"]]
        assert [r["staff_note"] for r in c["off"]] == [r["staff_note"] for r in c["on"]]


@pytest.mark.parametrize(
    "mutation", ["raw", "missing", "config", "facts", "behavior", "file", "prompt", "decision"]
)
def test_rewrite_capture_fails_closed(fake_rewrite_capture, mutation, monkeypatch):
    from isp_support_agent import rewrite_experiment as experiment
    from isp_support_agent.recordings import canonical, digest

    root, bundle, _ = fake_rewrite_capture
    if mutation == "raw":
        bundle["captures"][0]["raw_output"] = "changed"
    elif mutation == "missing":
        bundle["captures"].pop()
    elif mutation == "config":
        bundle["captures"][0]["config_hash"] = "wrong"
    elif mutation == "facts":
        next(r for r in bundle["captures"] if r["method"] == "rewrite")["facts"][
            "required_clauses"
        ] = []
        bundle["captures_sha256"] = digest(canonical(bundle["captures"]))
    elif mutation == "behavior":
        bundle["conversations"][0]["on"][2]["message"] = "changed"
        bundle["conversations_sha256"] = digest(canonical(bundle["conversations"]))
    elif mutation == "file":
        path = root / "src/isp_support_agent/rewriting.py"
        path.write_text(path.read_text() + "\n")
    elif mutation == "prompt":
        monkeypatch.setattr(experiment, "REWRITE_PROMPT", "changed")
    else:
        bundle["decision_rule"] = {}
    with pytest.raises(SystemExit, match="Re-record.*live model"):
        experiment.replay_experiment(bundle, root)


def test_rewrite_adoption_is_separate_from_safety_gate():
    from isp_support_agent.rewrite_experiment import gate

    gate(
        {
            "leaks": 0,
            "required_facts_preserved": 1,
            "acceptance_rate": 0.1,
            "added_latency_ms": {"p95": 10000},
        }
    )
    with pytest.raises(SystemExit):
        gate({"leaks": 1, "required_facts_preserved": 1})
    with pytest.raises(SystemExit):
        gate({"leaks": 0, "required_facts_preserved": 0.99})


def test_committed_live_rewrite_replay_is_offline_and_reproducible(tmp_path):
    import json
    from pathlib import Path

    from isp_support_agent import rewrite_experiment as experiment

    root = Path(__file__).resolve().parents[1]
    path = root / "eval/rewrite-recordings.json"
    if not path.exists():
        pytest.skip("Live rewrite experiment not yet recorded")
    bundle = json.loads(path.read_text())
    result = experiment.replay_experiment(bundle, root)
    experiment.gate(result)
    experiment.write_results(result, tmp_path)
    experiment.write_review(bundle, tmp_path, tmp_path / "rewrite-review-key.json")
    for name in ("rewrite-results.json", "rewrite-results.md"):
        assert (tmp_path / name).read_bytes() == (root / "docs" / name).read_bytes()


def test_exact_replay_failure_is_not_swallowed_as_template_fallback():
    from isp_support_agent.rewriting import RewriteRecordingError

    class Model:
        async def rewrite(self, text, facts):
            raise RewriteRecordingError("stale request")

    with pytest.raises(RewriteRecordingError, match="stale"):
        asyncio.run(rewrite_reply("Gracias.", "gracias", Model(), enabled=True, verified=True))


@pytest.mark.parametrize(
    "addition,code",
    [
        (" Esto es una mierda.", "tone_profanity"),
        (" Son unos idiotas.", "tone_insult"),
        (" Qué onda, wey.", "tone_unprofessional_register"),
        (" Como si fuera mi culpa.", "tone_unprofessional_register"),
        (" 🙂", "tone_emoji"),
    ],
)
def test_tone_rejections_are_explicit_and_fall_back(addition, code):
    template = "Registré el folio 1001."
    facts = public_facts(template, "ESTOY HARTO DE SU COCHINERO")
    assert facts["tone_profile"]["calm_required"]
    assert rejection_reason(template + addition, facts) == code

    class Model:
        async def rewrite(self, text, facts):
            return text + addition

    sent, result = asyncio.run(
        rewrite_reply(template, "ESTOY HARTO", Model(), enabled=True, verified=True)
    )
    assert sent == template and result["reason"] == code


def test_emoji_is_optional_neutral_and_limited_to_one():
    template = "Registré el folio 1001."
    facts = public_facts(template, "gracias 🙂")
    assert rejection_reason(template + " 🙂", facts) is None
    assert rejection_reason(template + " 🙂🙂", facts) == "tone_emoji"
    assert rejection_reason(template + " 🙄", public_facts(template, "🙄")) == "tone_emoji"


def test_usted_is_consistent_across_turns_without_changing_required_facts():
    from isp_support_agent.rewriting import render_public_reply

    template = "Revisé tu enlace y responde bien. Prueba acercarte al módem."
    text, session, profile = render_public_reply(template, "¿Usted puede ayudarme?", {})
    assert text == "Revisé su enlace y responde bien. Pruebe acercarse al módem."
    facts = public_facts(text, profile=profile)
    assert rejection_reason("Entiendo. " + text, facts) is None
    assert rejection_reason("Te entiendo. " + text, facts) == "tone_pronoun_mismatch"
    assert (
        rejection_reason(text.replace("su enlace", "tu enlace"), facts) == "tone_pronoun_mismatch"
    )
    again, _, p = render_public_reply(
        "¡Con gusto! Si necesitas algo más, aquí estoy.", "gracias", session
    )
    assert "necesita" in again and p["address_form"] == "usted"


def test_graph_uses_same_trusted_usted_template_with_rewriting_on_and_off():
    class Model(StubLanguageModel):
        async def rewrite(self, text, facts):
            return (
                "Entiendo. " if facts["tone_profile"]["address_form"] == "usted" else "Entiendo. "
            ) + text

    for enabled in (False, True):
        r = AgentRuntime(Settings(rewrite=enabled), model=Model())
        try:
            verified(r)
            reply = r.turn("customer-a", "¿Usted puede revisar mi conexión lenta?")
            assert "su enlace" in reply.message and "tu enlace" not in reply.message
            closing = r.turn("customer-a", "gracias")
            assert "necesitas" not in closing.message
        finally:
            r.close()


def test_review_pack_mandatory_angry_and_casual_issue_turns(fake_rewrite_capture, tmp_path):
    import json

    from isp_support_agent.rewrite_experiment import write_review

    _, bundle, result = fake_rewrite_capture
    write_review(bundle, tmp_path, tmp_path / "key.json")
    key = json.loads((tmp_path / "key.json").read_text())
    for case_id in ("blind/x04", "blind/x12", "blind/x17", "blind/g11"):
        case = next(c for c in bundle["cases"] if c["id"] == case_id)
        item = next(k for k in key if k["case"] == case_id and k["turn"] == 2)
        assert item and case["issue_message"] in (tmp_path / "rewrite-review.md").read_text()
    assert result["draft_tone_flags"] == {} and result["sent_tone_violations"] == 0


def test_tone_gate_is_not_relaxed():
    from isp_support_agent.rewrite_experiment import gate

    with pytest.raises(SystemExit):
        gate({"leaks": 0, "required_facts_preserved": 1, "sent_tone_violations": 1})


def test_raw_tone_flags_and_delivered_tone_safety_are_distinct():
    from isp_support_agent.rewrite_experiment import DECISION, gate, metrics

    template = "Registré el folio 1001."
    facts = public_facts(template, "ESTOY HARTO")
    off = {"message": template, "input": "ESTOY HARTO", "elapsed_ms": 1}
    on = {
        **off,
        "elapsed_ms": 5,
        "rewrite": {
            "attempted": True,
            "accepted": False,
            "reason": "tone_profanity",
            "facts": facts,
        },
    }
    bundle = {
        "cases": [{"group": "demo"}],
        "conversations": [{"off": [off], "on": [on]}],
        "captures": [{"raw_output": template + " Mierda.", "facts": facts, "elapsed_ms": 4}],
        "config_hash": "test",
        "decision_rule": DECISION,
        "owner_decision": {},
    }
    result = metrics(bundle)
    assert result["draft_tone_flags"] == {"tone_profanity": 1}
    assert result["sent_tone_violations"] == result["leaks"] == 0
    assert result["required_facts_preserved"] == 1
    assert result["reasons"] == {"tone_profanity": 1}
    gate(result)
    on["message"] += " Mierda."
    with pytest.raises(SystemExit):
        gate(metrics(bundle))


def test_detail_profile_uses_only_bounded_style_values():
    from isp_support_agent.rewriting import tone_profile

    assert tone_profile("k onda") == {
        "address_form": "tu",
        "detail_level": "simple",
        "calm_required": False,
        "greeting_allowed": False,
        "allowed_emojis": [],
    }
    profile = tone_profile("¿Usted podría ayudarme? " + "detalle " * 25)
    assert profile["detail_level"] == "detailed" and profile["address_form"] == "usted"
