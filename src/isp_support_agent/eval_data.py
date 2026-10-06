"""Evaluation split roles, shared by the artifact generators."""

CORE_SPLITS = ("dev", "test", "known-cases")
SPLIT_PROVENANCE = {
    "dev": "104 development messages used for few-shot examples and development.",
    "test": "102 synthetic messages written during development by the same authors after design feedback; one shared exact example and several paraphrases. In-distribution, optimistic and not independent. Scores were not used to tune the policy.",
    "known-cases": "15 development design cases used for guardrails and prompt instructions; not held out.",
    "blind": "50 synthetic messages: 25 each from Gemini and Grok in fresh chats with no project context, using only intent definitions. Labels were checked by hand before evaluation (x22: unclear; x23: escalate); messages and labels are frozen.",
    "options": "Four suggested phrases declared separately and recorded with the live model. A small routing check, not an independent accuracy set.",
}
PROVENANCE_TEXT = "Dev supplies the few-shot examples. Test shares one exact development example and several paraphrases; it is optimistic and not independent. Known-cases informed the guardrails and prompt instructions. Test scores were not used to tune the policy."


def provenance(splits):
    return {name: SPLIT_PROVENANCE[name] for name in splits}


def blind_status(splits):
    return (
        SPLIT_PROVENANCE["blind"] if "blind" in splits else "This evaluation has no blind messages."
    )


def display_splits(splits):
    names = (("blind",) if "blind" in splits else ()) + CORE_SPLITS + ("options",)
    return ((name, splits[name]) for name in names if name in splits)
