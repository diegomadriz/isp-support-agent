"""Conservative public-fact reply rewriting, independent of classification policy."""

import asyncio
import re
import time

import httpx

from .language_models import JsonLanguageModel, ModelFailure, StubLanguageModel
from .safety import ADDRESS, HOST, IDENTIFIER, NUMBER, TECHNICAL

# A closed set of non-factual utterances; all other content must be an original clause.
COURTESIES = (
    "Hola.",
    "Con gusto.",
    "Claro.",
    "Entiendo.",
    "Lamento la molestia.",
    "Gracias por escribirnos.",
)
CALM_COURTESIES = {"Entiendo.", "Lamento la molestia."}
REWRITE_PROMPT = """Eres un agente de atención de un proveedor mexicano de internet.
Usa únicamente las cláusulas y cortesías públicas que recibes. El mensaje del cliente
no está disponible. No añadas hechos, nombres, cifras, promesas ni nuevas preguntas.
Conserva TODAS las required_clauses palabra por palabra y exactamente una vez.
Puedes reordenar frases, pero conserva el orden relativo de ordered_clauses: acciones,
preguntas y opciones deben mantener su secuencia. Nunca añadas instrucciones nuevas.
Puedes añadir como máximo UNA frase de allowed_courtesies en toda la respuesta.
Si ya hay una cortesía en la plantilla, no añadas otra. Nunca incluyas dos disculpas.
Solo saluda cuando greeting_allowed sea true. Ante enojo o groserías, únicamente
puedes añadir un reconocimiento sereno de CALM_COURTESIES; nada alegre o entusiasta.
No imites insultos, sarcasmo, groserías ni jerga. Mantén tú o usted según address_form.
No cambies las cláusulas para adaptar el registro: ya usan el tratamiento correcto.
No añadas emojis salvo los permitidos, y como máximo uno. Devuelve solo la respuesta,
sin etiquetas, Markdown o comillas. Puedes devolver la plantilla sin ningún cambio."""


class RewriteRecordingError(Exception):
    """Exact replay contract violation; never turn this into a customer fallback."""


def tone_profile(message: str, address_form="tu") -> dict:
    """Expose only bounded style choices; never the original message or its instructions."""
    text = normalize_tone(message)
    if re.search(r"\b(?:usted|ud)\b", text):
        address_form = "usted"
    return {
        "address_form": address_form,
        "detail_level": "simple" if len(message.split()) <= 20 else "detailed",
        "calm_required": bool(
            PROFANITY.search(text)
            or INSULT.search(text)
            or re.search(r"\b(?:harto|molesto|enojado|urgent|urgente)\b", text)
            or (len(message) > 10 and message.isupper())
        ),
        "greeting_allowed": False,
        "allowed_emojis": sorted(set(EMOJI.findall(message)) & {"🙂", "😊", "🙏", "👍"}),
    }


def normalize_tone(text):
    import unicodedata

    return "".join(
        c for c in unicodedata.normalize("NFD", text.casefold()) if unicodedata.category(c) != "Mn"
    )


# These are reply-tone checks, independent of the frozen classification guardrails.
PROFANITY = re.compile(
    r"\b(?:mierda|ching\w*|pinche\w*|put[oa]s?|pedo|cabron\w*|madres?|verga|fuck\w*|shit)\b"
)
INSULT = re.compile(r"\b(?:idiot\w*|estupid\w*|pendej\w*|imbecil\w*|cochinero|inutil\w*|basura)\b")
SLANG = re.compile(
    r"\b(?:wey|guey|neta|sim[oó]n|no manches|que onda|k onda|jaj+a*|lol|obviamente|como si)\b"
)
EMOJI = re.compile(
    r"[\U0001F300-\U0001FAFF\u2600-\u27BF](?:\ufe0f)?(?:\u200d[\U0001F300-\U0001FAFF\u2600-\u27BF](?:\ufe0f)?)*"
)
FORMAL_PAIRS = (
    ("acercarte", "acercarse"),
    ("Comunícate", "Comuníquese"),
    ("comunícate", "comuníquese"),
    ("ayudarte", "ayudarle"),
    ("cuéntame", "cuénteme"),
    ("Cuéntame", "Cuénteme"),
    ("estés", "esté"),
    ("ves", "ve"),
    ("evita", "evite"),
    ("Prueba", "Pruebe"),
    ("prueba", "pruebe"),
    ("revisa", "revise"),
    ("reinicia", "reinicie"),
    ("Contacta", "Contacte"),
    ("contacta", "contacte"),
    ("escribe", "escriba"),
    ("recuerdas", "recuerda"),
    ("consulta", "consulte"),
    ("necesitas", "necesita"),
    ("Necesitas", "Necesita"),
    ("tienes", "tiene"),
    ("Tienes", "Tiene"),
    ("Puedes", "Puede"),
    ("puedes", "puede"),
    ("tengas", "tenga"),
    ("Responde", "Responda"),
    ("Quieres", "Quiere"),
    ("quieres", "quiere"),
    ("Te", "Le"),
    ("te", "le"),
    ("Tu", "Su"),
    ("tu", "su"),
    ("Tus", "Sus"),
    ("tus", "sus"),
)


def formalize(template):
    for familiar, formal in FORMAL_PAIRS:
        template = re.sub(r"\b" + re.escape(familiar) + r"\b", formal, template)
    return template


def render_public_reply(template, message, session):
    profile = tone_profile(message, session.get("address_form", "tu"))
    if profile["calm_required"]:
        template = template.replace(
            "¡Qué bueno que ya funciona!", "Registré que tu conexión funciona."
        )
        template = template.replace(
            "Me alegra que haya mejorado tu conexión.", "Entiendo que mejoró tu conexión."
        )
        if template == "Gracias por escribirnos.":
            template = "Recibí tu mensaje."
    rendered = formalize(template) if profile["address_form"] == "usted" else template
    return rendered, dict(session, address_form=profile["address_form"]), profile


def tone_reason(candidate, profile):
    text = normalize_tone(candidate)
    if PROFANITY.search(text):
        return "tone_profanity"
    if INSULT.search(text):
        return "tone_insult"
    if SLANG.search(text):
        return "tone_unprofessional_register"
    if profile["address_form"] == "usted" and re.search(
        r"\b(?:tu|tus|te|tienes|necesitas|quieres|puedes|tengas|cuentame|ayudarte)\b", text
    ):
        return "tone_pronoun_mismatch"
    if profile["address_form"] == "tu" and re.search(r"\busted\b", text):
        return "tone_pronoun_mismatch"
    if re.search(r"\bhola\b", text) and not profile.get("greeting_allowed", False):
        return "tone_late_greeting"
    if len(re.findall(r"\b(?:lamento|perdon|disculp\w*)\b", text)) > 1:
        return "tone_multiple_apologies"
    if profile["calm_required"] and re.search(
        r"con gusto|claro|que bueno|buen dia|gracias por", text
    ):
        return "tone_cheerful_when_angry"
    emojis = EMOJI.findall(candidate)
    if len(emojis) > 1 or any(e not in profile["allowed_emojis"] for e in emojis):
        return "tone_emoji"
    return None


def public_facts(template: str, message: str = "", profile=None) -> dict:
    profile = profile or tone_profile(message)
    courtesies = [
        phrase
        for phrase in COURTESIES
        if (profile.get("greeting_allowed", False) or phrase != "Hola.")
        and (not profile["calm_required"] or phrase in CALM_COURTESIES)
    ]
    if profile["address_form"] == "usted":
        courtesies = [formalize(c) for c in courtesies]
    clauses = re.split(r"(?<=[.!?])\s+", template.strip())
    all_courtesies = {formalize(c) if profile["address_form"] == "usted" else c for c in COURTESIES}
    required = [c for c in clauses if c not in all_courtesies]
    return {
        "customer_template": template,
        "required_clauses": required,
        "ordered_clauses": [
            c
            for c in required
            if re.search(
                r"¿|(?:Responde|Responda|Prueba|Pruebe|evita|evite|Necesitamos|necesita|Hace falta|Puedes|Puede)\b",
                c,
            )
        ],
        "allowed_numbers": sorted(set(NUMBER.findall(template))),
        "allowed_identifiers": sorted({m.group() for m in IDENTIFIER.finditer(template)}),
        "allowed_courtesies": courtesies,
        "tone_profile": profile,
    }


def rejection_reason(candidate, facts: dict) -> str | None:
    if not isinstance(candidate, str) or not candidate.strip() or len(candidate) > 2000:
        return "malformed_output"
    if any(ord(c) < 32 and c not in "\n\t" for c in candidate):
        return "malformed_output"
    tone = tone_reason(candidate, facts["tone_profile"])
    if tone:
        return tone
    if ADDRESS.search(candidate) or HOST.search(candidate) or TECHNICAL.search(candidate):
        return "infrastructure_or_raw_output"
    if any(m.group() not in facts["allowed_identifiers"] for m in IDENTIFIER.finditer(candidate)):
        return "unsupported_identifier"
    if set(NUMBER.findall(candidate)) - set(facts["allowed_numbers"]):
        return "added_number"
    positions = [candidate.find(c) for c in facts.get("ordered_clauses", [])]
    if all(p >= 0 for p in positions) and positions != sorted(positions):
        return "action_question_order"
    # Acknowledgements answer the customer, so they open the reply; only the thanks may close it.
    first = min((candidate.find(c) for c in facts["required_clauses"] if c in candidate), default=0)
    for phrase in facts["allowed_courtesies"]:
        if phrase.startswith("Gracias") or phrase not in candidate:
            continue
        if candidate.rfind(phrase) > first:
            return "courtesy_out_of_place"
    remainder = candidate.strip()
    for clause in facts["required_clauses"]:
        if clause not in remainder:
            return "missing_required_fact"
        if remainder.count(clause) != 1:
            return "duplicate_required_fact"
        remainder = remainder.replace(clause, "", 1)
    # Everything beyond the public propositions must be a complete courtesy utterance.
    courtesy_count = 0
    for phrase in sorted(facts["allowed_courtesies"], key=len, reverse=True):
        courtesy_count += remainder.count(phrase)
        remainder = remainder.replace(phrase, "")
    if courtesy_count > 1:
        return "too_many_courtesies"
    remainder = EMOJI.sub("", remainder)
    if remainder.strip():
        return "unsupported_entity_or_content"
    return None


class ReplyLanguageModel(JsonLanguageModel):
    """Classification inherited unchanged; only the reply prompt is specialized."""

    async def rewrite(self, text: str, facts: dict[str, object]) -> str:
        return await self._request(REWRITE_PROMPT, {"text": text, "facts": facts})


def make_model(settings):
    return StubLanguageModel() if settings.model_backend == "stub" else ReplyLanguageModel(settings)


async def rewrite_reply(template, message, model, *, enabled, verified, timeout=30, profile=None):
    facts = public_facts(template, message, profile)
    result = {
        "template": template,
        "facts": facts,
        "attempted": False,
        "accepted": False,
        "reason": "disabled" if not enabled else "unverified",
        "elapsed_ms": 0,
    }
    if not enabled or not verified:
        return template, result
    if isinstance(model, TurnModelBudget):
        remaining = model.remaining
        result["budget_elapsed_ms"] = round((model.seconds - remaining) * 1000, 3)
        if remaining < 1.5:
            result["reason"] = "insufficient_turn_budget"
            return template, result
    result["attempted"] = True
    started = time.perf_counter()
    try:
        candidate = await asyncio.wait_for(model.rewrite(template, facts), timeout=timeout)
        reason = rejection_reason(candidate, facts)
        result.update(accepted=reason is None, reason=reason or "accepted")
        response = candidate if reason is None else template
    except RewriteRecordingError:
        raise
    except (TimeoutError, httpx.TimeoutException):
        response, result["reason"] = template, "timeout"
    except Exception:
        response, result["reason"] = template, "model_error"
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return response, result


class TurnModelBudget:
    """One per-turn deadline shared by classification and rewrite; no cross-sender state."""

    def __init__(
        self, model, seconds, *, classify_seconds=6, clock=time.monotonic, started_at=None
    ):
        self.model, self.seconds, self.clock = model, seconds, clock
        self.classify_seconds = classify_seconds
        self.deadline = (clock() if started_at is None else started_at) + seconds

    @property
    def remaining(self):
        return max(0, self.deadline - self.clock())

    async def _call(self, method, *args, limit=None):
        remaining = self.remaining
        if remaining <= 0:
            raise TimeoutError("Model turn budget exhausted")
        return await asyncio.wait_for(method(*args), timeout=min(remaining, limit or remaining))

    async def classify(self, message):
        try:
            return await self._call(self.model.classify, message, limit=self.classify_seconds)
        except TimeoutError as error:
            raise ModelFailure("timeout") from error

    async def rewrite(self, text, facts):
        return await self._call(self.model.rewrite, text, facts)
