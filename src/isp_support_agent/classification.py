"""Safety policy and a reference classifier; live routing is model-first."""

import re
import time
import unicodedata
from dataclasses import dataclass
from difflib import get_close_matches

from .models import Intent, IntentPrediction


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


def words(message: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", normalize(message))
    vocabulary = (
        "internet",
        "servicio",
        "conexion",
        "senal",
        "fibra",
        "cable",
        "rojo",
        "modem",
        "antena",
        "cortado",
        "caido",
        "tengo",
        "funciona",
    )

    # Conservative token-level edit tolerance; no message-specific typo list.
    def corrected(token):
        if len(token) < 4 or token in vocabulary:
            return token
        matches = get_close_matches(token, vocabulary, n=1, cutoff=0.85)
        if matches and abs(len(token) - len(matches[0])) <= 1:
            return matches[0]
        return token

    tokens = [corrected(t) for t in tokens]
    return " ".join(tokens)


CRITICAL = (
    r"\b(?:luz|foco|foquito|led|modem|router|ont|antena)\b.{0,35}\broj[oa]\b",
    r"\blos\s+(?:parpadea|esta|enciende|en)\b.{0,15}\brojo\b",
    r"\b(?:fibra|cable)\b.*\b(?:cort\w*|rot\w*|danad\w*)\b|\b(?:cort\w*|rot\w*|dan\w*)\b.*\b(?:fibra|cable)\b",
    r"\bsin\b.{0,18}\b(?:internet|servicio|senal|conexion|red)\b",
    r"\bno\b.{0,14}\b(?:tengo|tenemos|hay|llega|recibo)\b.{0,20}\b(?:internet|servicio|conexion|senal|red)\b",
    r"\bse\s+(?:me\s+)?(?:fue|cayo)\b.{0,15}\b(?:internet|servicio|red|conexion|senal)\b",
    r"\b(?:internet|servicio|red|conexion)\b.{0,40}\b(?:caid\w*|muert\w*|apag\w*)\b|\b(?:caid\w*|apag\w*)\b.{0,18}\b(?:internet|servicio|red)\b",
    r"\b(?:no|ya no)\s+(?:funciona|jala|jalo)\b.{0,12}\binternet\b|\bcorte\s+total\b",
    r"\b(?:colonia|zona|vecinos|area)\b.*\b(?:sin|falla\w*|caid\w*|corte)\b|\b(?:falla\w*|apagon|caid\w*)\b.*\b(?:zona|colonia|area)\b",
)


def critical_signal(message: str) -> bool:
    return any(re.search(pattern, words(message)) for pattern in CRITICAL)


def is_reset(message: str) -> bool:
    return words(message) in {"reset", "reiniciar", "salir", "volver al menu"}


def prediction(intent: Intent, confidence: float = 0.98, reason: str = "Reference rule", **kwargs):
    return IntentPrediction(intent=intent, confidence=confidence, reason=reason, **kwargs)


def security_request(message: str) -> bool:
    text = words(message)
    return bool(
        re.search(
            r"ignora.*instruccion|system prompt|instrucciones internas|"
            r"(?:dame|muestra|revela|ver|consulta).*datos.*(?:otro cliente|cliente\s+\d+)|"
            r"(?:ip|hostname|contrasena).*?(?:router|cliente)",
            text,
        )
    )


def wants_tickets(message: str) -> bool:
    return bool(re.search(r"\b(?:tickets?|folios?|reportes?)\b", words(message)))


def unsafe_resolution(message: str) -> bool:
    return "?" in message or bool(
        re.search(r"\b(?:pero|sigue|todavia|aun|intermiten\w*|lento|lenta)\b", words(message))
    )


def rules(message: str) -> IntentPrediction:
    """Benchmark and deterministic stub, not the live model's noncritical fallback."""
    text = words(message)
    if critical_signal(message):
        return prediction(Intent.ESCALATE, 1, "Safety guardrail", critical=True)
    if security_request(message):
        return prediction(Intent.UNCLEAR, 0, "Privacy guardrail")
    if is_reset(message):
        return prediction(Intent.RESET)
    if re.search(r"\b(?:gracias|agradezco|chido)\b", text) and not re.search(
        r"\b(?:ya|pero|internet|pago|folio)\b", text
    ):
        return prediction(Intent.THANKS)
    if re.search(
        r"ya (?:quedo|funciona|sirve)|(?:si|ya) funcion[oa]|resuelto|se soluciono|ya jala", text
    ) and not unsafe_resolution(message):
        return prediction(Intent.RESOLVED)
    if (
        wants_tickets(message)
        and ("?" in message or "reporte" in text)
        and not re.search(r"lent|intermiten|conexion|internet", text)
    ):
        return prediction(Intent.TICKETS)
    if re.search(
        r"lent[oa]|lentisim|intermiten|se va y viene|velocidad|lag|no carga|desconex|wifi|wi fi|buffer|netflix|se traba|problema.*internet",
        text,
    ):
        return prediction(Intent.DIAGNOSE)
    if re.search(
        r"cambi.*plan|cancel|darme de baja|domicilio|aclaracion|no puedo pagar|cargo.*(?:doble|incorrect)|reembolso|contratar|actualizar.*datos",
        text,
    ):
        return prediction(Intent.ADMIN)
    if re.search(r"pag[oa]|saldo|factura|debo|adeudo|recibo|cobro|mensualidad", text):
        return prediction(Intent.BILLING)
    if wants_tickets(message):
        return prediction(Intent.TICKETS)
    return prediction(Intent.UNCLEAR, 0.35, "Insufficient information")


@dataclass(frozen=True)
class Classification:
    prediction: IntentPrediction
    path: str
    fallback_reason: str | None = None
    model_error: str | None = None
    latency_ms: float = 0

    def dump(self):
        return {
            "prediction": self.prediction.model_dump(mode="json"),
            "path": self.path,
            "fallback_reason": self.fallback_reason,
            "model_error": self.model_error,
            "latency_ms": round(self.latency_ms, 3),
        }


def apply_policy(
    message: str, result: IntentPrediction, threshold: float, *, guards=True
) -> Classification:
    if guards and critical_signal(message):
        return Classification(
            prediction(Intent.ESCALATE, 1, "Safety guardrail", critical=True), "critical_rule"
        )
    if guards and security_request(message):
        return Classification(prediction(Intent.UNCLEAR, 0, "Privacy guardrail"), "privacy_guard")
    if result.critical or result.intent == Intent.ESCALATE:
        return Classification(
            prediction(Intent.ESCALATE, result.confidence, result.reason, critical=True), "model"
        )
    if guards and result.intent == Intent.RESOLVED and unsafe_resolution(message):
        return Classification(
            prediction(Intent.UNCLEAR, result.confidence, "Resolution needs clarification"),
            "model",
            "ambiguous_resolution",
        )
    if result.confidence < threshold:
        return Classification(
            prediction(Intent.UNCLEAR, result.confidence, "Clarification required"),
            "model",
            "low_confidence",
        )
    return Classification(result, "model")


async def classify_result(message: str, model, threshold: float, *, guards=True) -> Classification:
    guarded = apply_policy(message, prediction(Intent.UNCLEAR), threshold, guards=guards)
    if guarded.path in {"critical_rule", "privacy_guard"}:
        return guarded
    start = time.perf_counter()
    try:
        raw = IntentPrediction.model_validate(await model.classify(message))
        result = apply_policy(message, raw, threshold, guards=guards)
        return Classification(
            result.prediction,
            result.path,
            result.fallback_reason,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as error:
        return Classification(
            prediction(Intent.UNCLEAR, 0, "Model request failed"),
            "model",
            "model_error",
            getattr(error, "reason", type(error).__name__),
            (time.perf_counter() - start) * 1000,
        )


async def classify(message: str, model, threshold: float) -> IntentPrediction:
    return (await classify_result(message, model, threshold)).prediction
