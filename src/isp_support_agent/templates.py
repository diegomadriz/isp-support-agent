from .models import Verdict

VERDICT_MESSAGES = {
    "local_wifi_or_device": "Revisé tu enlace y responde bien. Prueba acercarte al módem y comparar con otro dispositivo. Si mejora cerca del módem, revisa la cobertura de Wi-Fi; si solo falla un equipo, reinicia ese dispositivo. Si sigue fallando, cuéntame qué sucede.",
    "weak_radio_signal": "Revisé tu enlace y la señal llega débil. Hace falta que el equipo técnico lo revise.",
    "last_mile_down": "Tu equipo de conexión no responde a la revisión. El equipo técnico necesita revisar el enlace.",
    "upstream_degraded": "Detecté una falla en la conexión hacia el proveedor. Necesitamos una revisión técnica.",
    "area_outage": "Hay una falla registrada en tu zona. Avisé al equipo técnico; por ahora no necesitas abrir otro reporte. No tengo un horario de restablecimiento confirmado.",
    "inconclusive": "No pude completar todas las comprobaciones. Necesitamos una revisión técnica para encontrar la causa.",
}
OPTIONS = (
    "Puedes escribirme, por ejemplo: «internet lento», «ver mis reportes», «pagos» u «otra cosa»."
)


CLARIFY = (
    "¿Necesitas ayuda con tu conexión, un pago o un reporte? Cuéntame qué está pasando. " + OPTIONS
)
DENIED = "No pude verificar tu identidad. Por seguridad, los intentos están bloqueados temporalmente. Comunícate con nosotros por teléfono para recibir ayuda."
VERIFY = "Para verificar tu identidad, escribe los últimos cuatro dígitos del teléfono registrado."
IDENTIFY = "Para empezar, escribe tu número de cliente, por ejemplo: cliente 12. Si no lo recuerdas, consulta tu recibo o comunícate con nosotros por teléfono."
WELCOME = (
    "Tu identidad quedó verificada. ¿En qué puedo ayudarte: conexión, pagos o reportes? " + OPTIONS
)
CRITICAL = "Lamento que estés sin servicio o con una falla de este tipo. Ya avisé al equipo técnico para que lo revise. Si ves un cable roto, evita tocarlo."
THANKS = "Gracias por escribirnos."
OTHER = "Cuéntame con tus palabras qué necesitas."
HANDOFF = "No logré entender bien qué necesitas, y prefiero no adivinar."


def verdict_message(verdict: Verdict) -> str:
    return VERDICT_MESSAGES[verdict.code]


def ticket_summary(tickets) -> str:
    if not tickets:
        return "No tienes reportes registrados."
    return (
        "Tus reportes: "
        + "; ".join(
            f"folio {t.id} ({'abierto' if t.status == 'open' else 'resuelto'})" for t in tickets
        )
        + "."
    )


def welcome(settings):
    greeting = (
        f"Soy {settings.bot_name} de {settings.company_name}."
        if settings.bot_name
        else f"Te atiende {'el equipo de soporte' if settings.company_name == 'Soporte' else settings.company_name}."
    )
    return greeting + " " + WELCOME
