"""Importable support nodes. Dependencies enter only through Runtime.context."""

import asyncio
import hashlib
import hmac
import json
import re

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from . import templates
from .classification import classify_result, critical_signal, is_reset, normalize, words
from .diagnostics import decide, run_probe
from .graph_state import AgentContext
from .models import Intent, ProbeOutcome, Service
from .rewriting import render_public_reply, rewrite_reply
from .routing import (
    confirmation_answer,
    customer_id_from,
    is_greeting,
    needs_ticket_summary,
    reset_session,
)


async def resume_is_critical(answer, c, turn=None, customer=None):
    if critical_signal(answer):
        return True
    if (
        re.fullmatch("\\d{4}", answer.strip())
        or customer_id_from(answer) is not None
        or confirmation_answer(answer) != "retry"
    ):
        return False
    result = await classify_result(answer, c.model, c.settings.confidence_threshold)
    if result.model_error and turn is not None:
        await unavailable_alert(turn, c, customer)
    return result.prediction.intent == Intent.ESCALATE


def validate_resume(answer):
    if not isinstance(answer, str) or not answer.strip() or len(answer) > 2000:
        raise ValueError("Resume must be a nonempty string of at most 2000 characters")
    return answer


async def alert(turn, context, customer=None):
    # Sync ports may wait on SQLite writers; keep checkpoint commits free to run.
    await asyncio.to_thread(
        context.operations.critical_alert,
        getattr(context, "turn_request_id", "") or turn["request_id"],
        turn["sender_id"],
        customer,
    )


async def unavailable_alert(turn, context, customer=None):
    await asyncio.to_thread(
        context.operations.unclassified_alert,
        getattr(context, "turn_request_id", "") or turn["request_id"],
        turn["sender_id"],
        customer,
    )


async def identity_begin(state, runtime: Runtime[AgentContext]):
    session, turn = (dict(state.get("session", {})), dict(state["turn"]))
    candidate = customer_id_from(turn["message"])
    critical = critical_signal(turn["message"])
    if not session.get("verified") and critical:
        await alert(turn, runtime.context)
    if candidate is not None or is_greeting(turn["message"]):
        turn["intent"] = "welcome"
    if session.get("verified") and (candidate is None or candidate == session.get("customer_id")):
        return {"identity_status": "ok", "turn": turn}
    if candidate is None:
        candidate = session.get("customer_id")
    if candidate != session.get("customer_id"):
        session = reset_session(session)
    session.update(customer_id=candidate, verified=False)
    locked = (
        candidate is not None
        and (await asyncio.to_thread(runtime.context.operations.budget, candidate))[1]
    )
    return {
        "session": session,
        "turn": turn,
        "candidate": candidate,
        "identity_questions": 0,
        "remaining": 3,
        "feedback": "Ya avisé al equipo técnico. " if critical else "",
        "identity_status": "denied"
        if locked
        else "verify"
        if candidate is not None
        else "identify",
    }


async def identify(state, runtime: Runtime[AgentContext]):
    turn = dict(state["turn"])
    prefix = state.get("feedback", "") or (
        "Ese dato no parece un número de cliente. " if state["identity_questions"] else ""
    )
    answer = validate_resume(
        interrupt({"kind": "customer_id", "message": prefix + templates.IDENTIFY})
    )
    if is_reset(answer):
        turn["response"] = "Conversación reiniciada. " + templates.IDENTIFY
        return {"session": reset_session(state["session"]), "turn": turn, "identity_status": "stop"}
    candidate = customer_id_from(answer, allow_bare=True)
    questions = state["identity_questions"] + 1
    critical = await resume_is_critical(answer, runtime.context, turn)
    if critical:
        await alert(turn, runtime.context)
        turn["message"] = answer
        turn.pop("intent", None)
    if candidate is None:
        if questions >= 3:
            turn["response"] = (
                "Necesito tu número de cliente para continuar. Si no lo recuerdas, comunícate con nosotros por teléfono."
            )
        return {
            "turn": turn,
            "identity_questions": questions,
            "feedback": "Ya avisé al equipo técnico. " if critical else "",
            "identity_status": "stop" if questions >= 3 else "identify",
        }
    base = (
        state["session"]
        if candidate == state["session"].get("customer_id")
        else reset_session(state["session"])
    )
    session = dict(base, customer_id=candidate, verified=False)
    locked = (await asyncio.to_thread(runtime.context.operations.budget, candidate))[1]
    return {
        "session": session,
        "turn": turn,
        "candidate": candidate,
        "feedback": "",
        "identity_status": "denied" if locked else "verify",
    }


async def verify(state, runtime: Runtime[AgentContext]):
    c = runtime.context
    turn = dict(state["turn"])
    if (await asyncio.to_thread(c.operations.budget, state["candidate"]))[1]:
        return {"identity_status": "denied"}
    answer = validate_resume(
        interrupt({"kind": "verification", "message": state.get("feedback", "") + templates.VERIFY})
    )
    if is_reset(answer):
        turn["response"] = "Conversación reiniciada. " + templates.IDENTIFY
        return {"session": reset_session(state["session"]), "turn": turn, "identity_status": "stop"}
    if await resume_is_critical(answer, c, turn):
        await alert(turn, c)
        turn["message"] = answer
        turn.pop("intent", None)
        return {
            "turn": turn,
            "identity_status": "verify",
            "feedback": "Ya avisé al equipo técnico. ",
        }
    if re.match("^cliente\\b", normalize(answer)):
        candidate = customer_id_from(answer)
        if candidate is None:
            return {"identity_status": "identify"}
        base = (
            state["session"]
            if candidate == state["session"].get("customer_id")
            else reset_session(state["session"])
        )
        return {
            "candidate": candidate,
            "session": dict(base, customer_id=candidate, verified=False),
            "remaining": 3,
            "feedback": "",
            "identity_status": "denied"
            if (await asyncio.to_thread(c.operations.budget, candidate))[1]
            else "verify",
        }
    suffix = answer.strip()
    if not re.fullmatch("\\d{4}", suffix):
        return {
            "identity_status": "verify",
            "feedback": "Escribe solo los cuatro dígitos; esta respuesta no cuenta como intento. ",
        }
    customer = await asyncio.to_thread(c.tickets.customer, state["candidate"])
    matches = bool(customer and hmac.compare_digest(suffix, customer.phone_last4))
    valid, remaining = await asyncio.to_thread(c.operations.verify, state["candidate"], matches)
    return {
        "session": dict(state["session"], customer_id=state["candidate"], verified=valid),
        "remaining": remaining,
        "feedback": f"Los dígitos no coinciden. Te {('queda' if remaining == 1 else 'quedan')} {remaining} {('intento' if remaining == 1 else 'intentos')}. "
        if not valid
        else "",
        "identity_status": "ok" if valid else "denied" if not remaining else "verify",
    }


async def denied(state, runtime: Runtime[AgentContext]):
    return {
        "turn": dict(state["turn"], response=templates.DENIED),
        "session": dict(state["session"], verified=False),
    }


async def diagnosis_begin(state, runtime: Runtime[AgentContext]):
    services = await asyncio.to_thread(
        runtime.context.tickets.services, state["session"]["customer_id"]
    )
    return {"service": services[0].model_dump() if services else {}}


async def probe(task, runtime: Runtime[AgentContext]):
    c = runtime.context
    outcome = await run_probe(
        c.network, task["kind"], Service.model_validate(task["service"]), c.settings.probe_timeout
    )
    return {"outcomes": [outcome.model_dump(mode="json")]}


async def verdict(state, runtime: Runtime[AgentContext]):
    results = {r["kind"]: ProbeOutcome.model_validate(r) for r in state.get("outcomes", [])}
    decision = decide(results)
    turn = dict(
        state["turn"],
        verdict=decision.model_dump(),
        category="technical",
        action="create",
        response=templates.verdict_message(decision),
        staff_note=json.dumps(
            {
                "simulation": True,
                "verdict": decision.model_dump(),
                "probes": {k: v.model_dump(mode="json") for k, v in sorted(results.items())},
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    if decision.code == "area_outage":
        await asyncio.to_thread(
            runtime.context.operations.alert,
            turn["request_id"],
            state["session"]["customer_id"],
            "area_outage",
        )
    return {"turn": turn}


async def prepare(state, runtime: Runtime[AgentContext]):
    c = runtime.context
    turn, session = (dict(state["turn"]), dict(state["session"]))
    action = turn.get("action", "create")
    category = turn.get("category", "technical")
    tickets = await asyncio.to_thread(c.tickets.tickets, session["customer_id"])
    existing = next((t for t in tickets if t.status == "open" and t.category == category), None)
    turn["existing_ticket_id"] = existing.id if existing else None
    same = (
        session.get("issue_category") == category
        and session.get("active_issue")
        and (action != "resolve")
    )
    counter = session.get("issue_counter", 0) + (0 if same else 1)
    key = (
        session.get("active_issue")
        if same
        else hashlib.sha256(
            f"{turn['sender_id']}:{session['customer_id']}:{category}:{counter}".encode()
        ).hexdigest()
    )
    session.update(issue_counter=counter, issue_category=category, active_issue=key)
    turn["key"] = key + (":resolved" if action == "resolve" else "")
    if action == "resolve":
        question = (
            f"¿Confirmas que el problema terminó y cierro el folio {existing.id}?"
            if existing
            else "¿Confirmas que el problema terminó y registro que quedó resuelto?"
        )
    elif existing:
        question = f"Ya tienes el folio {existing.id} abierto. ¿Agrego esta revisión al reporte?"
    elif category == "administrative":
        question = (
            "¿Quieres que registre esta solicitud administrativa para que el equipo la revise?"
        )
    elif category == "general_support":
        question = "¿Quieres que una persona del equipo revise tu caso?"
    else:
        question = "¿Quieres que abra un reporte para dar seguimiento?"
    prefix = turn.get("response", "")
    if needs_ticket_summary(turn) and turn.get("intent") != "tickets":
        prefix += " " + templates.ticket_summary(tickets)
    return {
        "session": session,
        "turn": turn,
        "question": (prefix + " " + question + " Responde sí o no.").strip(),
        "confirmation_attempts": 0,
    }


async def ask(state, runtime: Runtime[AgentContext]):
    answer = validate_resume(
        interrupt({"kind": "confirmation", "message": state["turn"]["response"]})
    )
    turn = dict(state["turn"])
    if await resume_is_critical(answer, runtime.context, turn, state["session"]["customer_id"]):
        await alert(turn, runtime.context, state["session"]["customer_id"])
        turn.update(
            message=answer,
            intent="escalate",
            category="technical",
            action="create",
            response=templates.CRITICAL,
            staff_note=json.dumps({"source": "critical_resume", "severity": "high"}),
        )
        return {"turn": turn, "confirmation": "critical"}
    result = confirmation_answer(answer)
    count = state["confirmation_attempts"] + 1
    if result == "retry" and count >= 3:
        result = "no"
        turn["response"] = (
            "No pude confirmar tu respuesta y no hice cambios. Si quieres continuar, vuelve a solicitar la revisión."
        )
    elif result == "no":
        turn["response"] = (
            "De acuerdo, no hice cambios en tus reportes. Si necesitas algo más, aquí estoy."
        )
    elif result == "reset":
        turn["response"] = "Conversación reiniciada. " + templates.IDENTIFY
        return {"session": reset_session(state["session"]), "turn": turn, "confirmation": "reset"}
    return {"turn": turn, "confirmation": result, "confirmation_attempts": count}


async def commit(state, runtime: Runtime[AgentContext]):
    c = runtime.context
    turn, session = (dict(state["turn"]), dict(state["session"]))
    try:
        if turn.get("action") == "resolve":
            if turn.get("existing_ticket_id"):
                ticket = await asyncio.to_thread(
                    c.tickets.resolve, turn["existing_ticket_id"], turn["key"]
                )
                turn["response"] = (
                    f"¡Qué bueno que ya funciona! Cerré el folio {ticket.id} y registré la resolución."
                )
            else:
                turn["response"] = (
                    "¡Qué bueno que ya funciona! Registré que el problema quedó resuelto."
                )
                await asyncio.to_thread(
                    c.operations.alert, turn["key"], session["customer_id"], "resolved"
                )
            turn["response"] += " " + templates.OPTIONS
            session.update(active_issue=None, ticket_id=None)
        else:
            if turn.get("existing_ticket_id"):
                ticket = await asyncio.to_thread(
                    c.tickets.add_note, turn["existing_ticket_id"], turn["staff_note"], turn["key"]
                )
                turn["response"] = (
                    f"Agregué la revisión al folio {ticket.id}. El equipo de soporte le dará seguimiento."
                )
            else:
                ticket = await asyncio.to_thread(
                    c.tickets.create_ticket,
                    session["customer_id"],
                    turn["category"],
                    turn["staff_note"],
                    turn["key"],
                )
                turn["response"] = (
                    f"Tu solicitud quedó registrada con el folio {ticket.id}. El equipo de soporte le dará seguimiento."
                )
            session["ticket_id"] = ticket.id
    except Exception as error:
        turn["response"] = "No pude registrar la solicitud. Intenta de nuevo más tarde."
        turn["operation_error"] = type(error).__name__
    return {"session": session, "turn": turn}


async def classify_node(state, runtime: Runtime[AgentContext]):
    c = runtime.context
    result = await classify_result(
        state["turn"]["message"], c.model, c.settings.confidence_threshold
    )
    turn = dict(state["turn"], classification=result.dump(), intent=result.prediction.intent.value)
    if result.model_error:
        await unavailable_alert(turn, c, state["session"]["customer_id"])
        turn["staff_note"] = json.dumps(
            {
                "source": "unclassified_message",
                "reason": "model unavailable",
                "message": turn["message"],
                "error": result.model_error,
            },
            ensure_ascii=False,
        )
    session = dict(state["session"])
    if result.prediction.intent != Intent.UNCLEAR:
        session.pop("last_unclear", None)
    return {"session": session, "turn": dict(turn)}


async def welcome(state, runtime: Runtime[AgentContext]):
    return {"turn": dict(state["turn"], response=templates.welcome(runtime.context.settings))}


async def unclear(state, runtime: Runtime[AgentContext]):
    turn, session = dict(state["turn"]), dict(state["session"])
    message = words(turn["message"])
    repeated = message == session.get("last_unclear")
    session["last_unclear"] = message
    if repeated and turn.get("classification", {}).get("path") != "privacy_guard":
        turn.update(response=templates.HANDOFF, category="general_support", action="create")
        turn["staff_note"] = json.dumps(
            {"source": "general_support_request", "message": turn["message"]}, ensure_ascii=False
        )
    else:
        turn["response"] = templates.OTHER if message == "otra cosa" else templates.CLARIFY
    return {"turn": turn, "session": session}


async def thanks(state, runtime: Runtime[AgentContext]):
    return {"turn": dict(state["turn"], response=templates.THANKS)}


async def billing(state, runtime: Runtime[AgentContext]):
    customer = await asyncio.to_thread(
        runtime.context.tickets.customer, state["session"]["customer_id"]
    )
    response = f"Tu pago está {customer.payment_status}."
    if customer.payment_status == "pendiente":
        response += f" Tienes un saldo de ${customer.amount_due} MXN, con vencimiento el {customer.due_date}. Puedes pagar por {customer.payment_method}."
    return {"turn": dict(state["turn"], response=response)}


async def tickets_node(state, runtime: Runtime[AgentContext]):
    return {
        "turn": dict(
            state["turn"],
            response=templates.ticket_summary(
                await asyncio.to_thread(
                    runtime.context.tickets.tickets, state["session"]["customer_id"]
                )
            ),
        )
    }


async def escalate(state, runtime: Runtime[AgentContext]):
    await alert(state["turn"], runtime.context, state["session"]["customer_id"])
    return {
        "turn": dict(
            state["turn"],
            response=templates.CRITICAL,
            action="create",
            category="technical",
            staff_note=json.dumps(
                {"source": "critical_escalation", "severity": "high", "simulation": True}
            ),
        )
    }


async def administrative(state, runtime: Runtime[AgentContext]):
    cancellation = bool(re.search("baja|cancel", words(state["turn"]["message"])))
    response = (
        "Entiendo que quieres dar de baja el servicio."
        if cancellation
        else "El equipo administrativo puede ayudarte con esa solicitud."
    )
    return {
        "turn": dict(
            state["turn"],
            action="create",
            category="administrative",
            response=response,
            staff_note=json.dumps(
                {
                    "source": "administrative_request",
                    "request": "cancellation" if cancellation else "review",
                }
            ),
        )
    }


async def resolved(state, runtime: Runtime[AgentContext]):
    category = state["session"].get("issue_category") or "technical"
    return {
        "turn": dict(
            state["turn"],
            action="resolve",
            category=category,
            response="Me alegra que haya mejorado tu conexión.",
        )
    }


async def reset(state, runtime: Runtime[AgentContext]):
    return {
        "session": reset_session(state["session"]),
        "turn": dict(state["turn"], response="Conversación reiniciada. " + templates.IDENTIFY),
    }


async def respond(state, runtime: Runtime[AgentContext]):
    c = runtime.context
    turn = dict(state["turn"])
    response = turn.get("response", templates.CLARIFY)
    if (
        state["session"].get("verified")
        and needs_ticket_summary(turn)
        and (turn.get("intent") in {"billing", "diagnose"})
    ):
        response += " " + templates.ticket_summary(
            await asyncio.to_thread(c.tickets.tickets, state["session"]["customer_id"])
        )
    response, session, profile = render_public_reply(response, turn["message"], state["session"])
    profile["greeting_allowed"] = c.first_reply
    turn.update(response=response, tone_profile=profile)
    return {"turn": turn, "session": session}


async def render_confirmation(state, runtime: Runtime[AgentContext]):
    response, session, profile = render_public_reply(
        state["question"], state["turn"]["message"], state["session"]
    )
    profile["greeting_allowed"] = runtime.context.first_reply
    return {
        "turn": dict(state["turn"], response=response, tone_profile=profile),
        "session": session,
    }


async def rewrite(state, runtime: Runtime[AgentContext]):
    c = runtime.context
    turn = dict(state["turn"])
    response, result = await rewrite_reply(
        turn["response"],
        turn["message"],
        c.model,
        enabled=c.settings.rewrite,
        verified=state["session"].get("verified", False),
        timeout=c.settings.rewrite_timeout,
        profile=turn.get("tone_profile"),
    )
    return {"turn": dict(turn, response=response, rewrite=result)}


async def noop(state, runtime: Runtime[AgentContext]):
    return {}
