"""Pure routing and input parsers, separate from graph nodes."""

import re
from typing import Literal

from langgraph.graph import END
from langgraph.types import Send

from .classification import is_reset, normalize, wants_tickets, words
from .graph_state import ConfirmationState, DiagnosticState, IdentityState, SupportState


def customer_id_from(text: str, *, allow_bare=False) -> int | None:
    prefix = r"(?:cliente\s*#?\s*)?" if allow_bare else r"cliente\s*#?\s*"
    match = re.fullmatch(prefix + r"(\d{1,9})", normalize(text).strip())
    return int(match.group(1)) if match else None


def reset_session(session: dict) -> dict:
    # Retain a monotonically increasing issue generation across resets.
    return {"issue_counter": session.get("issue_counter", 0) + 1}


def identity_route(state: IdentityState) -> str:
    return state["identity_status"]


def intent_route(state: SupportState) -> str:
    if not state["session"].get("verified"):
        return "respond"
    intent = state["turn"].get("intent", "unclear")
    return "diagnostics" if intent == "diagnose" else intent


def confirmation_answer(answer: str) -> Literal["yes", "no", "retry", "reset"]:
    text = words(answer)
    if is_reset(answer):
        return "reset"
    if re.match(r"^no (?:se|estoy segur[oa]|entiendo)\b", text):
        return "retry"
    if re.match(r"^(?:no\b|nel\b|mejor no\b|cancel\w*\b)", text):
        return "no"
    if re.match(
        r"^(?:si\b|simon\b|sip\b|claro\b|ok\b|va\b|sale\b|dale\b|orale\b|andale\b|adelante\b|confirmo\b|por favor\b|por fa\b|porfa\b)",
        text,
    ):
        return "yes"
    return "retry"


def confirmation_route(state: ConfirmationState) -> str:
    return {"yes": "commit", "retry": "ask", "critical": "prepare", "no": END, "reset": END}[
        state["confirmation"]
    ]


def probe_route(state: DiagnosticState):
    if not state.get("service"):
        return "verdict"
    return [
        Send("probe", {"kind": k, "service": state["service"]})
        for k in ("cpe", "radio", "upstream", "outage")
    ]


def after_diagnostics(state: SupportState) -> str:
    return (
        "ticket_confirmation" if state["turn"]["verdict"]["next_action"] == "ticket" else "respond"
    )


def after_unclear(state: SupportState) -> str:
    return (
        "ticket_confirmation" if state["turn"].get("category") == "general_support" else "respond"
    )


def needs_ticket_summary(turn: dict) -> bool:
    secondary = turn.get("classification", {}).get("prediction", {}).get("secondary_intents", [])
    return wants_tickets(turn["message"]) or "tickets" in secondary


def after_identity(state: SupportState) -> str:
    if not state["session"].get("verified"):
        return "respond"
    return "welcome" if state["turn"].get("intent") == "welcome" else "classify"


def is_greeting(message):
    return words(message) in {
        "hola",
        "hola buenas tardes",
        "hola buenos dias",
        "buenas tardes",
        "buenas",
    }


def start_route(state):
    return "reset" if is_reset(state["turn"]["message"]) else "identity"
