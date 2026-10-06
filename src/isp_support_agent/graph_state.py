"""Typed conversation, turn and subgraph contracts."""

import operator
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from .operations import OperationStore
from .ports import LanguageModel, NetworkProbe, TicketSystem
from .settings import Settings


class ConversationState(TypedDict, total=False):
    address_form: Literal["tu", "usted"]
    customer_id: int | None
    verified: bool
    issue_counter: int
    issue_category: str
    active_issue: str | None
    ticket_id: int | None
    last_unclear: str


class TurnState(TypedDict, total=False):
    sender_id: str
    message: str
    request_id: str
    intent: str
    classification: dict
    response: str
    staff_note: str
    verdict: dict
    category: str
    action: Literal["create", "resolve"]
    existing_ticket_id: int | None
    key: str
    tone_profile: dict
    rewrite: dict
    operation_error: str


class SupportState(TypedDict, total=False):
    session: ConversationState
    turn: TurnState


class IdentityState(SupportState, total=False):
    identity_status: Literal["identify", "verify", "ok", "denied", "stop"]
    candidate: int | None
    identity_questions: int
    remaining: int
    feedback: str


class DiagnosticState(SupportState, total=False):
    outcomes: Annotated[list[dict], operator.add]
    service: dict


class ConfirmationState(SupportState, total=False):
    confirmation: Literal["yes", "no", "retry", "reset", "critical"]
    confirmation_attempts: int
    question: str


@dataclass
class AgentContext:
    tickets: TicketSystem
    network: NetworkProbe
    model: LanguageModel
    settings: Settings
    operations: OperationStore
    first_reply: bool = False
    turn_request_id: str = ""
