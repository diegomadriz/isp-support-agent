from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Intent(StrEnum):
    ESCALATE = "escalate"
    DIAGNOSE = "diagnose"
    BILLING = "billing"
    ADMIN = "admin"
    TICKETS = "tickets"
    RESOLVED = "resolved"
    RESET = "reset"
    UNCLEAR = "unclear"
    THANKS = "thanks"


class IntentPrediction(Record):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    reason: str
    critical: bool = False
    secondary_intents: tuple[Intent, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def compatible_names(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        value.pop("category", None)  # Categories are owned by ticket policy.
        aliases = {
            "technical": "diagnose",
            "technical_support": "diagnose",
            "diagnostic": "diagnose",
            "diagnostics": "diagnose",
            "administrative": "admin",
            "payment": "billing",
            "payments": "billing",
            "ticket_status": "tickets",
            "resolution": "resolved",
            "clarification": "unclear",
            "greeting": "unclear",
            "closing": "thanks",
            "critical": "escalate",
            "outage": "escalate",
        }
        intent = str(value.get("intent", "")).lower().strip()
        value["intent"] = aliases.get(intent, intent)
        return value

    @field_validator("reason")
    @classmethod
    def bound_reason(cls, value: str) -> str:
        return value[:500]  # Do not discard otherwise valid verbose model output.

    @property
    def category(self) -> str:
        return {
            Intent.ESCALATE: "technical",
            Intent.DIAGNOSE: "technical",
            Intent.ADMIN: "administrative",
            Intent.BILLING: "query",
            Intent.TICKETS: "query",
        }.get(self.intent, "none")


class Customer(Record):
    id: int
    phone_last4: str = Field(pattern=r"^\d{4}$")
    payment_status: Literal["al corriente", "pendiente"]
    amount_due: int = Field(default=499, ge=0)
    due_date: str = "2026-10-15"
    payment_method: str = "transferencia con tu número de cliente como referencia"


class Service(Record):
    customer_id: int
    cpe_address: str
    radio_address: str
    area: str


class Ticket(Record):
    id: int
    customer_id: int
    category: str
    status: Literal["open", "resolved"] = "open"
    notes: tuple[str, ...] = ()


class CpeResult(Record):
    reachable: bool
    loss_pct: float = Field(ge=0, le=100)
    rtt_ms: float | None = Field(default=None, ge=0)
    address: str


class RadioResult(Record):
    signal_dbm: float
    noise_dbm: float
    quality_pct: float = Field(ge=0, le=100)
    capacity_mbps: float = Field(ge=0)
    address: str


class UpstreamResult(Record):
    loss_pct: float = Field(ge=0, le=100)
    latency_ms: float = Field(ge=0)
    hop_count: int = Field(ge=0)
    hops: tuple[str, ...]


class OutageResult(Record):
    active: bool
    area: str


ProbeKind = Literal["cpe", "radio", "upstream", "outage"]
ProbeData = CpeResult | RadioResult | UpstreamResult | OutageResult


class ProbeOutcome(Record):
    kind: ProbeKind
    data: ProbeData | None = None
    error: Literal["timeout", "unreachable", "failed"] | None = None


class Verdict(Record):
    code: Literal[
        "local_wifi_or_device",
        "weak_radio_signal",
        "last_mile_down",
        "upstream_degraded",
        "area_outage",
        "inconclusive",
    ]
    severity: Literal["low", "medium", "high"]
    next_action: Literal["guidance", "ticket"]


class CustomerReply(Record):
    message: str
    awaiting: Literal["customer_id", "verification", "confirmation"] | None = None
