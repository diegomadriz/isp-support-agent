import asyncio
import json
from copy import deepcopy
from importlib.resources import files
from threading import RLock

from .models import CpeResult, Customer, OutageResult, RadioResult, Service, Ticket, UpstreamResult


def scenarios() -> dict[str, dict]:
    data = json.loads(files("isp_support_agent").joinpath("fixtures/scenarios.json").read_text())

    def merge(base, overrides):
        merged = deepcopy(base)
        for key, value in overrides.items():
            merged[key] = (
                merge(merged[key], value)
                if isinstance(value, dict) and isinstance(merged.get(key), dict)
                else deepcopy(value)
            )
        return merged

    return {name: merge(data["seed"], fixture) for name, fixture in data["scenarios"].items()}


class MockTicketSystem:
    """Fixture-seeded in-memory ticket API with atomic dedupe and idempotent writes."""

    def __init__(self, fixture: dict):
        self._lock = RLock()
        self._customers = {c["id"]: Customer.model_validate(c) for c in fixture["customers"]}
        self._services = [Service.model_validate(s) for s in fixture["services"]]
        self._tickets = {t["id"]: Ticket.model_validate(t) for t in fixture.get("tickets", [])}
        self._keys: dict[str, int] = {}
        self._connection = None

    def enable_persistence(self, connection, lock=None):
        self._lock = lock or self._lock
        self._connection = connection
        connection.execute(
            "CREATE TABLE IF NOT EXISTS mock_ticket_store (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
        )
        row = connection.execute("SELECT payload FROM mock_ticket_store WHERE id=1").fetchone()
        if row:
            saved = json.loads(row[0])
            self._tickets = {t["id"]: Ticket.model_validate(t) for t in saved["tickets"]}
            self._keys = saved["keys"]
        else:
            self._save()

    def _save(self):
        if self._connection:
            payload = json.dumps(
                {"tickets": [t.model_dump() for t in self._tickets.values()], "keys": self._keys}
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO mock_ticket_store VALUES (1, ?)", (payload,)
            )
            self._connection.commit()

    def customer(self, customer_id: int) -> Customer | None:
        return self._customers.get(customer_id)

    def services(self, customer_id: int) -> list[Service]:
        return [s for s in self._services if s.customer_id == customer_id]

    def tickets(self, customer_id: int) -> list[Ticket]:
        with self._lock:
            return [t for t in self._tickets.values() if t.customer_id == customer_id]

    def categories(self) -> tuple[str, ...]:
        return ("technical", "administrative", "general_support")

    def create_ticket(self, customer_id: int, category: str, note: str, key: str) -> Ticket:
        with self._lock:
            if key in self._keys:
                saved = self._tickets[self._keys[key]]
                if saved.customer_id != customer_id or saved.category != category:
                    raise ValueError("Idempotency key belongs to another account or category")
                return saved
            if customer_id not in self._customers or category not in self.categories():
                raise ValueError("Invalid customer or category")
            # Atomic second check protects concurrent conversations about one account.
            existing = next(
                (
                    t
                    for t in self.tickets(customer_id)
                    if t.status == "open" and t.category == category
                ),
                None,
            )
            if existing:
                return self.add_note(existing.id, note, key)
            ticket = Ticket(
                id=max(self._tickets, default=1000) + 1,
                customer_id=customer_id,
                category=category,
                notes=(note,),
            )
            self._tickets[ticket.id] = ticket
            self._keys[key] = ticket.id
            self._save()
            return ticket

    def add_note(self, ticket_id: int, note: str, key: str) -> Ticket:
        with self._lock:
            if key in self._keys:
                saved = self._tickets[self._keys[key]]
                if saved.id != ticket_id:
                    raise ValueError("Idempotency key belongs to another ticket")
                return saved
            ticket = self._tickets[ticket_id]
            ticket = ticket.model_copy(update={"notes": (*ticket.notes, note)})
            self._tickets[ticket_id] = ticket
            self._keys[key] = ticket_id
            self._save()
            return ticket

    def resolve(self, ticket_id: int, key: str) -> Ticket:
        with self._lock:
            ticket = self.add_note(ticket_id, "Customer reported resolution.", key)
            ticket = ticket.model_copy(update={"status": "resolved"})
            self._tickets[ticket_id] = ticket
            self._save()
            return ticket


class SimulatedNetwork:
    """Cancellation-friendly probes: no sockets, processes or device transports."""

    def __init__(self, fixture: dict, latency: float = 0.002):
        self.fixture = fixture
        self.latency = latency

    async def _run(self, kind, model, service):
        await asyncio.sleep(self.latency)
        failure = self.fixture.get("failures", {}).get(kind)
        if failure == "timeout":
            await asyncio.sleep(60)  # wait_for cancels this; never blocks a worker after timeout.
        if failure == "unreachable":
            raise ConnectionError("Simulated probe unreachable")
        data = dict(self.fixture[kind])
        if kind in {"cpe", "radio"}:
            data["address"] = service.cpe_address if kind == "cpe" else service.radio_address
        if kind == "outage":
            data["area"] = service.area
        return model.model_validate(data)

    async def cpe(self, service: Service) -> CpeResult:
        return await self._run("cpe", CpeResult, service)

    async def radio(self, service: Service) -> RadioResult:
        return await self._run("radio", RadioResult, service)

    async def upstream(self, service: Service) -> UpstreamResult:
        return await self._run("upstream", UpstreamResult, service)

    async def outage(self, service: Service) -> OutageResult:
        return await self._run("outage", OutageResult, service)


class FakeMessageSender:
    def __init__(self):
        self.messages: list[dict[str, str]] = []
        self.typing_events: list[dict[str, str]] = []
        self._typing_keys: set[str] = set()
        self._keys: set[str] = set()
        self._lock = RLock()

    def send(self, sender_id: str, message: str, key: str) -> None:
        with self._lock:
            if key not in self._keys:
                self.messages.append({"sender_id": sender_id, "message": message, "key": key})
                self._keys.add(key)

    def read_and_typing(self, sender_id: str, message_id: str) -> None:
        with self._lock:
            if message_id not in self._typing_keys:
                self.typing_events.append(
                    {
                        "sender_id": sender_id,
                        "message_id": message_id,
                        "status": "read",
                        "typing_indicator": "text",
                    }
                )
                self._typing_keys.add(message_id)
