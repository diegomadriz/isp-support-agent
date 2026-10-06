import asyncio
import hashlib
import json
import logging
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import replace
from threading import Lock, Thread

import aiosqlite
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from langsmith import tracing_context

from .adapters import MockTicketSystem, SimulatedNetwork, scenarios
from .classification import is_reset
from .graph import AgentContext, build_graph
from .models import CustomerReply
from .operations import OperationStore
from .rewriting import TurnModelBudget, make_model
from .settings import Settings

LOGGER = logging.getLogger("isp_support_agent")


def log_event(event: str, sender_id: str, **fields):
    LOGGER.info(
        json.dumps(
            {
                "event": event,
                "sender_hash": hashlib.sha256(sender_id.encode()).hexdigest()[:12],
                **fields,
            },
            sort_keys=True,
        )
    )


class RateLimiter:
    def __init__(self, limit: int, clock=time.monotonic):
        self.limit = limit
        self.clock = clock
        self._lock = Lock()
        self._times = defaultdict(deque)

    def allow(self, sender: str) -> bool:
        with self._lock:
            now = self.clock()
            # Expire inactive senders, not just timestamps on the currently active sender.
            self._times = defaultdict(
                deque, {s: q for s, q in self._times.items() if q and q[-1] > now - 60}
            )
            times = self._times[sender]
            while times and times[0] <= now - 60:
                times.popleft()
            if len(times) >= self.limit:
                return False
            times.append(now)
            return True


class AgentRuntime:
    """Async graph engine; the CLI/Flask bridge owns one event loop per runtime.

    Locks live around whole turns here, never across graph nodes. Single-process demo.
    """

    def __init__(
        self,
        settings=None,
        *,
        tickets=None,
        network=None,
        model=None,
        checkpointer=None,
        operations=None,
    ):
        self.settings = settings or Settings()
        fixture = scenarios()[self.settings.scenario]
        self.tickets = tickets or MockTicketSystem(fixture)
        self.network = network or SimulatedNetwork(fixture["network"], self.settings.probe_latency)
        self.model = model or make_model(self.settings)
        self.operations = operations or OperationStore(cooldown=self.settings.lockout_seconds)
        self.context = AgentContext(
            self.tickets, self.network, self.model, self.settings, self.operations
        )
        self.graph = build_graph(
            checkpointer or MemorySaver(serde=JsonPlusSerializer(allowed_json_modules=[])),
        )
        self.limiter = RateLimiter(self.settings.rate_limit)
        self._locks = {}
        self._notified = {}
        self._customer_owners = set()
        self._staff = {}
        self._connection = None
        self._loop = asyncio.new_event_loop()
        self._worker = Thread(target=self._loop.run_forever, daemon=True, name="isp-graph-loop")
        self._worker.start()
        self._closed = False

    def _run(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    @classmethod
    def persistent(cls, settings):
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        operations = OperationStore(settings.db_path, cooldown=settings.lockout_seconds)
        tickets = MockTicketSystem(scenarios()[settings.scenario])
        # Keep the mock's idempotency ledger with durable checkpoints, on the existing operations connection.
        tickets.enable_persistence(operations.connection, operations.lock)
        runtime = cls(settings, tickets=tickets, operations=operations)

        async def setup():
            runtime._connection = await aiosqlite.connect(settings.db_path)
            saver = AsyncSqliteSaver(
                runtime._connection, serde=JsonPlusSerializer(allowed_json_modules=[])
            )
            await saver.setup()
            runtime.graph = build_graph(saver)

        try:
            runtime._run(setup())
        except Exception:
            runtime.close()
            raise
        return runtime

    @staticmethod
    def config(sender):
        return {"configurable": {"thread_id": sender}, "recursion_limit": 80}

    @staticmethod
    def _session(snapshot):
        session = snapshot.values.get("session", {})
        for task in snapshot.tasks:
            if getattr(task, "state", None) and hasattr(task.state, "values"):
                nested = AgentRuntime._session(task.state)
                if nested:
                    session = nested
        return session

    @staticmethod
    def _turn(snapshot):
        turn = snapshot.values.get("turn", {})
        for task in snapshot.tasks:
            if getattr(task, "state", None) and hasattr(task.state, "values"):
                nested = AgentRuntime._turn(task.state)
                if nested:
                    turn = nested
        return turn

    @asynccontextmanager
    async def sender_lock(self, sender):
        entry = self._locks.setdefault(sender, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            async with entry[0]:
                yield
        finally:
            entry[1] -= 1
            if not entry[1]:
                self._locks.pop(sender, None)

    async def aturn(self, sender, message):
        if (
            not isinstance(sender, str)
            or not sender
            or len(sender) > 128
            or not isinstance(message, str)
            or not message.strip()
            or len(message) > 2000
        ):
            raise ValueError("Invalid sender or resume value")
        if not self.limiter.allow(sender):
            now = time.monotonic()
            already = self._notified.get(sender, 0) > now - 60
            self._notified = {s: t for s, t in self._notified.items() if t > now - 60}
            self._notified[sender] = now if not already else self._notified[sender]
            log_event("rate_limited", sender)
            return CustomerReply(
                message=""
                if already
                else "Has enviado muchos mensajes. Espera un momento antes de continuar."
            )
        async with self.sender_lock(sender):
            budget_clock = getattr(self.model, "budget_clock", time.monotonic)
            turn_started = budget_clock()
            config = self.config(sender)
            snapshot = await self.graph.aget_state(config, subgraphs=True)
            session = self._session(snapshot)
            customer = session.get("customer_id")
            if customer in self._customer_owners:
                return CustomerReply(
                    message="Ya hay una revisión en curso para tu cuenta. Intenta de nuevo en un momento."
                )
            if customer is not None:
                self._customer_owners.add(customer)
            turn_context = replace(
                self.context,
                turn_request_id=secrets.token_hex(16),
                first_reply=not await asyncio.to_thread(self.operations.has_replied, sender),
            )
            if self.settings.model_backend != "stub":
                turn_context = replace(
                    turn_context,
                    model=TurnModelBudget(
                        self.model,
                        8,
                        classify_seconds=self.settings.model_timeout,
                        clock=budget_clock,
                        started_at=turn_started,
                    ),
                )
            payload = (
                Command(resume=message)
                if snapshot.next
                else {
                    "session": snapshot.values.get("session", {}),
                    "turn": {
                        "sender_id": sender,
                        "message": message,
                        "request_id": turn_context.turn_request_id,
                    },
                }
            )
            trace = []
            try:
                with tracing_context(enabled=False):
                    async for namespace, update in self.graph.astream(
                        payload, config, context=turn_context, stream_mode="updates", subgraphs=True
                    ):
                        trace.extend(
                            "/".join([*(part.split(":")[0] for part in namespace), name])
                            for name in update
                            if name != "__interrupt__"
                        )
                final = await self.graph.aget_state(config, subgraphs=True)
                state = final.values
                pending = [i for task in final.tasks for i in task.interrupts]
                if pending:
                    value = pending[0].value
                    reply = CustomerReply(message=value["message"], awaiting=value["kind"])
                else:
                    reply = CustomerReply(message=state["turn"]["response"])
                current_session = self._session(final)
                current_customer = (
                    current_session.get("customer_id") if current_session.get("verified") else None
                )
                previous = self._staff.get(sender, {})
                if previous.get("customer_id") != current_customer:
                    previous = {}
                self._staff[sender] = {
                    "customer_id": current_customer,
                    "trace": trace,
                    "staff_note": state.get("turn", {}).get("staff_note")
                    or previous.get("staff_note", ""),
                    "classification": state.get("turn", {}).get("classification", {}),
                    "rewrite": self._turn(final).get("rewrite", {}),
                    "runtime_path": (
                        state["turn"]["classification"]["path"]
                        if "classify" in trace
                        else "interrupt_resume"
                        if snapshot.next
                        else "reset_control"
                        if is_reset(message)
                        else "identity_or_welcome"
                    ),
                }
                log_event(
                    "turn_complete",
                    sender,
                    awaiting=reply.awaiting,
                    route=state.get("turn", {}).get("intent", "identity"),
                )
                await asyncio.to_thread(self.operations.mark_replied, sender)
                return reply
            except Exception as error:
                log_event("turn_failed", sender, exception_type=type(error).__name__)
                raise
            finally:
                if customer is not None:
                    self._customer_owners.discard(customer)

    def turn(self, sender, message):
        return self._run(self.aturn(sender, message))

    def snapshot(self, sender):
        return self._run(self.graph.aget_state(self.config(sender), subgraphs=True))

    def staff_snapshot(self, sender):
        state = self.snapshot(sender)
        session = self._session(state)
        return (
            self._staff.get(sender, {"trace": [], "staff_note": "", "classification": {}})
            if session.get("verified")
            else {
                "trace": self._staff.get(sender, {}).get("trace", []),
                "staff_note": "",
                "classification": {},
            }
        )

    def close(self):
        if self._closed:
            return
        self._closed = True

        async def finish():
            if hasattr(self.model, "close"):
                await self.model.close()
            if self._connection:
                await self._connection.close()

        self._run(finish())
        self.operations.close()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._worker.join(timeout=5)
        self._loop.close()
