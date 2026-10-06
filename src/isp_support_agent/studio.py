"""Studio uses the same Runtime.context injection, with a serializable scenario selector."""

from functools import lru_cache

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict

from .adapters import MockTicketSystem, SimulatedNetwork, scenarios
from .graph import AgentContext, build_graph
from .operations import OperationStore
from .rewriting import make_model
from .settings import Settings


@lru_cache(maxsize=1)
def studio_settings():
    return Settings.from_env()


@lru_cache(maxsize=16)
def resources(scenario):
    settings = studio_settings().model_copy(update={"scenario": scenario})
    fixture = scenarios()[scenario]
    return AgentContext(
        MockTicketSystem(fixture),
        SimulatedNetwork(fixture["network"], settings.probe_latency),
        make_model(settings),
        settings,
        studio_operations(),
    )


@lru_cache(maxsize=1)
def studio_operations():
    # Customer verification budgets also survive switching the simulated scenario.
    return OperationStore(cooldown=studio_settings().lockout_seconds)


class StudioContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario: str = "healthy"

    @property
    def first_reply(self):
        # Direct Studio calls have no channel reply ledger; conservatively omit greetings.
        return False

    @property
    def tickets(self):
        return resources(self.scenario).tickets

    @property
    def network(self):
        return resources(self.scenario).network

    @property
    def model(self):
        return resources(self.scenario).model

    @property
    def settings(self):
        return resources(self.scenario).settings

    @property
    def operations(self):
        return resources(self.scenario).operations


def graph(config: RunnableConfig):
    # Studio sends {"scenario": "healthy"} as invocation context. Resources, including
    # the lockout ledger, outlive graph factory calls within this local server process.
    return build_graph(context_schema=StudioContext)
