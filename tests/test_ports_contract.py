"""Contract checks providers can reuse with their own factory implementations."""

import asyncio

from isp_support_agent.adapters import MockTicketSystem, SimulatedNetwork, scenarios
from isp_support_agent.models import CpeResult, OutageResult, RadioResult, Service, UpstreamResult


def assert_ticket_contract(store, customer_id):
    assert store.customer(customer_id).id == customer_id
    assert all(s.customer_id == customer_id for s in store.services(customer_id))
    category = store.categories()[0]
    first = store.create_ticket(customer_id, category, "evidence", "contract-create")
    retry = store.create_ticket(customer_id, category, "evidence", "contract-create")
    assert first.id == retry.id
    store.add_note(first.id, "follow-up", "contract-note")
    store.add_note(first.id, "follow-up", "contract-note")
    assert len(next(t for t in store.tickets(customer_id) if t.id == first.id).notes) == 2
    closed = store.resolve(first.id, "contract-resolve")
    assert closed.status == store.resolve(first.id, "contract-resolve").status == "resolved"


async def assert_network_contract(probe, service):
    results = await asyncio.gather(
        probe.cpe(service), probe.radio(service), probe.upstream(service), probe.outage(service)
    )
    for result, kind in zip(
        results, [CpeResult, RadioResult, UpstreamResult, OutageResult], strict=True
    ):
        assert isinstance(result, kind)


def test_mock_ticket_contract():
    assert_ticket_contract(MockTicketSystem(scenarios()["healthy"]), 12)


def test_simulated_network_contract():
    fixture = scenarios()["healthy"]
    asyncio.run(
        assert_network_contract(
            SimulatedNetwork(fixture["network"]), Service.model_validate(fixture["services"][0])
        )
    )
