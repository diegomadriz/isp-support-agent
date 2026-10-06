import asyncio

from .models import (
    CpeResult,
    OutageResult,
    ProbeOutcome,
    RadioResult,
    Service,
    UpstreamResult,
    Verdict,
)
from .ports import NetworkProbe


async def run_probe(network: NetworkProbe, kind, service: Service, timeout: float) -> ProbeOutcome:
    async def bounded():
        return await asyncio.wait_for(getattr(network, kind)(service), timeout=timeout)

    try:
        return ProbeOutcome(kind=kind, data=await bounded())
    except TimeoutError:
        return ProbeOutcome(kind=kind, error="timeout")
    except ConnectionError:
        return ProbeOutcome(kind=kind, error="unreachable")
    except Exception:
        return ProbeOutcome(kind=kind, error="failed")


def decide(results: dict[str, ProbeOutcome]) -> Verdict:
    """Ordered decision table: strong fault evidence wins; healthy requires all four probes."""
    cpe = results.get("cpe", ProbeOutcome(kind="cpe")).data
    radio = results.get("radio", ProbeOutcome(kind="radio")).data
    upstream = results.get("upstream", ProbeOutcome(kind="upstream")).data
    outage = results.get("outage", ProbeOutcome(kind="outage")).data
    if isinstance(outage, OutageResult) and outage.active:
        return Verdict(code="area_outage", severity="high", next_action="guidance")
    if isinstance(cpe, CpeResult) and (not cpe.reachable or cpe.loss_pct == 100):
        return Verdict(code="last_mile_down", severity="high", next_action="ticket")
    if isinstance(radio, RadioResult) and (radio.signal_dbm < -75 or radio.quality_pct < 70):
        return Verdict(code="weak_radio_signal", severity="medium", next_action="ticket")
    if isinstance(upstream, UpstreamResult) and (
        upstream.loss_pct >= 5 or upstream.latency_ms > 100
    ):
        return Verdict(code="upstream_degraded", severity="medium", next_action="ticket")
    if (
        isinstance(cpe, CpeResult)
        and isinstance(radio, RadioResult)
        and isinstance(upstream, UpstreamResult)
        and isinstance(outage, OutageResult)
        and cpe.reachable
        and cpe.loss_pct == 0
        and cpe.rtt_ms is not None
        and cpe.rtt_ms <= 50
        and radio.capacity_mbps > 0
    ):
        return Verdict(code="local_wifi_or_device", severity="low", next_action="guidance")
    return Verdict(code="inconclusive", severity="medium", next_action="ticket")
