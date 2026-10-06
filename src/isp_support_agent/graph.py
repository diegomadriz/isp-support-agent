"""One support graph: explicit wiring of three compiled subgraphs."""

from langgraph.graph import END, START, StateGraph

from . import nodes
from .graph_state import (
    AgentContext,
    ConfirmationState,
    DiagnosticState,
    IdentityState,
    SupportState,
)
from .models import Intent
from .routing import (
    after_diagnostics,
    after_identity,
    after_unclear,
    confirmation_route,
    identity_route,
    intent_route,
    probe_route,
    start_route,
)

# Public parser used by CLI and tests.
from .routing import confirmation_answer as confirmation_answer


def build_graph(checkpointer=None, *, context_schema=AgentContext):
    identity = StateGraph(
        IdentityState,
        context_schema=context_schema,
        input_schema=SupportState,
        output_schema=SupportState,
    )
    identity.add_node("begin", nodes.identity_begin)
    identity.add_node("identify", nodes.identify)
    identity.add_node("verify", nodes.verify)
    identity.add_node("denied", nodes.denied)
    identity.add_node("ok", nodes.noop)
    identity.add_node("stop", nodes.noop)
    identity.add_edge(START, "begin")
    destinations = ["identify", "verify", "denied", "ok", "stop"]
    for node in ("begin", "identify", "verify"):
        identity.add_conditional_edges(node, identity_route, destinations)
    for node in ("denied", "ok", "stop"):
        identity.add_edge(node, END)
    diagnostics = StateGraph(
        DiagnosticState,
        context_schema=context_schema,
        input_schema=SupportState,
        output_schema=SupportState,
    )
    diagnostics.add_node("prepare_probes", nodes.diagnosis_begin)
    diagnostics.add_node("probe", nodes.probe)
    diagnostics.add_node("verdict", nodes.verdict)
    diagnostics.add_edge(START, "prepare_probes")
    diagnostics.add_conditional_edges("prepare_probes", probe_route, ["probe", "verdict"])
    diagnostics.add_edge("probe", "verdict")
    diagnostics.add_edge("verdict", END)
    confirmation = StateGraph(
        ConfirmationState,
        context_schema=context_schema,
        input_schema=SupportState,
        output_schema=SupportState,
    )
    confirmation.add_node("prepare", nodes.prepare)
    confirmation.add_node("reply", nodes.render_confirmation)
    confirmation.add_node("rewrite", nodes.rewrite)
    confirmation.add_node("ask", nodes.ask)
    confirmation.add_node("commit", nodes.commit)
    confirmation.add_edge(START, "prepare")
    confirmation.add_edge("prepare", "reply")
    confirmation.add_edge("reply", "rewrite")
    confirmation.add_edge("rewrite", "ask")
    confirmation.add_conditional_edges(
        "ask",
        confirmation_route,
        {"commit": "commit", "ask": "reply", "prepare": "prepare", END: END},
    )
    confirmation.add_edge("commit", END)
    graph = StateGraph(SupportState, context_schema=context_schema)
    graph.add_node("identity", identity.compile())
    graph.add_node("diagnostics", diagnostics.compile())
    graph.add_node("ticket_confirmation", confirmation.compile())
    graph.add_node("classify", nodes.classify_node)
    graph.add_node("welcome", nodes.welcome)
    graph.add_node("unclear", nodes.unclear)
    graph.add_node("thanks", nodes.thanks)
    graph.add_node("billing", nodes.billing)
    graph.add_node("tickets", nodes.tickets_node)
    graph.add_node("escalate", nodes.escalate)
    graph.add_node("admin", nodes.administrative)
    graph.add_node("resolved", nodes.resolved)
    graph.add_node("reset", nodes.reset)
    graph.add_node("respond", nodes.respond)
    graph.add_node("rewrite", nodes.rewrite)
    graph.add_conditional_edges(
        START,
        start_route,
        ["reset", "identity"],
    )
    graph.add_conditional_edges("identity", after_identity, ["classify", "welcome", "respond"])
    graph.add_conditional_edges(
        "classify",
        intent_route,
        [i.value for i in Intent if i != Intent.DIAGNOSE] + ["diagnostics", "respond"],
    )
    graph.add_conditional_edges(
        "diagnostics", after_diagnostics, ["ticket_confirmation", "respond"]
    )
    for name in ("escalate", "admin", "resolved"):
        graph.add_edge(name, "ticket_confirmation")
    for name in (
        "welcome",
        "thanks",
        "billing",
        "tickets",
        "reset",
        "ticket_confirmation",
    ):
        graph.add_edge(name, "respond")
    graph.add_conditional_edges("unclear", after_unclear, ["ticket_confirmation", "respond"])
    graph.add_edge("respond", "rewrite")
    graph.add_edge("rewrite", END)
    return graph.compile(checkpointer=checkpointer)
