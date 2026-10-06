# Architecture

One async LangGraph `StateGraph` composes three subgraphs: identity, diagnostics and ticket confirmation. The [expanded diagram](graph.mmd) is generated with `get_graph(xray=True).draw_mermaid()` and checked in tests.

`session` holds verified identity and stable issue identifiers. `turn` holds the incoming message and its results and is replaced at the start of a turn. Probe evidence stays in the diagnostics subgraph, where an append reducer joins four `Send` tasks. The staff trace comes from `astream(stream_mode="updates", subgraphs=True)`, rather than checkpointed state. Nodes are module-level async functions; pure routers live separately. Dependencies enter through `Runtime.context`. `build_graph` contains wiring only.

## Ports

- `TicketSystem` owns customers, services, categories, tickets, creation, notes and resolution. The mock uses shared scenario fixtures, atomic category dedupe and durable idempotency records alongside the operation ledger.
- `NetworkProbe` returns typed CPE, radio, upstream and outage results. Its only shipped implementation is a deterministic simulation with configurable latency and failures. Each async probe has a timeout; partial failure remains a partial result.
- `LanguageModel` classifies raw Spanish text, preserving accents, and optionally rewrites public replies. Ollama and OpenAI-compatible adapters use structured output. The stub is a deterministic stand-in.
- `MessageSender` marks inbound messages read, shows typing and sends text with an idempotency key. The fake records calls; the Cloud adapter supplies the HTTP transport.

The runtime streams the graph asynchronously on a dedicated event loop. Synchronous ticket and ledger calls run in worker threads so SQLite commits cannot block that loop. Studio exposes the same ports through a serializable scenario context and a shared operational ledger.

## Decisions

Safety rules read every message first, including interrupt resumes. A critical rule or a model critical prediction alerts staff; confidence cannot downgrade a critical prediction. Private-data requests are blocked by policy. The model classifies other requests. Low confidence asks for clarification. Timeout or error without a rule hit also asks for clarification and records an **unclassified message, model unavailable** alert.

Rules take priority because missing a cut cable or complete outage costs more than an unnecessary review. They catch 12 of 17 blind critical messages and 17 of 30 synthetic-test critical messages. [Measured model results](eval-ollama.md) and [graph replay](eval-replay.md) distinguish critical intent recognition from staff alert coverage. The prompt's few-shot examples come from dev; additional instructions describe confusions in development design cases.

A fixed decision table owns the verdict: area outage, unresponsive CPE, weak radio, degraded upstream, healthy link, otherwise inconclusive. Strong evidence can outweigh a partial probe failure. A known area outage alerts staff and explains the outage without another individual ticket or an invented restoration time.

## Identity and consent

Identity asks for a parseable customer ID, then interrupts for the phone suffix. Three wrong four-digit guesses lock the customer for 15 minutes across resets, senders and restarts. Other replies do not spend attempts. Unknown IDs receive the same feedback. Only the customer-ID prompt accepts a bare ID; switching accounts requires verification again. Critical messages during verification acknowledge the staff alert without showing account data.

Ticket creation, notes and closure require customer confirmation through `interrupt()`. Reads happen before the interrupt; writes happen after it in a separate commit node. Categories and idempotency keys belong to policy. Staff alerts do not require ticket consent. Rate limits run before the graph, one customer has one active diagnostic run, and anonymous alerts collapse per hashed sender per minute.

Typing «otra cosa» asks an open question. Repeating the same unclear message offers a general-support request for a person to review, and opening it requires consent. The four suggested phrases have both end-to-end tests and live model recordings.

## Replies

Customer replies contain only `message` and `awaiting`. Staff evidence never enters the customer channel. The reply node renders a Spanish template; a separate rewrite node may reorder its sentences and add one short stock courtesy; acknowledgements open the reply and only the thanks may close it. Facts stay word for word. Actions, questions and options keep their relative order. A closed courtesy list rejects any other added content, alongside explicit checks for infrastructure, numbers, identifiers and tone. Greetings are allowed only in the first reply; angry messages allow only a calm acknowledgement and no reply allows two apologies. These literal checks do not establish semantic understanding.

Rewriting defaults on with a live model and off with the stub. It never runs before verification or on staff notes. The model receives public clauses and a bounded style profile, never raw customer text or evidence. Both templates and rewrites use tú or usted consistently. Errors, timeout or rejection send the template.

Each turn shares an eight-second model deadline: classification gets at most six seconds, and rewriting starts only if at least 1.5 seconds remain. Graph scheduling, probes, queueing and outbound delivery can add time. The [rewrite trial](rewrite-results.md) measures full graph turns, including classification, and retains raw responses and the budget clock for exact offline replay.

## Evaluation and deployment

The blind set has 50 synthetic messages, 25 each from Gemini and Grok in fresh chats given only intent definitions. Labels were checked by hand before evaluation. Dev has 104 messages used for examples. Test has 102 messages written during development by the same authors after design feedback, sharing one exact example and several paraphrases; its score is optimistic and not independent. Known-cases contains 15 design examples used for guardrails and prompt instructions. No policy was tuned on test or blind results.

CI replays recorded raw responses through the current graph and fails on stale inputs, missing critical alerts or accuracy more than one percentage point below the graph baseline. A separate replay checks rewriting against the current graph and validator. [Evaluation details](evaluation.md) explain the metrics and limits.

The production server disables the staff panel by default; `isp-agent serve` explicitly enables it for local use. SQLite stores checkpoints, customer lockouts, alert receipts and webhook replay records. [Deployment](deploy.md) covers the single-process server, provider contracts and retention.
