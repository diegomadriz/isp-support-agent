# Deploying ISP Support Agent

The shipped ticket store and network are simulated. Real integrations are injected ports; no real device transport is included. Run one app worker: the runtime's diagnostic gate is process-local.

## Containers

```sh
docker compose up --build
# Customer demo: http://127.0.0.1:5000
curl http://127.0.0.1:5000/healthz
curl http://127.0.0.1:5000/readyz
```

The slim Python 3.12 image runs Gunicorn as UID 10001, with one worker and four threads. Checkpoints, the operation ledger and mock ticket idempotency records persist on the `checkpoints` volume at `/data`. `/healthz` checks the HTTP process; `/readyz` checks the worker and SQLite. Model availability is not readiness: provider errors deliberately degrade to clarification and templates.

Copy `.env.example` to `.env`, set values, and restart the app. Secrets are read from environment and never baked into the image. Use a stable random `AGENT_SESSION_SECRET` so web conversations survive app restarts. `AGENT_COMPANY_NAME` defaults to `Soporte`; `AGENT_BOT_NAME` is optional. The production server disables the staff panel by default (`AGENT_STAFF_PANEL=false`), including in Compose. `isp-agent serve` enables it explicitly for the local demo. Keep it off behind a public proxy.

Optional Ollama, on a host with sufficient memory and disk:

```sh
docker compose -f docker-compose.yml -f compose.ollama.yml --profile ollama up --build
```

The override sets the app's model configuration, waits for Ollama to be healthy, and runs an init service that pulls `gemma4:latest` before starting the app. Ollama is pinned to 0.24.0. Pulling weights requires disk space, memory and time; check the model size before starting. The default stub Compose setup never starts or pulls Ollama. Weights live on the separate `models` volume. Validate without downloading anything:

```sh
docker compose -f docker-compose.yml -f compose.ollama.yml --profile ollama config --quiet
```

For an OpenAI-compatible provider, set `AGENT_MODEL_BACKEND=openai`, `AGENT_MODEL_BASE_URL`, `AGENT_MODEL_NAME` and `AGENT_MODEL_KEY`. No provider endpoint or model is assumed. Live rewrite defaults on; `AGENT_REWRITE=false` opts out. Classification gets at most six seconds within an eight-second turn budget. Rewriting starts only with at least 1.5 seconds remaining. Rejection, errors or expiry send a template. When classification fails without a rule hit, the customer gets clarification and staff get an unclassified-message alert. Queueing, probes and delivery add time beyond that budget. Most offline demo scenarios use the stub and templates. Two additional conversations replay captured live responses through the current graph without a model.

## WhatsApp Cloud API

Create and configure a Meta app and business phone number. Expose `/webhook/whatsapp` through HTTPS, subscribe to message events, and set `AGENT_WHATSAPP_VERIFY_TOKEN` and `AGENT_WHATSAPP_APP_SECRET`. Meta's GET verification token must match. The POST handler verifies the signature on raw bytes, rejects malformed inputs, ignores stale timestamps and deduplicates message IDs in SQLite.

Set `AGENT_WHATSAPP_SENDER=cloud`, `AGENT_WHATSAPP_ACCESS_TOKEN`, and `AGENT_WHATSAPP_BASE_URL` to your configured versioned phone-number `/messages` endpoint. The shipped `CloudMessageSender` implements `MessageSender`: read plus typing before processing, then plain text replies. The default fake records those calls. [Meta's read-plus-typing request](https://www.postman.com/meta/whatsapp-business-platform/request/lhf0duq/send-typing-indicator-and-read-receipt) documents the endpoint. Tokens need the appropriate permissions and message-window policy; manage and rotate them outside this repository.

Webhook acknowledgement does not wait for classification or rewriting. A bounded background worker handles turns, saves replies, and retries pending jobs on restart. Outbound delivery can be repeated after an ambiguous transport failure; do not claim exactly-once delivery. No real Meta send was run here. Tests use local fake HTTP clients.

## Reverse proxy and TLS

Keep the app port on loopback and terminate TLS at a reverse proxy. Route `/`, `/api/chat`, health endpoints and `/webhook/whatsapp` to the app; preserve request bytes for webhook signature verification. Deny external `/api/staff` explicitly: a proxy connecting over loopback must not make the staff endpoint public. Authenticate any staff-facing replacement separately. Set request-body limits and allow a response timeout above the model budget. Restrict health endpoints as appropriate for your infrastructure.

## Provider ports

`TicketSystem` owns customer lookup, services, categories, tickets, create, note and resolve. It must enforce customer authorization, atomic same-category open-ticket dedupe and durable idempotency keys. [HTTP adapter template](../examples/http_ticket_adapter.py) is an untested skeleton with illustrative paths and schema, not a ready integration. Supply your own URL/token from environment and adapt the server contract before injecting it into `AgentRuntime(settings, tickets=...)`.

`NetworkProbe` returns frozen Pydantic CPE, radio, upstream and outage results scoped to the customer's `Service`. Implement its four async methods with cancellation-safe I/O and bounded timeouts. Return partial failures through the graph's `ProbeOutcome`; do not move technical data into customer templates. Ship and audit your own adapter separately; this repository includes no SSH or HTTPS device transport.

From a clone of this repository, reuse `assert_ticket_contract` and `assert_network_contract` from `tests/test_ports_contract.py` with provider factories and isolated test accounts. Run:

```sh
pytest tests/test_ports_contract.py tests/test_graph.py tests/test_web.py
pytest
isp-agent eval --graph-replay eval/recordings.json --gate
isp-agent rewrite-eval --replay eval/rewrite-recordings.json --gate
```

Also verify atomic dedupe under concurrency, permissions, restart persistence, timeout cancellation and provider API failure behavior against a local fake before contacting an actual server.

## Retention and backups

Checkpoints include identity state and customer conversation content; the operations ledger includes inbound message IDs, replies, alerts and lockout budgets. Choose retention based on your obligations; no automatic expiry is shipped. Back up SQLite consistently with its WAL, and stop the app before deleting conversation history. Preserve message receipts for at least the webhook freshness window and lockout budgets for the active cooldown, or replay/verification guarantees disappear. Restore and test backups and permissions for UID 10001. Protect the volume; do not include it in images or source control. Start a fresh volume only when intentionally discarding all demo state.
