# Evaluation

The datasets are synthetic Spanish support messages, not real customer conversations. Accuracy measures the labeled intent. Critical recall measures the flagged critical messages recognized by classification; graph recall measures those that actually produce a staff alert. A model failure may alert staff while returning a clarifying question, so graph recall and intent recognition are reported separately.

## Datasets

- **Blind:** 50 messages, 25 each from Gemini and Grok in fresh chats with no project context and only intent definitions. Labels were checked by hand before the first evaluation; x22 is unclear and x23 is escalate. Messages and labels are frozen. This is the most independent split, but still synthetic.
- **Dev:** 104 development messages used for few-shot examples and design.
- **Test:** 102 synthetic messages written during development by the same authors after design feedback. It shares one exact development example and several paraphrases, so it is in-distribution, optimistic and not independent. Scores were not used to tune the policy.
- **Known-cases:** 15 development design examples used to write guardrails and prompt instructions; not held out.
- **Options:** four suggested phrases, recorded live as a small routing check. They are not an independent accuracy dataset.

The few-shot examples come from dev. The prompt also describes confusions in the development design cases. Frozen manifests and recording fingerprints detect data changes.

## Classifier

Rules-only is the reference baseline. Model-only parses structured predictions and applies the confidence threshold without safety rules. Combined uses safety rules or a model critical flag, then confidence for other predictions. Safety and privacy guards may skip a model call; every other classification calls the model. The stub is a deterministic fixture and is never presented as learned-model accuracy.

[Live classifier results](eval-ollama.md) include accuracy, per-intent precision and recall, confusion matrices, critical recall, path counts, fallback reasons, errors and measured latency. The dev threshold curve is reported rather than treated as proof that confidence is calibrated. A flat curve supplies no threshold-selection signal; 0.7 remains the configured default.

The classifier timeout is six seconds. In the app, classification shares an eight-second turn deadline with rewriting. A failure without a safety or privacy rule returns clarification and alerts staff. Critical rules still alert staff immediately. Rules alone catch 12 of 17 blind critical messages and 17 of 30 test critical messages.

## Offline gate

```sh
isp-agent eval --mode rules-only
isp-agent eval --graph-replay eval/recordings.json --gate
isp-agent eval --replay docs/eval-ollama.json --output docs/eval-ollama.json
```

The gate parses raw recorded provider responses and routes each message through the current compiled graph. Messages start with a verified fixture identity; ticket consent is not supplied. Separate tests cover verification, interrupt resumes, lockouts and ticket writes. Every critical message must reach a staff alert. Accuracy must stay within one percentage point of the committed graph baseline. The prompt, guardrails, schema, settings, data and model configuration are fingerprinted; stale recordings fail closed.

Replay tests fixed responses, not future model drift or availability. Recorded latency is retained; offline replay speed is not a live performance measurement. To record fresh responses with the pinned local model, configure Ollama and run:

```sh
AGENT_MODEL_TIMEOUT=6 isp-agent eval --mode suite \
  --record eval/recordings.json --output docs/eval-ollama.json
```

Do not change labels or tune the policy on blind or test results. Additional independently written blind messages can be recorded with `isp-agent eval --record-blind` before freezing their first results.

## Reply rewriting

[Rewrite results](rewrite-results.md) pair full live conversations with rewriting enabled and disabled. Both arms use the same graph and fixtures, with live classification and raw response capture. The conversations cover all demo scenarios, one full conversation per blind message and the four suggestions. The two arms may differ when predictions or timeouts differ; matched input turns supply the length and added-latency comparisons. Node paths are exported from actual LangGraph stream events.

The model may reorder template sentences and add one short stock courtesy. Facts stay word for word, and actions, questions and options keep their order. The closed courtesy list is the main protection against added content. Profanity, insults, heavy slang, late greetings, multiple apologies, cheerful phrasing to angry customers, address mismatch and disallowed emojis reject a draft. Errors and rejected drafts send the template. Metrics distinguish raw draft flags from delivered violations.

The earlier trial accepted 268 of 271 drafts with zero leaks and all required facts kept. Its p95 added latency was 4.29 seconds, missing the three-second target set before the trial. Classification was held fixed. After seeing that result, the limit was raised to eight seconds because a few seconds is normal in WhatsApp chats. Warmer replies were preferred in a side-by-side read without item-by-item scores. This enabled rewriting; it did not pass the original target.

The current trial measures complete graph turns including classification. Classification gets up to six seconds; rewriting starts only with at least 1.5 seconds remaining within the shared eight-second deadline. Timings exclude the WhatsApp queue and outbound transport. Replay retains the measured budget clock and verifies both arms against the current graph.

```sh
AGENT_MODEL_TIMEOUT=6 isp-agent rewrite-eval --record eval/rewrite-recordings.json --gate
isp-agent rewrite-eval --replay eval/rewrite-recordings.json --gate
```

The rewrite safety gate requires zero delivered leaks and tone violations and 100% required facts. Raw responses, configuration hashes and exact requests are committed. The twenty-pair review form and its separate answer key are kept outside this repository, with at most two occurrences of any customer message and x04, x12, x17 and g11 included. No new human scores are inferred.
