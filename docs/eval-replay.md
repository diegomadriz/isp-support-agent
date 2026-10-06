# Offline recorded-model graph replay

Raw live responses are parsed and routed by the current compiled graph, with no model or network. Each message starts with a verified fixture identity; ticket consent is not supplied. Identity/resume flows are tested separately.

50 synthetic messages: 25 each from Gemini and Grok in fresh chats with no project context, using only intent definitions. Labels were checked by hand before evaluation (x22: unclear; x23: escalate); messages and labels are frozen. Dev supplies the few-shot examples. Test shares one exact development example and several paraphrases; it is optimistic and not independent. Known-cases informed the guardrails and prompt instructions. Test scores were not used to tune the policy.

Model config SHA-256: `5714599fa78585bc715264e998503fc0f4600d6c9ab197cb963159d91493f94e`.

| Split | Graph accuracy | Critical recall | Model calls | Accuracy floor |
| --- | --- | --- | --- | --- |
| blind | 96.0% (48/50) | 100.0% (17/17) | 35 | 95.00% |
| dev | 94.2% (98/104) | 100.0% (36/36) | 60 | 93.23% |
| test | 98.0% (100/102) | 100.0% (30/30) | 80 | 97.04% |
| known-cases | 93.3% (14/15) | 100.0% (6/6) | 9 | not gated |
| options | 100.0% (4/4) | 0.0% (0/0) | 4 | not gated |

blind paths: `{"control:welcome": 1, "critical_rule": 12, "model": 35, "privacy_guard": 2}`.


dev paths: `{"control:reset": 4, "critical_rule": 36, "model": 60, "privacy_guard": 4}`.


test paths: `{"control:reset": 4, "control:welcome": 1, "critical_rule": 17, "model": 80}`.


known-cases paths: `{"critical_rule": 6, "model": 9}`.


options paths: `{"model": 4}`.

The CI gate requires 100% critical recall on dev, synthetic test, and blind when recorded, and accuracy on each within one percentage point of its pinned graph-replay baseline. Known-cases are development design cases and are not gated. Checksums reject stale policy, prompt, schema, model settings or datasets; re-record with a live model after such changes. This is a regression gate, not a guarantee of future live inference or arbitrary-message safety.
