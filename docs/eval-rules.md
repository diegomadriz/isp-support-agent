# Reference rules evaluation

These scores measure reference rules only. The offline stub is a deterministic demo fixture, not a learned classifier.

50 synthetic messages: 25 each from Gemini and Grok in fresh chats with no project context, using only intent definitions. Labels were checked by hand before evaluation (x22: unclear; x23: escalate); messages and labels are frozen. Dev supplies the few-shot examples. Test shares one exact development example and several paraphrases; it is optimistic and not independent. Known-cases informed the guardrails and prompt instructions. Test scores were not used to tune the policy.

| Split | Rules accuracy | Critical recall |
| --- | --- | --- |
| blind | 74.0% (37/50) | 70.6% (12/17) |
| dev | 100.0% (104/104) | 100.0% (36/36) |
| test | 59.8% (61/102) | 56.7% (17/30) |
| known-cases | 100.0% (15/15) | 100.0% (6/6) |
| options | 100.0% (4/4) | 0.0% (0/0) |

Full confusion matrices and per-intent metrics are in the adjacent JSON. CI gates recorded real-model graph replay instead of this reference baseline.
