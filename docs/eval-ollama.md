# Evaluation: ollama/gemma4:latest

Threshold: 0.7; timeout: 6 seconds. Prompt SHA-256: `4d3896c16b30bbb36eb75c82b24c87f797385beb6ee84ac75aefe9491bd46fe8`.

50 synthetic messages: 25 each from Gemini and Grok in fresh chats with no project context, using only intent definitions. Labels were checked by hand before evaluation (x22: unclear; x23: escalate); messages and labels are frozen. Dev supplies the few-shot examples. Test shares one exact development example and several paraphrases; it is optimistic and not independent. Known-cases informed the guardrails and prompt instructions. Test scores were not used to tune the policy.

Model-only uses the structured model output and confidence threshold without safety rules. Combined uses safety rules OR the model critical flag, then the confidence threshold. The deterministic stub is a rules stand-in, not learned inference.

| Split | Mode | Accuracy | Critical recall | Model calls | Clarify | p50 / p95 ms |
| --- | --- | --- | --- | --- | --- | --- |
| blind | rules-only | 74.0% (37/50) | 70.6% (12/17) | 0 | 22.0% | 0 / 0 |
| blind | model-only | 94.0% (47/50) | 94.1% (16/17) | 50 | 12.0% | 3027.588 / 3505.489 |
| blind | combined | 96.0% (48/50) | 100.0% (17/17) | 36 | 12.0% | 2925.231 / 3459.501 |
| dev | rules-only | 100.0% (104/104) | 100.0% (36/36) | 0 | 9.6% | 0 / 0 |
| dev | model-only | 92.3% (96/104) | 97.2% (35/36) | 104 | 8.7% | 2820.552 / 3283.562 |
| dev | combined | 93.3% (97/104) | 100.0% (36/36) | 64 | 8.7% | 2610.758 / 3132.908 |
| test | rules-only | 59.8% (61/102) | 56.7% (17/30) | 0 | 42.2% | 0 / 0 |
| test | model-only | 96.1% (98/102) | 96.7% (29/30) | 102 | 9.8% | 2854.478 / 3379.794 |
| test | combined | 97.1% (99/102) | 100.0% (30/30) | 85 | 9.8% | 2805.126 / 3278.059 |
| known-cases | rules-only | 100.0% (15/15) | 100.0% (6/6) | 0 | 0.0% | 0 / 0 |
| known-cases | model-only | 93.3% (14/15) | 100.0% (6/6) | 15 | 6.7% | 2948.188 / 3405.336 |
| known-cases | combined | 93.3% (14/15) | 100.0% (6/6) | 9 | 6.7% | 2895.872 / 3405.336 |
| options | rules-only | 100.0% (4/4) | 0.0% (0/0) | 0 | 25.0% | 0 / 0 |
| options | model-only | 100.0% (4/4) | 0.0% (0/0) | 4 | 25.0% | 2600.32 / 2690.792 |
| options | combined | 100.0% (4/4) | 0.0% (0/0) | 4 | 25.0% | 2600.32 / 2690.792 |

Benchmark ran 275 inferences to obtain model-only scores; combined would call the model only for its counted model path. No noncritical cheap rule path is enabled.

## Dev threshold sensitivity

| Threshold | Accuracy | Clarification |
| --- | --- | --- |
| 0.3 | 93.3% | 8.7% |
| 0.4 | 93.3% | 8.7% |
| 0.5 | 93.3% | 8.7% |
| 0.6 | 93.3% | 8.7% |
| 0.7 | 93.3% | 8.7% |
| 0.8 | 93.3% | 8.7% |
| 0.9 | 93.3% | 8.7% |

The dev curve is flat from 0.3 through 0.9. Confidence supplies no observed threshold-selection signal on these messages; 0.7 is retained as the configured default, not an empirically optimal threshold.

## blind: rules-only

Paths: `{"reference_rules": 50}`. Fallbacks: `{}`. Model errors: `{}`.

| Intent | Precision | Recall | Support |
| --- | --- | --- | --- |
| escalate | 100.0% | 70.6% | 17 |
| diagnose | 90.0% | 90.0% | 10 |
| billing | 50.0% | 75.0% | 4 |
| admin | 100.0% | 100.0% | 4 |
| tickets | 75.0% | 100.0% | 3 |
| resolved | 100.0% | 50.0% | 2 |
| reset | 0.0% | 0.0% | 2 |
| unclear | 27.3% | 50.0% | 6 |
| thanks | 100.0% | 100.0% | 2 |

Confusion matrix: expected rows, predicted columns.

| Expected | escalate | diagnose | billing | admin | tickets | resolved | reset | unclear | thanks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| escalate | 12 | 0 | 1 | 0 | 1 | 0 | 0 | 3 | 0 |
| diagnose | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| billing | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 0 |
| admin | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| tickets | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| resolved | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| reset | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| unclear | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 3 | 0 |
| thanks | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |

## blind: model-only

Paths: `{"model": 50}`. Fallbacks: `{}`. Model errors: `{}`.

| Intent | Precision | Recall | Support |
| --- | --- | --- | --- |
| escalate | 100.0% | 94.1% | 17 |
| diagnose | 83.3% | 100.0% | 10 |
| billing | 100.0% | 100.0% | 4 |
| admin | 100.0% | 100.0% | 4 |
| tickets | 100.0% | 100.0% | 3 |
| resolved | 100.0% | 100.0% | 2 |
| reset | 100.0% | 50.0% | 2 |
| unclear | 83.3% | 83.3% | 6 |
| thanks | 100.0% | 100.0% | 2 |

Confusion matrix: expected rows, predicted columns.

| Expected | escalate | diagnose | billing | admin | tickets | resolved | reset | unclear | thanks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| escalate | 16 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnose | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| billing | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| admin | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| tickets | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| resolved | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| reset | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| unclear | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 5 | 0 |
| thanks | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |

## blind: combined

Paths: `{"critical_rule": 12, "model": 36, "privacy_guard": 2}`. Fallbacks: `{}`. Model errors: `{}`.

| Intent | Precision | Recall | Support |
| --- | --- | --- | --- |
| escalate | 100.0% | 100.0% | 17 |
| diagnose | 90.9% | 100.0% | 10 |
| billing | 100.0% | 100.0% | 4 |
| admin | 100.0% | 100.0% | 4 |
| tickets | 100.0% | 100.0% | 3 |
| resolved | 100.0% | 100.0% | 2 |
| reset | 100.0% | 50.0% | 2 |
| unclear | 83.3% | 83.3% | 6 |
| thanks | 100.0% | 100.0% | 2 |

Confusion matrix: expected rows, predicted columns.

| Expected | escalate | diagnose | billing | admin | tickets | resolved | reset | unclear | thanks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| escalate | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnose | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| billing | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| admin | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| tickets | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| resolved | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| reset | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| unclear | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 5 | 0 |
| thanks | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |

## dev: rules-only

Paths: `{"reference_rules": 104}`. Fallbacks: `{}`. Model errors: `{}`.

| Intent | Precision | Recall | Support |
| --- | --- | --- | --- |
| escalate | 100.0% | 100.0% | 36 |
| diagnose | 100.0% | 100.0% | 18 |
| billing | 100.0% | 100.0% | 12 |
| admin | 100.0% | 100.0% | 11 |
| tickets | 100.0% | 100.0% | 7 |
| resolved | 100.0% | 100.0% | 6 |
| reset | 100.0% | 100.0% | 4 |
| unclear | 100.0% | 100.0% | 10 |
| thanks | 0.0% | 0.0% | 0 |

Confusion matrix: expected rows, predicted columns.

| Expected | escalate | diagnose | billing | admin | tickets | resolved | reset | unclear | thanks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| escalate | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnose | 0 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| billing | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 |
| admin | 0 | 0 | 0 | 11 | 0 | 0 | 0 | 0 | 0 |
| tickets | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| resolved | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| reset | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| unclear | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 |
| thanks | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## dev: model-only

Paths: `{"model": 104}`. Fallbacks: `{}`. Model errors: `{}`.

| Intent | Precision | Recall | Support |
| --- | --- | --- | --- |
| escalate | 100.0% | 97.2% | 36 |
| diagnose | 94.7% | 100.0% | 18 |
| billing | 66.7% | 100.0% | 12 |
| admin | 100.0% | 54.5% | 11 |
| tickets | 100.0% | 100.0% | 7 |
| resolved | 100.0% | 100.0% | 6 |
| reset | 100.0% | 75.0% | 4 |
| unclear | 100.0% | 90.0% | 10 |
| thanks | 0.0% | 0.0% | 0 |

Confusion matrix: expected rows, predicted columns.

| Expected | escalate | diagnose | billing | admin | tickets | resolved | reset | unclear | thanks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| escalate | 35 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnose | 0 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| billing | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 |
| admin | 0 | 0 | 5 | 6 | 0 | 0 | 0 | 0 | 0 |
| tickets | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| resolved | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| reset | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 1 |
| unclear | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| thanks | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## dev: combined

Paths: `{"critical_rule": 36, "model": 64, "privacy_guard": 4}`. Fallbacks: `{}`. Model errors: `{}`.

| Intent | Precision | Recall | Support |
| --- | --- | --- | --- |
| escalate | 100.0% | 100.0% | 36 |
| diagnose | 94.7% | 100.0% | 18 |
| billing | 70.6% | 100.0% | 12 |
| admin | 100.0% | 54.5% | 11 |
| tickets | 100.0% | 100.0% | 7 |
| resolved | 100.0% | 100.0% | 6 |
| reset | 100.0% | 75.0% | 4 |
| unclear | 100.0% | 90.0% | 10 |
| thanks | 0.0% | 0.0% | 0 |

Confusion matrix: expected rows, predicted columns.

| Expected | escalate | diagnose | billing | admin | tickets | resolved | reset | unclear | thanks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| escalate | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnose | 0 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| billing | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 |
| admin | 0 | 0 | 5 | 6 | 0 | 0 | 0 | 0 | 0 |
| tickets | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| resolved | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| reset | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 1 |
| unclear | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| thanks | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## test: rules-only

Paths: `{"reference_rules": 102}`. Fallbacks: `{}`. Model errors: `{}`.

| Intent | Precision | Recall | Support |
| --- | --- | --- | --- |
| escalate | 100.0% | 56.7% | 30 |
| diagnose | 85.7% | 37.5% | 16 |
| billing | 75.0% | 75.0% | 12 |
| admin | 100.0% | 41.7% | 12 |
| tickets | 77.8% | 87.5% | 8 |
| resolved | 100.0% | 16.7% | 6 |
| reset | 100.0% | 100.0% | 4 |
| unclear | 20.9% | 90.0% | 10 |
| thanks | 75.0% | 75.0% | 4 |

Confusion matrix: expected rows, predicted columns.

| Expected | escalate | diagnose | billing | admin | tickets | resolved | reset | unclear | thanks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| escalate | 17 | 0 | 0 | 0 | 1 | 0 | 0 | 12 | 0 |
| diagnose | 0 | 6 | 0 | 0 | 1 | 0 | 0 | 9 | 0 |
| billing | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 3 | 0 |
| admin | 0 | 1 | 2 | 5 | 0 | 0 | 0 | 4 | 0 |
| tickets | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 1 | 0 |
| resolved | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 4 | 1 |
| reset | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| unclear | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 9 | 0 |
| thanks | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |

## test: model-only

Paths: `{"model": 102}`. Fallbacks: `{}`. Model errors: `{}`.

| Intent | Precision | Recall | Support |
| --- | --- | --- | --- |
| escalate | 100.0% | 96.7% | 30 |
| diagnose | 94.1% | 100.0% | 16 |
| billing | 85.7% | 100.0% | 12 |
| admin | 100.0% | 83.3% | 12 |
| tickets | 100.0% | 100.0% | 8 |
| resolved | 100.0% | 100.0% | 6 |
| reset | 100.0% | 75.0% | 4 |
| unclear | 100.0% | 100.0% | 10 |
| thanks | 80.0% | 100.0% | 4 |

Confusion matrix: expected rows, predicted columns.

| Expected | escalate | diagnose | billing | admin | tickets | resolved | reset | unclear | thanks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| escalate | 29 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnose | 0 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| billing | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 |
| admin | 0 | 0 | 2 | 10 | 0 | 0 | 0 | 0 | 0 |
| tickets | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| resolved | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| reset | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 1 |
| unclear | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 |
| thanks | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |

## test: combined

Paths: `{"critical_rule": 17, "model": 85}`. Fallbacks: `{}`. Model errors: `{}`.

| Intent | Precision | Recall | Support |
| --- | --- | --- | --- |
| escalate | 100.0% | 100.0% | 30 |
| diagnose | 100.0% | 100.0% | 16 |
| billing | 85.7% | 100.0% | 12 |
| admin | 100.0% | 83.3% | 12 |
| tickets | 100.0% | 100.0% | 8 |
| resolved | 100.0% | 100.0% | 6 |
| reset | 100.0% | 75.0% | 4 |
| unclear | 100.0% | 100.0% | 10 |
| thanks | 80.0% | 100.0% | 4 |

Confusion matrix: expected rows, predicted columns.

| Expected | escalate | diagnose | billing | admin | tickets | resolved | reset | unclear | thanks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| escalate | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnose | 0 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| billing | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 |
| admin | 0 | 0 | 2 | 10 | 0 | 0 | 0 | 0 | 0 |
| tickets | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| resolved | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| reset | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 1 |
| unclear | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 |
| thanks | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |

## known-cases: rules-only

Paths: `{"reference_rules": 15}`. Fallbacks: `{}`. Model errors: `{}`.

| Intent | Precision | Recall | Support |
| --- | --- | --- | --- |
| escalate | 100.0% | 100.0% | 6 |
| diagnose | 100.0% | 100.0% | 4 |
| billing | 100.0% | 100.0% | 1 |
| admin | 100.0% | 100.0% | 2 |
| tickets | 100.0% | 100.0% | 1 |
| resolved | 0.0% | 0.0% | 0 |
| reset | 0.0% | 0.0% | 0 |
| unclear | 0.0% | 0.0% | 0 |
| thanks | 100.0% | 100.0% | 1 |

Confusion matrix: expected rows, predicted columns.

| Expected | escalate | diagnose | billing | admin | tickets | resolved | reset | unclear | thanks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| escalate | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnose | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| billing | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| admin | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| tickets | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| resolved | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| reset | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unclear | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| thanks | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

## known-cases: model-only

Paths: `{"model": 15}`. Fallbacks: `{}`. Model errors: `{}`.

| Intent | Precision | Recall | Support |
| --- | --- | --- | --- |
| escalate | 100.0% | 100.0% | 6 |
| diagnose | 100.0% | 75.0% | 4 |
| billing | 100.0% | 100.0% | 1 |
| admin | 100.0% | 100.0% | 2 |
| tickets | 100.0% | 100.0% | 1 |
| resolved | 0.0% | 0.0% | 0 |
| reset | 0.0% | 0.0% | 0 |
| unclear | 0.0% | 0.0% | 0 |
| thanks | 100.0% | 100.0% | 1 |

Confusion matrix: expected rows, predicted columns.

| Expected | escalate | diagnose | billing | admin | tickets | resolved | reset | unclear | thanks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| escalate | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnose | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| billing | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| admin | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| tickets | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| resolved | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| reset | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unclear | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| thanks | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

## known-cases: combined

Paths: `{"critical_rule": 6, "model": 9}`. Fallbacks: `{}`. Model errors: `{}`.

| Intent | Precision | Recall | Support |
| --- | --- | --- | --- |
| escalate | 100.0% | 100.0% | 6 |
| diagnose | 100.0% | 75.0% | 4 |
| billing | 100.0% | 100.0% | 1 |
| admin | 100.0% | 100.0% | 2 |
| tickets | 100.0% | 100.0% | 1 |
| resolved | 0.0% | 0.0% | 0 |
| reset | 0.0% | 0.0% | 0 |
| unclear | 0.0% | 0.0% | 0 |
| thanks | 100.0% | 100.0% | 1 |

Confusion matrix: expected rows, predicted columns.

| Expected | escalate | diagnose | billing | admin | tickets | resolved | reset | unclear | thanks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| escalate | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnose | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| billing | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| admin | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| tickets | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| resolved | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| reset | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unclear | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| thanks | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

## options: rules-only

Paths: `{"reference_rules": 4}`. Fallbacks: `{}`. Model errors: `{}`.

| Intent | Precision | Recall | Support |
| --- | --- | --- | --- |
| escalate | 0.0% | 0.0% | 0 |
| diagnose | 100.0% | 100.0% | 1 |
| billing | 100.0% | 100.0% | 1 |
| admin | 0.0% | 0.0% | 0 |
| tickets | 100.0% | 100.0% | 1 |
| resolved | 0.0% | 0.0% | 0 |
| reset | 0.0% | 0.0% | 0 |
| unclear | 100.0% | 100.0% | 1 |
| thanks | 0.0% | 0.0% | 0 |

Confusion matrix: expected rows, predicted columns.

| Expected | escalate | diagnose | billing | admin | tickets | resolved | reset | unclear | thanks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| escalate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnose | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| billing | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| admin | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tickets | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| resolved | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| reset | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unclear | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| thanks | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## options: model-only

Paths: `{"model": 4}`. Fallbacks: `{}`. Model errors: `{}`.

| Intent | Precision | Recall | Support |
| --- | --- | --- | --- |
| escalate | 0.0% | 0.0% | 0 |
| diagnose | 100.0% | 100.0% | 1 |
| billing | 100.0% | 100.0% | 1 |
| admin | 0.0% | 0.0% | 0 |
| tickets | 100.0% | 100.0% | 1 |
| resolved | 0.0% | 0.0% | 0 |
| reset | 0.0% | 0.0% | 0 |
| unclear | 100.0% | 100.0% | 1 |
| thanks | 0.0% | 0.0% | 0 |

Confusion matrix: expected rows, predicted columns.

| Expected | escalate | diagnose | billing | admin | tickets | resolved | reset | unclear | thanks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| escalate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnose | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| billing | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| admin | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tickets | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| resolved | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| reset | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unclear | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| thanks | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## options: combined

Paths: `{"model": 4}`. Fallbacks: `{}`. Model errors: `{}`.

| Intent | Precision | Recall | Support |
| --- | --- | --- | --- |
| escalate | 0.0% | 0.0% | 0 |
| diagnose | 100.0% | 100.0% | 1 |
| billing | 100.0% | 100.0% | 1 |
| admin | 0.0% | 0.0% | 0 |
| tickets | 100.0% | 100.0% | 1 |
| resolved | 0.0% | 0.0% | 0 |
| reset | 0.0% | 0.0% | 0 |
| unclear | 100.0% | 100.0% | 1 |
| thanks | 0.0% | 0.0% | 0 |

Confusion matrix: expected rows, predicted columns.

| Expected | escalate | diagnose | billing | admin | tickets | resolved | reset | unclear | thanks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| escalate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnose | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| billing | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| admin | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tickets | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| resolved | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| reset | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unclear | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| thanks | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
