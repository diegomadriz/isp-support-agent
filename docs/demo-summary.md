# Offline demo results

| Scenario | Verified | Verdict | Tickets | Staff alerts |
| --- | --- | --- | --- | --- |
| healthy | True | local_wifi_or_device | 0 | 0 |
| weak_radio_signal | True | weak_radio_signal | 1 | 0 |
| last_mile_down | True | last_mile_down | 1 | 0 |
| upstream_packet_loss | True | upstream_degraded | 1 | 0 |
| area_outage | True | area_outage | 0 | 1 |
| probe_timeout | True | inconclusive | 1 | 0 |
| existing_open_ticket | True | weak_radio_signal | 1 | 0 |
| billing | True | — | 0 | 0 |
| unknown_customer | False | — | 0 | 0 |
| failed_verification | False | — | 0 | 0 |
| unclear | True | — | 0 | 0 |
| critical_mixed | True | — | 0 | 1 |
| administrative | True | — | 1 | 0 |
| confirmed_resolution | True | — | 1 | 0 |
| slow_internet_first | True | weak_radio_signal | 1 | 0 |
| clarify_then_diagnose | True | — | 0 | 0 |

Runtime paths across all demo turns: `{"critical_rule": 1, "identity_or_welcome": 18, "interrupt_resume": 32, "reset_control": 1, "stub": 34}`.
Identity/welcome and explicit reset are graph control paths. Other noncritical classification uses the deterministic stub. These are behavior transcripts, not model accuracy measurements.
