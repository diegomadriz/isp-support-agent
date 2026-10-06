# Customer reply rewriting

The model may reorder template sentences and add one short courtesy phrase. Acknowledgements open the reply; only the thanks may close it. Facts stay word for word. Actions and questions keep their order. The closed courtesy list prevents extra content; greetings are allowed only in the first reply, and angry messages allow only a calm acknowledgement. No reply contains two apologies.

Both arms use live local gemma4 through the current graph. Classification gets up to six seconds within an eight-second turn budget. Rewriting starts only with at least 1.5 seconds left; otherwise the template is sent. Every recorded conversation includes identity and any ticket consent.

| Metric | Result |
| --- | --- |
| Conversations | 70 |
| Sent replies | 381 |
| Rewrite attempts | 294 |
| Accepted | 262/294 (89.1%) |
| Delivered leaks | 0 |
| Required facts | 381/381 |
| Detected tone violations | 0 |
| Complete turn latency median | 4036.667 ms |
| Complete turn latency p95 | 8004.946 ms |
| Verified turn latency median | 5534.676 ms |
| Verified turn latency p95 | 8005.456 ms |
| Added latency median | 3447.352 ms |
| Added latency p95 | 4033.68 ms |
| Mean length change | 3.808% |

Result codes: `{"accepted": 262, "courtesy_out_of_place": 1, "missing_required_fact": 4, "timeout": 27}`.
Raw tone flags: `{}`.
Paired turns with a different node path: 0.

Timing starts at the chat turn call and ends when the reply returns, including classification, graph work and rewriting. It excludes the WhatsApp queue and outbound delivery. Replay uses the recorded clock to reproduce rewrite eligibility; it does not measure live speed. Added latency compares matched input turns from the two arms, whose model predictions can differ.

The earlier 64-conversation trial accepted 268 of 271 drafts and measured p95 added latency of 4.29 seconds, missing the three-second target set before the trial. Classification was held fixed in that trial. After seeing the result, the limit was raised to eight seconds because a few seconds is normal in a WhatsApp chat. Warmer replies were preferred in a side-by-side read, without item-by-item scores. That was a decision to enable rewriting, not a pass under the original target.

Config SHA-256: `330fcbd7f199fb14cee93bafec5f5cdaab017749e4d4db794223baed3253148f`. Raw responses, requests, turn paths and measured timings are in `eval/rewrite-recordings.json`. The offline gate rejects stale inputs and replays both arms through the current graph. The review form and answer key are kept outside the public repository.
