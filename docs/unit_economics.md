# Unit economics

What one answer, one eval run and one month cost. Only model calls are paid. They run
pay-per-token on Vertex AI, with nothing deployed and no GPU billed by the hour. Retrieval, the
encoder, the guardrails and the citation checks run locally and cost nothing.

> **In one line:** about **a tenth of a cent per question**, **$0.20–0.23 per 62-question eval
> run**, and **about $5 a month** at 100 questions a day plus 10 eval runs. The server costs more
> than the models.

## The models

| role | model | price per 1M tokens (in / out) | when it runs |
|---|---|---|---|
| Answer | `openai/gpt-oss-120b-maas` | $0.09 / $0.36, or $0.15 / $0.60 | 1.34 calls per question (1, or 2 with the retry) |
| Planner | `google/gemini-3.1-flash-lite` | $0.25 / $1.50 | once per new question; eval questions are cached |
| Planner fallback | `google/gemini-3.5-flash-lite` | $0.30 / $2.50 | only when the planner fails |
| Web summary | `google/gemini-3.1-flash-lite` | $0.25 / $1.50 | after a refusal (about 23% of questions); off in evals |
| Judges (evals only) | `google/gemini-3.5-flash-lite` | $0.30 / $2.50 | 2 calls per eval question: correctness, groundedness |

Prices are in USD from third-party trackers as of November 2026; Google bills in CAD and adds
GST/HST. The two gpt-oss figures are the two that sources disagree on. Check
[Google's pricing page](https://cloud.google.com/vertex-ai/generative-ai/pricing) before relying
on them. The free-tier judges after the first in the chain cost nothing.

## Per call

| call | tokens (in / out) | per 1,000 calls |
|---|---|---|
| Answer | ~3,400 / ~310, measured on the agent rows (hidden reasoning included) | **$0.42 – $0.70** |
| Planner | ~120 / ~50 (est.) | **$0.11** |
| Planner fallback | ~120 / ~50 (est.) | **$0.16** |
| Web summary | ~1,500 / ~250 (est.) | **$0.75** |
| Judge | ~600 for correctness, ~3,500 for groundedness / ~300 (est.) | **$1.37** |

The answer's input is almost all retrieved excerpts, so context size is the main lever on the
per-question cost. Groundedness judging re-reads every excerpt, which is why a judge call costs
more than an answer.

## Per question and per eval run

| unit | what it includes | cost |
|---|---|---|
| A live question | 1.34 answer calls, 1 planner call, a web summary 23% of the time | **$0.0008 – $0.0012** |
| A 62-question eval run | ~83 answer calls ($0.035–0.06) and 124 judge calls ($0.17) | **$0.20 – $0.23** |

Judges are about 80% of an eval run.

## Per month

| traffic | live questions | eval runs | models total |
|---|---|---|---|
| 100 questions a day | 3,000 ($2.5–3.6) | 10 ($2.0–2.3) | **~$4.6 – 5.9** |
| 300 questions a day | 9,000 ($7.6–10.8) | 10 ($2.0–2.3) | **~$9.6 – 13.1** |
| 10,000 questions a month | 10,000 ($8.4–12.0) | 10 ($2.0–2.3) | **~$10.4 – 14.3** |

Live traffic scales linearly. Eval runs are a fixed cost, so at 10,000 questions a month they fall
from about 40% of the bill to 15–20%.

For comparison, the one server in the [production plan](production_plan.md) is estimated at
$25–30 a month, so hosting, not the models, is the larger bill.

## Checked against the real bill

The work behind the current benchmark came to **$1.36** on Google Cloud: 9 generation rows, 6 of
them judged on Vertex, plus a few dozen test calls. The token math above gives $1.33 at the lower
gpt-oss price and $1.54 at the higher one. If the bill is in USD, it points to the lower price.

Google's cost estimator cannot price `gpt-oss-120b` MaaS; it only models Gemini. Its first run,
$9.73, priced the answers as a Gemini Enterprise "Orchestrator". Asked for the Gemini calls alone,
it matches the token math above to the cent:

| per month | Google estimator | this doc |
|---|---|---|
| Planner + web summary, 3,000 questions | $0.83 | $0.84 |
| Planner + web summary, 10,000 questions | $2.78 | $2.78 |
| Judges, 10 eval runs | $1.69 | $1.69 |
| Agent Platform compute | $0.04 / $0.14 | $0 (not used: calls go straight to the endpoint) |
| gpt-oss answers, 3,000 / 10,000 questions | not priced | $2.03–3.37 / $5.94–9.91 |
| **Total, 3,000 questions** | $2.57 (Gemini only) | **$4.6 – 5.9** |
| **Total, 10,000 questions** | $4.62 (Gemini only) | **$10.4 – 14.3** |

To see the real split, open Cloud Billing → Reports and group by SKU.

## Not counted

Hosting, LangSmith, exchange-rate swings, and any change in Google's prices. Token counts marked
est. come from prompt sizes, not from the bill; the next eval run's usage can replace them.
