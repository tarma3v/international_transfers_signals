# Temperature T10 frozen protocol — 10:00 routed probability and benefit

Registered 2026-09-06 after T9 was completed and before T10 metrics were
computed. The 2025--2026 outcome table remains open retrospective evidence.

T9 showed zero same-day CNY candle coverage before 09:30 in 2025--2026 but
79.2% coverage at 10:00. T10 therefore makes no earlier market claim. It adds
exactly one optional 10:00 snapshot and retains T4 history-only when a completed
same-day CNY candle is absent.

The probability expert is chosen separately by horizon using only the frozen
T9 2024 screen and calibrated Brier loss. The resulting fixed map is h1 hybrid
HGB; h3/h5/h10/h20 transparent CNY rank. Do not reselect this map on 2025--2026.

For expected future-only CBR basis points, fit one quarterly Ridge per horizon
with the same T6 parameters, post-2022 start, half-life, target maturity and
two-day embargo. Inputs are frozen T4 features, T9 early-market state, the
selected raw score and selected calibrated probability. The routed output uses
this prediction only when the 10:00 CNY candle is physically available;
otherwise both probability and benefit are copied exactly from T4/T6 premarket.

Report full routed probability Brier/log loss/ECE/AUC/AP and benefit MAE/RMSE/
Spearman/sign accuracy on 2024 and open 2025--2026. Verify exact fallback,
selected-head identity on available rows, source hashes, a complete output
rebuild, and future target/maturity prefix invariance. This is a narrow gap
closure from 10:00 to 10:30, not evidence that the modern market is informative
from 07:00.

For both Brier and absolute-error deltas versus the exact T4/T6 fallback, report
fixed-seed paired circular date-block bootstrap intervals at 20 and 50 dates.
Negative delta means T10 is better.
