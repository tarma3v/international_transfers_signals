# AP40-E frozen protocol: causal weekly optimal-stopping logistic model

Registered 2026-09-06 before AP40 predictions, signals or scorecards. AP37 is
the frozen strict leader. AP40 evaluates exactly one model and one sequential
policy; there is no feature, regularization, threshold or controller grid.

An opportunity is an eligible row where frozen AP26 core or frozen AP23 fallback
fires. Its utility is mean(y3,y5,y10,y20). Within each currency and ISO week,
the take-now label is 1 when current utility is at least the maximum utility of
all later opportunities in that week; it is 1 when no later opportunity exists.
Label maturity is the latest publication-h20 maturity among the current and all
later opportunities used by that label. Training requires this maturity to be
earlier than quarterly origin minus two days. Future opportunity membership and
outcomes are target construction only, never inference features.

Current-row features are frozen before fit: three AP26/AP34/AP35 causal OOS
ranks; their mean/min/max/std and top30 support count; announced/current change;
weekday sine/cosine and days to Friday; core/fallback flags; AP37 fallback local
precision minus causal overall precision; frozen AP37 prior trailing rate,
log1p capped days since signal, prior AP37 week usage, current-month-empty flag
and day-of-month/31. All are available before the current decision. Missing
values use train medians; StandardScaler is train-only.

Fit one LogisticRegression per quarterly OOS origin with C=0.1, lbfgs,
class_weight=balanced and max_iter=1000. Model rows are mature past opportunity
labels only. Logistic sample weights decay with fixed half-life 730 days. If
fewer than 100 train labels or one class is present, predict 0.5. The sole
take-now threshold is 0.5.

Run one sequential router from scratch. AP26 core is preserved unless all are
true: predicted take probability is below 0.5, weekday is before Friday,
trailing-365 selected rate is at least 1, gap is below 10 days, and the current
currency-month already has a signal or day is before 24. Friday, rate-deficit,
silence-10 and month24 are hard take guards. A vetoed core does not enter state.
For non-core rows reuse exact AP37 mature-precision late-week fallback and
unconditional silence-10 rescue. Warmup84 and sequential max2/ISO-week remain.

Select the sole candidate on early 2023 under unchanged gates; AP37 is the
registered fallback. Opened 2024-2026 remains retrospective. Audit target and
maturity construction, all current-only features, all quarterly refits,
imputation/scaling/weights, router state, future feature/label/source corruption
prefix invariance and paired 20/50-date uncertainty. Acceptance remains min
h3/h5/h10/h20 lift above 2.4, min currency rate at least 1, zero empty complete
months and max2/week. H1 is validity only; receipt timestamps and bank execution
remain unvalidated.
