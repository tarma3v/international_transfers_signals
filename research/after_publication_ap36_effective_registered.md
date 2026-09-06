# AP36-E frozen protocol: mature-only local Brier Hedge over three predictors

Registered 2026-09-06 before AP36 ensemble signals or scorecards. AP34 Ridge
survival and AP35 distributional CatBoost are weaker globally but provide
different causal OOS rankings. AP36 tests one outcome-aware regime ensemble;
there is no model or policy grid.

Frozen experts are AP26 y20-shrink200 pace score, AP34 residual-survival score
and AP35 distributional-CatBoost score. Convert each to its own same-currency
causal percentile using the standard prior250/warmup40 rank. At each decision
date estimate each expert's Brier loss against effective-reference y20 using
only eligible rows in the previous730 calendar days whose publication-h20
maturity is earlier than the current date minus two days. Global loss shrinks
toward.25 with strength40; each currency loss shrinks toward that global loss
with strength40. Form softmax weights exp(-25*local_loss) and combine only finite
expert ranks, renormalizing their weights. No current or immature label enters.

Pass this sole Hedge score as pace expert through the exact frozen AP21
dual-paced-month controller with frozen rolling primary and reserve. Treat that
output as core and apply the exact frozen AP33 calendar router with frozen AP23
fallback. No alternative window, prior, eta, label, expert subset or policy
threshold is evaluated.

Select the sole fresh candidate on early2023 with unchanged joint gates; frozen
AP33 is the registered fallback. Opened2024-2026 remains retrospective. Audit
expert inputs/ranks, every mature mask, Brier loss/count/weight, policy state,
future-label/expert corruption prefix invariance and paired20/50-date uncertainty.
Acceptance remains min h3/h5/h10/h20 lift>2.4,min currency rate>=1,zero empty
complete months,max2/week. H1 is known validity only; historical receipts remain
calendar-assumed and bank execution is unvalidated.
