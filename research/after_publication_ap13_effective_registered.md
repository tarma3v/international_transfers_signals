# AP13-E frozen protocol: recent ExtraTrees and causal cadence repair

Registered 2026-09-06 before fitting or scoring any AP13 candidate. AP12 later
results are opened: its full ExtraTrees 2.475 is a frozen control and cannot be
called a fresh selected winner. Primary reference remains TODAY-EFFECTIVE CBR;
tomorrow's announced fixing is the known first next observation. Historical
18:00 receipts remain CALENDAR-ASSUMED and executable bank prices unvalidated.

## Frozen support and causality

Reuse AP12's 5,755 events, same 133-feature 18:30 matrix with20-minute market
delay, same effective outcomes, known-down veto, publication-h20 maturity cap,
2-calendar-day embargo,17 quarterly origins, early2023 selection and opened
later2024-2026 scorecard. Conditional y5 is learned only on announced>=current
rows. h1 is a deterministic validity check; selector ranks h3/h5/h10/h20 only.
No AP13 threshold or route may use a target whose h5 maturity date is not before
the current decision date.

## Six new conditional scores

Use AP12 ExtraTrees hyperparameters unchanged:400 trees, depth8, leaf25,
max_features .6, seed20260906. No depth/leaf grid.

1. `extra_roll2`: training prefix restricted to the previous730 calendar days.
2. `extra_roll3`: previous1095 calendar days.
3. `extra_decay730`: full prefix with deterministic half-life730-day
   sample weights `2^(-age/730)`.
4. `local_extra`: per-currency full-feature ExtraTrees when >=100 eligible rows,
   shrunk toward frozen global AP12 ExtraTrees by n/(n+150); otherwise global.
5. `router_extra_local`: HistGB meta-classifier on the fixed AP12 compact59
   features. Training labels use only prior quarterly OOS expert predictions and
   choose which of frozen AP12 ExtraTrees/local-Hist has smaller squared error.
   Query score is p(router)*Extra +(1-p)*local; below200 usable meta rows or one
   class, use a50/50 blend.
6. `brier365_extra_local`: no fitted router. At each date use eligible expert
   predictions whose h5 outcomes matured strictly before that date and whose
   event dates are within365 days. After at least100 rows, blend experts with
   weights softmax(-20*Brier); otherwise50/50. All currencies of one date share
   the same prior-only weights.

All expert inputs are frozen quarterly OOS AP12 predictions; AP13 never refits
or rewrites them.

## Three online policies per new score

All policies keep a prior250 per-currency CDF, warmup40, strict primary top30%,
known-down veto and max2 signals per ISO week.

1. `primary`: fire only on the primary score.
2. `reserve7`: primary rule, plus a reserve opportunity after >=7 calendar days
   of silence when fixed known70/hazard30 causal rank is above .65.
3. `month24`: primary rule, plus a month rescue from day24 onward if the currency
   has not fired in that calendar month and known70/hazard30 rank is above .50.

Primary and reserve ranks are computed before the current row. Fallback never
ranks unseen future days of a week/month. Month rescue may still fail if no
eligible event occurs after day24. Carry eight frozen AP12 controls: AP12 full
Extra primary, AP12 early-selected local Hist, AP12 known70/hazard r27.5,
AP11 hazard urgent, AP10 known-z urgent, AP1 exact, AP1 cap2 and known-sign cd3.
Total:18 fresh policies +8 frozen controls.

## Selection and evidence

Among18 fresh policies only, early joint gates are: min adjusted lift over
h3/h5/h10/h20 >=1.3; per-currency average rate1..2/week; symmetric-benefit
bootstrap lower bound>0 on all four unknown horizons; max2 signals/ISO-week;
future-only benefit >=80% of known-sign cd3; and zero empty complete calendar
months in the early scope. Rank by minimum unknown-h lift, then mean lift, then
smaller maximum calendar gap. Write selection before later scorecards.

Report all h, hit/base/lift, cadence and empty months, benefits, model Brier,
paired20-date bootstrap versus AP12 full Extra, local selected, AP11 hazard,
AP10 known-z and AP1 exact/cap; selected and best late diagnostic50-date
sensitivity. Audit source hashes, old predictions unchanged, rolling/weighted
train masks, meta OOS/maturity rows, same-date Brier weights, all score and
controller prefixes, veto and weekly cap. Preserve negative results, update
PDF/checkpoint and push only ivan-experiments. This packet advances but does not
complete the indefinite goal.
