# Round 4: conditional publication-time and regime research

This round answers three separate product questions. They must not be mixed in
one headline because their information sets differ.

## Frozen chronology

- feature/model development: data available before each calibration year;
- general validation: 2017--2020;
- post-shock validation: 2022--2023;
- retrospective audit only: 2024--2026 (this block has already been inspected
  in earlier rounds and is not a fresh holdout);
- the h=5 label is admitted to training only when its actual fifth publication
  precedes the calibration period;
- thresholds are learned from the preceding calendar year, per corridor, and
  are never fitted to the scored year;
- candidate policy settings are selected on general validation, checked on the
  post-shock block, and only then applied unchanged to 2024--2026.

The requested frequency corridor remains 1--2 alerts per currency per week.
Headline comparisons report lift, future-only benefit, per-year and
per-currency minima, and a four-week moving-block bootstrap on the retrospective
block.

## A. Ordinary signal, before the next CBR publication

Only features available at the current row may be used. The new family is a
recency-weighted hierarchical empirical-Bayes state model. It estimates the
event rate in partially pooled states made from price position, medium-term
returns, volatility, currency, and calendar month. Sparse cells shrink to the
global prior. Several predeclared shrinkage, half-life, lower-confidence-bound,
and anchor-blend settings are screened on 2017--2020 only.

Target: `fav_h5`, i.e. today's rate is no greater than every one of the next
five published rates.

## B. Signal after today's publication of tomorrow's effective CBR rate

This is a distinct timestamp-aware product. At decision time the full next
effective CBR cross-section is public. The model may therefore use the next
rate, the known cushion from the current rate, and causal features recomputed at
that newly published point. It may not use any later rate. Rows for which the
known next rate is already below the current rate are structurally ineligible
for `fav_h5`.

Candidates include the existing known-next gate, deterministic cushion/range
scores, and global logistic, histogram boosting, extremely randomized trees,
and XGBoost conditional models. The h=5 target is unchanged; the model predicts
whether the remaining four unknown publications stay above the current anchor.

## C. Window-closing signal

Target: `close_h5`, i.e. the rate five publications later is above today's
rate. The timestamp is the ordinary pre-publication information set. Extended
feature logit, histogram boosting, ExtraTrees, XGBoost, the upper-range anchor,
and the same hierarchical state family are compared under the frozen
chronology.

## Leakage boundary

The ordinary and window-closing matrices never contain `i+1`. The publication-
time matrix contains `i+1` deliberately and is stored/evaluated separately. It
is valid only after the Bank of Russia has published the next effective rate;
it must never be described as an ordinary day-ahead forecast.
