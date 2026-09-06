# AP6 frozen packet: causal OOS stacking and conditional errors

Registered 2026-09-06 before AP6 fitting or scorecards. Previous AP5 turn made
progress: tested 66 policies, saved all results and pushed c66f4b0. No new policy
passed the fixed early joint gate. This packet changes the learning objective,
not the thresholds or the repeatedly opened retrospective period.

## Information and evaluation contract

Reuse AP2-D20/AP3/AP4/AP5 announcement events, publication-reference labels and
date support. Decision 18:30 MSK, CALENDAR-ASSUMED CBR receipt, market bars with
20 minute delay. The already announced fixing is the reference price; the next
h observations are unknown outcomes. Not historical receipt certification,
fresh holdout, or executable bank savings. h=1/3/5/10/20.

Base inputs are the six saved issued OOS experts: arcsinh CNY basis indicator,
clipped logits of direct HistGB/ExtraTrees and survival HistGB/global/local
logistic at h5. Base quarterly origins precede their predictions. No base model
is refit on a meta training row to manufacture a cleaner prediction.

Every meta model updates monthly from July 2022. Only source-complete rows with
all h20 labels mature STRICTLY before origin minus two calendar days can train.
Standardization fits on those training rows only. Rolling variants use the
previous 730 calendar days. No shuffle CV, no same-day feedback, no weekly top-k.
Early 2023 selection is written before the later scorecard. Repeatedly opened
2024-2026 remains retrospective, including legitimate walk-forward past-train.

## Small regime/context vector

Known change z; announced return5/return20 divided by available vol20;
log1p(vol20); price range20/range90; local-minus-common and peer-change std
divided by available vol20; sine/cosine weekday; month-end and pre-New-Year;
CNY latest basis, late change, quality and missingness; local latest basis,
quality and missingness; fixed currency one-hot; current expert disagreement
and CNY-versus-Hist disagreement. All point-in-time values, no future errors,
no year-specific rule or KZT/UZS-2026 split. The source-only controls exclude
expert disagreement, so they genuinely do not receive expert predictions.

## Sixteen model scores, fixed hyperparameters

1. Nonnegative-coefficient logistic on six experts, expanding and rolling730.
2. Unconstrained logistic on experts, expanding.
3. Logistic on experts+context, expanding and rolling730.
4. Separate currency logistic on experts, and n/(n+250) local/global shrinkage.
5. HistGB on experts; on experts+context expanding and rolling730.
6. HistGB regression on experts+context for mean of five survival labels.
7. Same-currency pairwise linear ranking on experts+context: positive/negative
   mature h5 labels within 60 calendar days, up to eight nearest negatives per
   positive, no sampling from the test. L2 penalty, train-only scaling. The
   linear score is not a calibrated probability.
8. Global HistGB correction of the issued AP5 equal rolling-calibrated h5
   probability; residual target y5 minus that issued probability.
9. Global HistGB correction of the local logistic stacker's own earlier OOS
   predictions. Train on genuinely issued mature local scores only. No
   in-sample local residuals; save origins, masks and corrections.
10. Source-only logistic and HistGB context controls.

Global minimum200 mature rows, local60. Logistic uses intercept, coefficient
L2 strength equivalent to C=.1; positive constraints only in family1. HistGB
max_iter100, learning_rate.05, max_leaf_nodes7, min_samples_leaf40, l2=10,
early_stopping=False, fixed seed. Regression uses squared error. Pairwise
logistic has no intercept and C=.1. No hyperparameter sweep within families.
If insufficient history or one class, use the available AP5 equal probability;
local fallback uses the current global expert logistic. Residual fallback is
zero correction. Probability residuals clip to [0,1]. Local correction uses
the full learned correction, no late weight search.

Each of the sixteen scores gets: unchanged AP3 urgent_cap2 controller; a 50%
CNY prior-CDF anchor; a 25% score prior-CDF correction to the existing AP4 raw
rank-mixture score. Plus exact AP3, AP4 and AP5 equal rolling controls: 51
policies total. No controller change. AP4 early joint selection is reused
unchanged, including all-h lift>=1.3, per-currency rate1..2, positive symmetric
benefit lower CI, max2/week, and future benefit at least80% of AP3 at every h.

## Required evidence

Persist all scores, signals, meta inputs, local origins, train logs, linear
coefficients, source hashes. Tests: strict maturity boundary, future-label and
future-feature corruption, prefix-stable residuals, train-only scaling,
positive coefficients, pair construction, local shrinkage, unchanged controls.
Compare all policies on identical support, early failures, paired bootstrap
against AP3/AP4 and source-only/equal/local controls, 20/50 date blocks, year
and currency slices, frequency gaps and probability error. Selected model is
not promoted simply for a better later number. Retain all negative results,
update summary/checkpoint/PDF, run suite, push only ivan-experiments.
