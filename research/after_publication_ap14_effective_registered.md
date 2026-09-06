# AP14-E frozen protocol: light causal cadence repair

Registered 2026-09-06 before computing any AP14 early or later scorecard.
AP12/AP13 results are already open and remain frozen controls. The reference is
TODAY-EFFECTIVE CBR; the announced fixing for tomorrow is a known first next
observation. Historical receipt near18:00 remains a calendar assumption, not a
verified timestamp, and CBR benefit is not an executable bank saving.

## Frozen data and target

Reuse AP13's5,755 rows,133 features,17 quarterly OOS origins, effective-price
outcomes, publication-h20 maturity cap and2-day embargo. The known-down veto is
mandatory. h1 is already determined after receipt and is validity-only; model
selection uses h3/h5/h10/h20. Early2023 selects; opened2024-2026 is a later
retrospective diagnostic, never a fresh holdout.

## Four fixed scores

No model is refit in this cadence packet. Reuse three quarterly OOS scores
unchanged and add one deterministic ensemble before evaluating it:

1. `extra_roll2`: AP13 ExtraTrees trained on the previous730 calendar days.
2. `local_extra`: AP13 per-currency ExtraTrees shrunk to frozen global AP12.
3. `blend_roll2_local`: arithmetic50/50 average of the two scores above.
4. `extra_ap12`: frozen expanding-prefix AP12 ExtraTrees.

The blend has no learned weight and cannot use early or later outcomes. It tests
whether the best all-h recent model and the best point h5 local model complement
each other without a router.

## Five light online cadence policies per score

Every policy converts the score to a prior250-row, per-currency causal rank with
warmup40, uses strict comparisons, applies known-down veto and caps each
currency at2 signals per ISO week. No rule ranks future days within a week.

1. `top325`: primary rank>.675 (top32.5%).
2. `top35`: primary rank>.65 (top35%).
3. `silence14_r80`: original top30%, plus a reserve only after at least14
   calendar days without a signal and prior-only known70/hazard30 rank>.80.
4. `silence21_r70`: original top30%, plus a reserve only after at least21
   calendar days without a signal and reserve rank>.70.
5. `adaptive105`: threshold adjusts from past decisions only. After at least56
   calendar days, compute the currency's signals in the prior182 days divided
   by the actually observed trailing duration in weeks. Use threshold.65 below
   rate.95, .675 below1.05, otherwise.70. Before56 days use.675. This controller
   does not inspect outcomes, so label maturity cannot leak into its state.

This is20 fresh policies. Frozen controls: AP12 full Extra, AP13 rolling2/local
primary, AP13 rolling2 reserve7, AP13 early-selected router month24, AP10
known-z, AP11 hazard and AP1 cap2.

## Selection and evidence

Among fresh policies only, early joint gates are: minimum adjusted lift over
h3/h5/h10/h20>=1.3; per-currency rate1..2/week; symmetric-benefit block
bootstrap lower bound>0 on all unknown horizons; max2 signals/ISO-week;
future-only benefit at least80% of frozen known-sign cd3; zero empty complete
calendar months. Rank passers by minimum unknown-h lift, then mean lift, then
smaller maximum calendar gap. If none pass, choose the highest minimum-lift
fresh policy satisfying rate/cap only and explicitly label the gate failure.
Write selection before constructing the later scorecard.

Report all horizons, benefits, frequency by currency, gaps/months, paired20- and
50-date CIs versus AP12/AP13/AP10/AP11/AP1 controls, and exact signal additions
relative to each score's AP13 top30 primary. Audit source hashes, unchanged
scores/outcomes, rank prefixes, dynamic thresholds, veto, weekly cap and
future-score corruption invariance. Preserve negative results and do not claim
that an opened-period winner is independently validated.
