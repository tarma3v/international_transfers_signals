# Round 3 protocol: barrier probability and adaptive experts

Date frozen: 2026-09-04

## Decision and primary metric

The product decision is whether to emit a currency-specific alert on publication date
`t`. The target is `fav_h5 = 1` when the current normalized CBR rate is no greater than
each of the next five CBR publications. The primary metric is leakage-safe future-only
lift at a realized alert frequency of 1--2 alerts per currency-week. Secondary metrics
are precision, frequency, annual/corridor minimum lift, and uncertainty across time
blocks.

## Frozen baselines

- Round-1 multiscale anchor: final 2024--2026 retrospective lift 1.2953.
- Round-2 equal mixture: shock validation 2022--2023 lift 1.2868; final retrospective
  lift 1.3276.
- Round-2 soft regime router: shock validation lift 1.2431; final retrospective lift
  1.3249.
- Round-2 global trailing weights: final retrospective lift 1.3831, but weaker general
  validation and not yet statistically distinguished from the anchor.

These numbers are frozen before round-3 model selection. Round 3 must not select a
method on the 2024--2026 retrospective interval.

## New hypotheses

1. Treat the label as a five-step first-passage/barrier event rather than an ordinary
   binary class. Estimate conditional hazards for the first future rate below today's
   rate and combine them into five-step survival probability.
2. Decompose the panel into a common return factor and currency-specific residuals,
   then estimate the probability that the simulated cumulative path remains nonnegative.
3. Combine genuinely diverse experts with delayed-feedback online learning. Update
   weights only after the five-publication outcome is fully observable; test global,
   per-currency, and change-aware variants.
4. Prefer expert consensus when disagreement signals epistemic uncertainty. Test
   trimmed means, lower-tail consensus, and disagreement penalties.
5. Use causal rolling normalization, path-shape, drawup/drawdown-duration, dispersion,
   and factor-residual features to improve cross-regime transfer.

## Validation design

- All features at date `t` use information available no later than `t`.
- Training rows are purged so that their target reach date is strictly before the next
  scoring interval.
- Model-family choices are screened by prequential/rolling-origin results through 2020.
- 2022--2023 is the adverse-regime validation gate.
- 2024--2026 is shown only as a final retrospective stress test and is not used for
  choosing hyperparameters, feature sets, thresholds, or ensemble weights.
- Because earlier rounds have already inspected every era, no claim of a pristine new
  holdout is allowed. Round-3 evidence is retrospective and must be labelled as such.

## Success criteria

A candidate is a credible improvement only if it:

1. beats the frozen anchor on both the pre-2021 rolling evidence and 2022--2023;
2. stays within 1--2 alerts per currency-week;
3. avoids catastrophic corridor/year failures;
4. survives block-bootstrap uncertainty and a multiplicity-aware comparison; and
5. has an auditable causal information set.

## Work sequence

1. Build a stitched out-of-fold prediction panel and audit label availability.
2. Run consensus and delayed-feedback online-mixture experiments.
3. Build and test a proper discrete-time hazard model.
4. Test factor/residual barrier models and new causal normalized features.
5. Perform error/regime diagnostics, uncertainty analysis, and failure analysis.
6. Write a cited report and render/visually verify the PDF deliverable.

