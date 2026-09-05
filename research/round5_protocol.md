# Round 5 protocol: ordinary pre-publication signal

Frozen before running the new candidate families on 2022--2026.

## Question

Can a signal using only CBR publications available at decision time exceed
future-only lift 1.30 for `fav_h5`, while emitting 1--2 alerts per currency-week?
The next effective CBR rate is forbidden in every feature and score.

## Honest information flow

- A row dated `t` may use prices and calendar information no later than `t`.
- For test year `Y`, target-bearing training rows must resolve before
  1 January `Y-1`; year `Y-1` is calibration-only.
- Thresholds are learned separately by currency from calibration scores.
- Candidate architecture and policy are selected on 2017--2020 only.
- The exact selected candidates are gated on 2022--2023.
- 2024--2026 is opened only after the gate and is labelled retrospective,
  because previous rounds already inspected it.
- No result on the existing history is called a new pristine holdout. Only a
  future frozen run can supply that evidence.

## New candidate families

1. **Self-normalized path models.** Raw trailing one-step returns are divided by
   a causal robust volatility estimate. This attempts to make 2014, 2022 and
   the quieter recent regime comparable instead of asking a tree to learn the
   scale change.
2. **ROCKET-style random convolutions.** Fixed random kernels summarize motifs
   in the last 64 normalized returns; a regularized linear classifier learns
   on those summaries. Kernels are generated from a fixed seed and never see a
   target.
3. **Direct multi-horizon survival consensus.** Five classifiers estimate the
   probability that the current rate remains favourable through horizons
   1,...,5. Their geometric/minimum consensus is compared with direct `h=5`
   classification.
4. **Ordinal/floor models.** Regress the number of future publications not
   cheaper than today and the standardized worst future return. This uses more
   target information than the single binary label without changing the
   runtime information set.
5. **Causal analogue bootstrap.** A rolling, past-only library of normalized
   path states estimates the five-step no-crossing rate among nearest previous
   states. No future state is admitted to the library.
6. **Era-expert consensus.** Models fitted on separate historical eras vote by
   calibration-normalized ranks. Median and lower-tail consensus test whether
   stable agreement transfers better than a single pooled model.

## Selection rule

For each architecture, choose its alert-rate/threshold policy on 2017--2020 by
the minimum of aggregate lift, macro-year lift, minimum year lift and minimum
currency lift, subject to:

- aggregate frequency 0.90--2.10;
- per-year frequency 0.75--2.25;
- minimum currency frequency at least 0.65;
- positive future-only benefit.

Advance at most two variants per family to 2022--2023. A candidate clears the
research gate only if aggregate and macro-year lift are both at least 1.30,
future-only benefit is positive, and the frequency constraints still hold.

## Reporting

Always report aggregate lift, macro-year lift, annual and currency minima,
frequency, future-only benefit, block-bootstrap uncertainty versus the locked
multiscale anchor, and the number of tried policies. The largest searched
number is explicitly post-hoc and is not the selected result.
