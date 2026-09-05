# Round 5: ordinary signal without tomorrow's CBR rate

Run date: 2026-09-04. Target: `fav_h5(t) = 1` when the rate at publication
`t` is no greater than every one of the next five published rates. The decision
uses information available no later than `t`. The next published/effective CBR
rate is never a feature.

## Bottom line

There is now a **causal, chronological metric pass**, but not a statistically
or operationally conclusive pass.

The best frozen candidate is `quarterly_reset_hist_anchor50`: a 50/50 rank
blend of a quarterly reset HistGradientBoosting model and the transparent
multiscale price anchor. Its alert policy was chosen on 2024 and then left
unchanged: target the upper 18% of scores using a trailing 120-publication
per-currency cutoff, with no cooldown.

| Evaluation | Signals | Alerts / currency-week | Lift | Future benefit, bps |
|---|---:|---:|---:|---:|
| 2024 policy screen | 295 | 1.167 | 1.224 | +36.8 |
| 2025 confirmation | 231 | 0.911 | **1.463** | +4.2 |
| 2026 chronological audit, through 2026-09-02 | 167 | 1.039 | **1.391** | +68.4 |
| 2025--2026 combined | 398 | 0.940 | **1.445** | +31.1 |

The combined macro-year lift is 1.427 and both individual years exceed 1.30.
All five currencies have combined lift above 1.19 and frequency between 0.885
and 0.992. In the partial 2026 audit every currency individually exceeds lift
1.30.

This result does **not** use tomorrow's rate and every prediction is causally
out of sample. It is still a retrospective research result: earlier rounds had
already inspected 2024--2026, so no untouched holdout remains.

## Why it works

The learned component uses 258 causal features:

- 54 compact price, volatility, cross-currency and cyclic calendar features;
- 204 self-normalized path summaries of the currency, the common five-currency
  factor and the currency residual, including lags, trends, drawdowns,
  autocorrelation and low-frequency spectral amplitudes.

At the start of every calendar quarter, HistGradientBoosting is refitted on
post-24.02.2022 observations whose entire five-publication target has already
resolved. It has shallow nine-leaf trees, strong L2 regularization and large
leaves. This handles the structural break without allowing old regimes to
dominate the current model.

The anchor is
`0.5*pct_range_90 + 0.3*pct_range_30 + 0.2*pct_range_180`. It is weak but
stable and transparent. Each component is converted to a per-currency rank
against past calibration scores, then the two ranks are averaged. The learned
model supplies nonlinear regime/context information; the anchor prevents the
reset model from overreacting. At the winner's identical alert policy,
2025--2026 lift is 1.177 for reset Hist alone, 1.378 for the anchor alone and
1.445 for their blend.

## Important stability warning

The formal aggregate pass is temporally clustered. In 2025 the winner emits no
signals in Q1 or Q2, 178 in Q3 and 53 in Q4. In 2026 it emits 26, 26 and 115 in
Q1--Q3. Its within-quarter lifts are much smaller than the annual lift. Much of
the aggregate gain comes from causally selecting quarters in which the target's
base rate is high. That is valid under the stated aggregate metric, but it is
not a stable one-to-two-alert experience throughout the year.

A stricter cadence gate was therefore added after this issue was observed:
every calendar quarter with data must contain 0.30--2.50 alerts per
currency-week. **No candidate crossed lift 1.30 under that stronger gate.**

## Statistical uncertainty and search correction

The four-week moving-block bootstrap for the winner gives:

- 2025 lift CI: 0.843--1.951; one-sided `P(lift <= 1)` about 0.050;
- 2026 lift CI: 0.651--1.940; one-sided `P(lift <= 1)` about 0.206;
- 2025--2026 lift CI: 0.968--1.931; one-sided `P(lift <= 1)` about 0.032;
- 2025--2026 future-benefit CI: -22.3 to +79.6 bps.

The circular-shift negative control gives unadjusted p=0.038 on 2025, but
p=0.502 after correcting against the maximum of all 11 frozen adaptation
candidates. For 2025--2026 those values are 0.068 and 0.219. Therefore the
observed lift is promising, but the current sample cannot rule out selection
luck after model search.

## New approaches tried in this round

### Broad new-family packet

Self-normalized path models, fixed target-free ROCKET convolutions, analogue
KNN, direct multi-horizon survival consensus, ordinal/future-floor regression,
domain-balanced models and era-expert consensus were selected on 2017--2020.
The best general-period model was invariant HistGradientBoosting at lift 1.250.
Every family failed the frozen 2022--2023 gate; the best gated lift was 1.198
from the anchor. A retrospective 2024--2026 number above 1.3 was not counted
because the candidate failed the earlier gate.

### Post-2022 reset and recency packet

Quarterly reset, rolling two/three-year windows and exponential recency weights
were tried for HistGradientBoosting, ExtraTrees and logistic regression, plus
rank ensembles. Only the 50/50 reset-Hist/anchor blend crossed the original
2025 gate and remained above 1.3 in 2026. Pure reset Hist at the same policy did
not: 1.074 in 2025 and 1.277 in 2026.

### Same-model percentile packet

To repair score-scale jumps between quarterly refits, raw scores were ranked
against the same model's last 60/120/250 past rows. Some aggregate 2025 lifts
were high (up to 1.485 for a Hist/anchor blend), but all high-lift variants had
at least one zero-alert quarter and failed the cadence gate. Historical rows
also participated in fit, making tree score distributions too optimistic.

### Cross-fitted calibration packet

The preceding one or two quarters were then excluded from model fitting and
used only to calibrate the current model's score distribution. This completely
removes in-sample score calibration. It produced smooth candidates, but the
best cadence-respecting 2025 Hist model achieved only lift 1.002. Cross-fitted
ExtraTrees and ensembles were worse or exceeded the frequency ceiling. This is
strong evidence that simple calibration tricks do not create a stable 1.30
signal.

## Leakage audit

- Physical truncation: all source series were cut at 2020-12-31 and both the
  258-feature base/summary matrix and raw path features were rebuilt. Every
  retained historical row matched the full-data build.
- Label purge: a training row is admitted only when its actual fifth future
  publication date is earlier than the refit date.
- Cross-fit purge: calibration quarters are later than every target reach date
  used in fit.
- Rolling thresholds are shifted by one observation; a current or future score
  cannot change an earlier decision.
- Tomorrow's CBR rate and any `i+1` eligibility flag are absent.

## Decision

Use `quarterly_reset_hist_anchor50` as the leading **research candidate** when
the official score is aggregate lift at average alert frequency: it honestly
crosses 1.30 in both chronological years without tomorrow's rate. Do not call
it production-proven or statistically settled. Freeze it now and collect a
genuinely untouched forward period; also report quarterly cadence next to the
official metric so a scale-shift burst cannot masquerade as a uniformly strong
signal.
