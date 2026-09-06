# M1 frozen metric-sensitivity protocol: future-only versus symmetric `+/-h` lift

Registered on 2026-09-07 before calculating any symmetric-local-minimum lift.
This is a metric audit of the frozen AP37-E policy, not a new model search.

## Why this audit exists

The case text mentions both a local minimum in a surrounding `+/-h` window and
a `send now` hit for which the rate stays no worse during the following `h`
observations. The case-owner Q&A maps lift to the latter future-only hit and
maps `+/-h` to the separate moment-benefit diagnostic. Because a team can still
read the first sentence as a binary target, this packet reports both readings
on exactly the same frozen signals and evaluation rows.

## Frozen inputs

- Policy: AP37-E
  `ap26_core_mature_precision_calendar_fallback_cap2`.
- Signals, eligibility, early/later scopes, groups and dates come byte-for-byte
  from `results/research/after_publication/ap37_effective/outputs.npz`.
- Reference convention remains AP37's today-effective CBR rate. No policy,
  feature, threshold, cadence rule, model or evaluation date may change.
- Headline period is the already opened AP37 later scope, 2024--2026. The 2023
  early scope is reported only as a consistency table.

## Two binary labels

For currency series value `v[i]` and publication horizon `h`:

1. Existing future-only hit:
   `y_future(i,h) = 1[v[i] <= min(v[i+1:i+h+1])]`.
2. Alternative symmetric local-minimum hit:
   `y_symmetric(i,h) = 1[v[i] <= min(v[i-h:i+h+1])]`.

Both require the complete relevant window. Equality counts as success. The
symmetric label is rebuilt from the raw CBR series and AP37 `current_index`;
its future edge receives the same publication-reference completeness mask as
the existing AP37 outcome. Horizons are `1, 3, 5, 10, 20`.

## Scorecard and uncertainty

For each label and horizon report signal count, hit rate, random-day base rate,
pooled lift, AP37 currency-year adjusted lift, and unchanged frequency. Also
report currency and year slices. A random day always uses the same valid rows,
period and corridor weighting as the corresponding signal score.

For the later scope compute 95% circular moving-block intervals for adjusted
lift with 500 deterministic draws for block lengths 20 and 50 unique dates.
This is descriptive sensitivity only: no result can promote, reject or retune
AP37. The audit must prove that all input signals/scopes are unchanged and that
the saved future-only point values reproduce AP37's scorecard.

## Interpretation rule

The report will not relabel AP37's existing future-only lift as `+/-h` lift.
If the symmetric lift differs, both values remain side by side. Official
symmetric moment benefit in basis points and future-only benefit in basis
points remain separate magnitude metrics and are not replaced by this binary
sensitivity.
