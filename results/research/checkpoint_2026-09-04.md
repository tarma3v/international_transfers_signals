# Research checkpoint — 2026-09-04

This file freezes the results known before the second deep-research round. It is
intended to prevent later experiments from being confused with genuinely unseen
validation.

## Scope and data

- Target corridors: TJS, UZS, KGS, AMD, KZT.
- Full CBR history available to the study: 2010-01-01 through 2026-09-02.
- Main target: `fav_h5`, meaning that today's normalized CBR rate is no higher
  than every rate in the next five CBR publications.
- Main metric: future-only lift. Product frequency target: 1--2 alerts per
  corridor per week.
- Development: through 2016; general validation: 2017--2020; calibration: 2021;
  predeclared shock/adaptation selection: 2022--2023; final audit: 2024--2026.

## Known results before round two

### Timestamp-aware policy

This policy is valid only after the next effective CBR rate has actually been
published. It fires when today's rate is no higher than the already published
next effective rate, with a three-calendar-day cooldown.

- Final period: 2024--2026.
- h=5 lift: 1.9585776475.
- Hit rate / base rate: 0.5767590618 / 0.2944785276.
- Frequency: 1.369343 alerts per corridor per week.
- Future-only benefit: +77.843965 bps.
- 95% four-week block-bootstrap lift interval: [1.7653117552, 2.1735784339].

This is not an ordinary h=1 forecast: the first future effective rate is public
at decision time. If the alert is sent before publication, using it is leakage.

### Strict past-only policies

The anchor-only family was selected on 2022--2023 and then applied unchanged to
2024--2026. Its h=5 winner was:

`0.5 * pct_range_90 + 0.3 * pct_range_30 + 0.2 * pct_range_180`

with a rolling 250-publication threshold targeting the top 20% of scores.

- h=5 lift: 1.2952678571.
- Hit rate / base rate: 0.3814285714 / 0.2944785276.
- Frequency: 1.021898 alerts per corridor per week.
- Future-only benefit: +34.962780 bps.
- Weakest year lift: 0.5078647065.
- Weakest corridor lift: 1.0767604703.

The broader model library selected on 2022--2023 a 75% `anchor_pct90` + 25%
five-year ExtraTrees ensemble. Applied unchanged to 2024--2026 it produced:

- h=5 lift: 1.2600585005.
- Frequency: 1.018978 alerts per corridor per week.
- Future-only benefit: +31.656740 bps.
- Weakest year lift: 0.8234489051.

The past-only diagnostic formula
`pct_range_90 + 0.035 * ret_20 + 0.015 * ret_60` produced h=5 lift 1.4056721195,
frequency 1.026277 and +47.440288 bps on 2024--2026, but it was identified after
inspection of that period. It is a hypothesis for future confirmation, not a
locked holdout result.

## Interpretation frozen for the next round

- Defensible ordinary past-only headline: approximately 1.26--1.30 lift at h=5.
- The 1.96 result is a separate post-publication product policy and must never be
  compared as though it were a conventional pre-publication forecast.
- The 2024--2026 observations are no longer unseen for any newly invented policy.
- A second research round must use nested/rolling selection inside earlier time
  blocks, report multiplicity, year/corridor stability and block uncertainty, and
  label any 2024--2026 improvement as retrospective unless its exact policy was
  already frozen above.

