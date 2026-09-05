# Round 6: leakage-free lift >= 1.30 at 1--2 alerts/week

Updated: 2026-09-05. Main horizon: five CBR publications (`h=5`).

## Case-definition update from the team Q&A

The official deliverable is a trigger, not a numerical exchange-rate forecast.
For every corridor and signal date, lift is the observed hit rate divided by
the random-day hit rate for the same corridor and calendar period. A hit at
horizon `h` means that the current normalized rate is no worse than each of the
next `h` CBR publications. The organizers expect the result at
`h=1,3,5,10,20`; there is no single official main horizon. Their benefit
metric compares the signal day with the mean over the symmetric `-h..+h`
window. We retain future-only benefit as the stricter supplementary business
view. The 1--2 alerts/week figure is a per-corridor self-check; the actual
communication cap of 1--2 total messages per client is a downstream allocation
problem.

Under that exact five-horizon evaluation, the current strongest stable
past-only score is `geometry75_cba_consensus_basis25`: 75% of the frozen
label-free CNY expert geometry plus 25% of the strictly lagged Armenian central
bank RUB/USD/CNY consensus basis. It uses no next CBR value.

| Combined 2025--2026 horizon | Case lift | Symmetric benefit, bps | Future-only benefit, bps |
|---:|---:|---:|---:|
| 1 | **1.623** | +18.0 | +70.3 |
| 3 | **1.913** | +31.3 | +81.5 |
| 5 | **1.931** | +38.1 | +85.3 |
| 10 | **1.927** | +47.1 | +87.5 |
| 20 | **1.879** | +70.3 | +79.4 |

Mean lift is **1.855**, and the frequency is about **1.26 signals per
corridor-week**. All 15 horizon-by-period lift tests (2025, 2026, combined)
remain above 1.30 after Holm correction; all 15 symmetric-benefit tests remain
positive. The CBA feature itself also passes its predeclared freshness control,
but the incremental lift of the final 25% overlay over its already strong
geometry control is not statistically resolved. This makes it the best frozen
operational candidate by the official scorecard, not a claim of a newly
pristine holdout.

## Executive result

The literal research target is now cleared with a wide margin without the next
CBR rate. The new market-data candidate `logit50_extra50` is a fixed 50/50
causal per-currency rank blend of a 19-feature logistic regression and a CNY
intraday ExtraTrees model. Its alert policy is target rate 22% with a causal
trailing 20-score threshold, fixed before the blend evaluation.

| Period | Future-only lift | Alerts / currency-week | Future benefit, bps | Min currency lift | Min quarter frequency |
|---|---:|---:|---:|---:|---:|
| 2024 policy screen | 1.625 | 1.246 | +69.1 | 1.539 | 0.907 |
| 2025 retrospective confirmation | **1.838** | **1.305** | **+41.7** | **1.696** | 1.138 |
| 2026 retrospective audit | **1.850** | **1.325** | **+117.2** | **1.411** | 1.011 |
| 2025--2026 combined | **1.846** | **1.284** | **+71.3** | **1.538** | 1.011 |

It clears lift 1.30 and 1--2 alerts per currency-week in every observed year
and quarter. The weakest quarter lift is 1.407 and the weakest currency lift is
1.538. Four-week block bootstrap on 2025--2026 gives a 95% lift interval of
[1.562, 2.137] and a benefit interval of [+28.1, +107.6] bps. As with all
continuation research, 2025--2026 has already been inspected and is not a new
pristine holdout; this candidate must be frozen for prospective confirmation.

A new fixed 75/25 blend with per-currency local experts raises the combined
retrospective point estimate to 1.867 at rate 1.265 and benefit +76.5 bps.
However, the paired lift improvement over the primary is only +0.021 with 95%
interval [-0.043, +0.095], and its minimum annual lift is microscopically lower.
It is therefore a frozen challenger, not a replacement selected on noise.

Packet AZ later found an even stronger smooth-additive challenger: 75% primary
plus 25% global all-spline GAM reaches lift 1.867/1.894 in 2025/2026, combined
1.892 at rate 1.221 and benefit +80.4 bps. It clears every operational point
gate, but the paired improvement interval [-0.046, +0.153] crosses zero and the
two-GAM max-adjusted difference p-value is 0.182. The primary therefore remains
unchanged; the higher figure is reported, not promoted.

Packet BB then produced a fully different nonparametric expert: a causal
nearest-neighbour Beta lower-confidence surface over primary/GAM agreement.
Its local/global-shrunk score reaches 1.981/1.807, combined 1.897 at rate 1.157
and benefit +83.7 bps. It remains the strongest standalone reliability score,
but its weaker 2026 and paired CI [-0.089, +0.268] prevent promotion.

Packet BE compresses the last 20 completed CNY sessions as an ordered waveform.
Its ExtraTrees model reaches 1.782/1.827 by year, combined 1.827 at rate 1.216.
The identical waveform delayed by 20 target rows falls to 1.248. A predeclared
paired audit confirms an aligned-minus-stale lift gain of +0.579 with 95% CI
[+0.285, +0.937] and Holm p=0.001. This validates fresh path shape as real
information. A 75/25 primary/wave-logit blend reaches 1.860 combined, but its
increment over primary is only +0.013, CI [-0.086, +0.126], so it is not
promoted.

Packet BG adds fixed random-convolution responses over that path. The standalone
logit is deliberately weak, but its predeclared 25% correction to primary gives
the new highest combined point estimate: 1.990/1.819 by year, combined 1.911 at
rate 1.204 and benefit +81.1 bps. Fresh convolutions beat their stale20 control
by +0.325 lift, paired CI [+0.056, +0.625], Holm p=0.022. The blend's gain over
primary is still uncertain: +0.065, CI [-0.129, +0.298], so it is frozen as a
new prospective challenger rather than promoted.

Packet BI then implements a causal error-regime stack over primary, waveform
and convolution OOF ranks. Its fixed 75/25 logit-stack blend reaches
1.992/1.872 by year, combined 1.941 at rate 1.254 and benefit +81.7 bps. This
is the new highest point estimate, and its +10.4 bps paired benefit interval is
[+0.5, +24.0]. Still, the lift-gain interval [-0.011, +0.249] crosses zero,
Holm p=0.129, and 2026Q2 cadence is 0.98. It is therefore not promoted.

Packet BK transports the path challengers across the full 2017--2026 lifecycle.
The fixed 75/25 primary/convolution blend reaches lift 1.766 at rate 1.175 and
benefit +65.4 bps versus primary lifecycle 1.745/1.201/+61.2. Every annual lift
remains at least 1.522 and every annual rate stays in [1,2]. The paired gain is
only +0.021, CI [-0.050, +0.096], so the full-history primary is also retained.

## What the winning model does

1. Official MOEX `CNYRUB_TOM` history supplies only the last completed trading
   day: for a signal dated `t`, every market feature obeys `TRADEDATE < t`.
2. The transparent component is quarterly refitted logistic regression over
   eight CNY intraday features, five currency indicators, past range positions
   30/90/180 and past target returns 1/5/20.
3. The nonlinear component is quarterly refitted ExtraTrees over the matched
   past-only target/official-macro panel plus the CNY intraday features.
4. Both models use only `h=5` labels whose fifth publication is already known
   before the refit. Their scores are ranked against preceding calibration
   scores per currency and averaged with fixed 50/50 weights.
5. At runtime, today's score is compared with only the preceding 20 scores for
   that currency. The current score enters history only after the decision.

No feature contains the next CBR rate or a same-day MOEX close. The model is not
a forecast of the numeric exchange-rate level: it ranks whether today's
normalized target rate is likely to remain no higher than each of the next five
published rates.

## Other useful passes

| Candidate | 2025 lift / rate | 2026 lift / rate | Combined | Main interpretation |
|---|---:|---:|---:|---|
| CNY logit 50% + CNY ExtraTrees 50% | **1.838 / 1.305** | **1.850 / 1.325** | **1.846 / 1.284** | new strongest retrospective candidate; all quarters pass |
| Shrunk causal reliability LCB | **1.981 / 1.175** | **1.807 / 1.195** | **1.897 / 1.157** | independent uncertainty expert; paired gain unproven |
| Primary 75% + global all-spline GAM 25% | **1.867 / 1.219** | **1.894 / 1.294** | **1.892 / 1.221** | smooth nonlinear challenger; paired gain unproven |
| Primary 75% + resolved-error regime logit 25% | **1.992 / 1.258** | **1.872 / 1.319** | **1.941 / 1.254** | new highest point; benefit gain supported, lift gain not yet |
| Primary 75% + fixed CNY-convolution logit 25% | **1.990 / 1.215** | **1.819 / 1.257** | **1.911 / 1.204** | high complementary point; paired lift gain unproven |
| 20-session CNY waveform ExtraTrees | **1.782 / 1.187** | **1.827 / 1.332** | **1.827 / 1.216** | fresh ordered path beats stale20 by +0.579 lift |
| Raw target+CNY trajectory analogues | 1.803 / 1.341 | 1.554 / 1.307 | 1.677 / 1.298 | interpretable analogue baseline; CNY dominates target path |
| Primary 75% + local-currency consensus 25% | **1.873 / 1.262** | **1.838 / 1.344** | **1.867 / 1.265** | better point estimate and benefit; paired gain is not significant |
| Global logit 50% + hierarchical interaction logit 50% | 1.676 / 1.262 | 1.659 / 1.500 | 1.697 / 1.324 | transparent linear partial pooling; small fallback gain |
| 19-feature CNY + anchor logit | 1.658 / 1.297 | 1.643 / 1.481 | 1.673 / 1.339 | strongest small explainable model |
| CNY intraday ExtraTrees | 1.792 / 1.290 | 1.758 / 1.300 | 1.776 / 1.265 | strongest single nonlinear component |
| `stack50_benefit50` | **1.502 / 1.254** | **1.314 / 1.406** | **1.421 / 1.284** | previous balanced primary without MOEX |
| `stack_resolved_extra` | 1.411 / 1.171 | 1.470 / 1.132 | 1.434 / 1.131 | strongest stable fav-h5 stack; 2025 benefit -2.2 bps |
| resolved weekly router | 1.412 / 1.120 | 1.371 / 1.294 | 1.412 / 1.162 | understandable model-of-models; positive benefit |
| direct benefit ranker + anchor | 1.395 / 1.159 | 1.334 / 1.207 | 1.370 / 1.152 | simplest new business-aligned ML |
| broad75 / CBR25 score | 1.429 / 1.294 | **1.670 / 1.095** | **1.511 / 1.190** | highest headline annual lift; severe quarterly sparsity |
| trusted CBR-only baseload | 1.352 / 1.459 | 1.322 / 1.910 | 1.373 / 1.598 | annual pass with only CBR FX/RUONIA/key-rate information |

All numbers are causal in information flow but retrospective in research
status. The later period had already been inspected in earlier rounds, so none
of these figures is described as a pristine prospective holdout.

## Broad official-CBR data packet

The round added 20 non-target reference currencies from the official Bank of
Russia XML directory and historical-rate endpoint: AUD, BRL, CAD, CHF, CNY,
CZK, DKK, EUR, GBP, HUF, INR, JPY, KRW, NOK, PLN, SEK, SGD, TRY, USD and ZAR.
The five targets AMD/KGS/KZT/TJS/UZS are excluded from panel selection. BYN,
HKD and NZD failed the predeclared 65% USD-row-coverage rule; MXN was absent
from the daily directory.

Every record is divided by its own nominal. The 5.3 MB archived payload has
SHA-256 `3a4ca238ad69200ea8de6a25da77c5336d144c69135ef6ee71208e883a8f7ced`.
The complete endpoint list, row counts, coverage and directory digest are in
`broad_cbr/data_manifest.json`. Source documentation:
<https://www.cbr.ru/development/sxml/>.

The feature packet adds 382 target-free as-of features: individual reference
returns and volatilities, equal-weight RUB factors, USD-leg-removed factors,
breadth/dispersion and target-minus-factor movements. A target row dated `t`
can use only CBR records dated no later than `t`.

## Lagged MOEX CNY/RUB market packet

Packet AE archived the official public MOEX ISS daily history for
`CNYRUB_TOM`, `USD000UTSTOM` and `EUR_RUB__TOM`. The exact payload is stored in
`data/moex_fx_history_2010_2026.json`; its SHA-256 is
`03693535c248cca35976366cd0fc5072ad45779e23d5b7703f3ad34c7e456e8f`.
`CNYRUB_TOM` has 3,384 valid rows from 2013-04-15 through 2026-09-03 and full
post-2022 coverage. Public source: <https://iss.moex.com/iss/history/>.

The as-of join is deliberately conservative: only `TRADEDATE < signal_date`,
never the same trading day's closing price. Zero market-suspension placeholder
rows are missing, stale observations older than seven calendar days are
missing, and no series is backward-filled. Physical corruption of every
same-date and future market row leaves past features bit-identical.

The matched fixed-policy ablation isolates the source of the improvement:

| Matrix | 2025 lift / rate | 2026 lift / rate | Combined lift |
|---|---:|---:|---:|
| no MOEX | 1.248 / 1.467 | 1.330 / 1.692 | 1.310 |
| lagged CNY only | **1.664 / 1.250** | **1.836 / 1.226** | **1.743** |
| lagged USD only | 1.265 / 1.420 | 1.528 / 1.425 | 1.367 |
| lagged EUR only | 1.326 / 1.514 | 1.291 / 1.686 | 1.313 |
| all three | 1.882 / 1.219 | 1.729 / 1.132 | 1.795 |

The cleaner production recommendation uses CNY, not the full three-instrument
panel: USD/EUR availability is discontinuous after trading changes, while CNY
is continuous. The most important CNY fields are the previous trading day's
close-versus-WAP, open-to-close move, one/two-day returns, volatility and trade
activity. Their interpretation is short-term RUB/CNY order flow preceding the
next CBR publication, not knowledge of that publication.

Packet AG supplies a strong negative control. CNY intraday features yield
combined lift 1.776; CNY trend features yield 1.659. Delaying exactly the same
CNY matrix by 20 target rows collapses lift to 1.315, essentially the 1.310
no-MOEX control. Thus feature identity alone does not explain the jump: temporal
alignment carries the information.

## Explainable compression and consensus

A 19-feature logistic regression retains most of the effect: lift 1.658 in
2025 and 1.643 in 2026 at rates 1.297 and 1.481. It uses only the eight lagged
CNY intraday features, five currency indicators, past range positions
30/90/180 and past target returns 1/5/20. Its combined block-bootstrap 95%
interval is [1.405, 1.929], and every currency lift is at least 1.524.

The fixed 50/50 causal-rank consensus with CNY ExtraTrees improves both years
to 1.838 and 1.850. The components' signal-set Jaccard is only 0.432: their
intersection has lift 1.996, while the ensemble keeps a working frequency by
admitting only the strongest residual scores. This is genuine complementarity,
not identical predictions averaged twice.

The outcome is not a knife-edge threshold artefact. Fourteen of 15 neighbouring
policies across target rates 18--30% and trailing windows 20/40/60 clear both
annual lift 1.30 and annual rate 1--2; the only miss has 2025 rate 0.994, not a
lift failure. The original 22%/20 policy remains primary and was not replaced
from this grid.

## Cross-era and 2022 shock audit

The same fixed 50/50 logit/ExtraTrees mechanism was reconstructed with
expanding quarterly refits beginning in 2016 and evaluated on 2017--2021,
always training only on earlier resolved labels since CNY history begins:

| Period | Lift | Alerts / currency-week | Benefit, bps |
|---|---:|---:|---:|
| 2017 | 2.180 | 1.246 | +13.9 |
| 2018 | 1.573 | 1.230 | +62.9 |
| 2019 | 1.995 | 1.061 | +25.8 |
| 2020 | 1.685 | 1.089 | +71.8 |
| 2021 | 1.902 | 1.188 | +48.1 |
| 2017--2021 | **1.845** | **1.147** | **+44.4** |

Combined bootstrap 95% CI is [1.682, 2.009]; every currency is at least 1.749.
This strongly rejects the narrow hypothesis that the CNY mechanism exists only
after 2022.

The transition year still matters. An all-history expanding consensus gives
lift 1.546 in 2022 and 1.550 in 2023. Its weakest shock quarter is 2022Q3 at
1.135, then it rebounds to 1.861 in Q4. A mechanical hard reset first becomes
trainable on 2022-10-01 and initially hurts: 2022 annual lift falls to 1.276.
It recovers to 1.465 in 2023. Therefore old history is valuable during the
transition, while an immediate post-24-February reset is too data-starved.

Once several thousand post-shock labels accumulate, the answer reverses:

| Training memory | 2025 lift / rate | 2026 lift / rate | Combined lift |
|---|---:|---:|---:|
| all history since 2013 | 1.778 / 1.140 | 1.690 / 1.288 | 1.754 |
| old history + post-2022 weight x3 | 1.713 / 1.152 | 1.822 / 1.325 | 1.797 |
| hard reset after 24.02.2022 | **1.838 / 1.305** | **1.850 / 1.325** | **1.846** |
| fixed 50/50 history/reset score | 1.773 / 1.183 | 1.782 / 1.232 | 1.784 |

Thus the practical regime policy is gradual: retain history through the shock,
then prefer a reset model after enough new-regime outcomes exist. The fixed
weight-x3 version is a useful business challenger because its combined minimum
currency lift is 1.670 and its benefit is positive in every later quarter, but
it does not beat hard reset on the primary classification objective.

### One causal lifecycle over all ten years

Packet AU removes the ambiguity of separate era tables. It composes only saved
causal scores and uses a mechanical handoff: keep the expanding model until a
quarterly refit has at least 2,000 resolved post-24.02.2022 target rows, then
use hard reset. This maps to 2024-01-01 without looking at a metric.

| Lifecycle | 2017--2026 lift | Rate | Benefit, bps | Min annual lift | Min annual rate | Min currency lift |
|---|---:|---:|---:|---:|---:|---:|
| always expanding | 1.725 | 1.171 | +60.8 | 1.546 | 1.061 | 1.670 |
| early reset at 700 resolved rows | 1.706 | 1.231 | +59.7 | 1.276 | 1.061 | 1.667 |
| **handoff at 2,000 resolved rows** | **1.745** | **1.201** | **+61.2** | **1.546** | **1.061** | **1.677** |

The selected lifecycle clears lift 1.30 and annual rate 1--2 in every year from
2017 through partial 2026. Its four-week block-bootstrap lift interval is
[1.628, 1.861], benefit interval [+39.9, +80.5] bps, and max-adjusted circular
shift p-value across the three predeclared lifecycles is 0.00025. The weakest
quarter remains the genuine 2022Q3 shock at lift 1.135; no year is hidden or
dropped. The result supports a sample-maturity handoff, not an immediate
calendar/SVO switch.

### Transport of the new path challengers

Packets BK--BL extend the exact convolution schema backward with expanding
quarterly training, then reuse the same 2,000-resolved-row handoff to the saved
post-2022 reset scores. No historical year is selected or dropped.

| 2017--2026 lifecycle | Lift | Rate | Benefit, bps | Min annual lift | Min annual rate | Min currency lift |
|---|---:|---:|---:|---:|---:|---:|
| primary resolved-2,000 control | 1.745 | 1.201 | +61.2 | 1.546 | 1.061 | 1.677 |
| convolution alone | 1.423 | 1.209 | +32.5 | 1.107 | 1.080 | 1.395 |
| **primary 75% + convolution 25% throughout** | **1.766** | **1.175** | **+65.4** | **1.522** | **1.089** | **1.711** |
| primary through 2023, regime blend from 2024 | 1.756 | 1.209 | +62.5 | 1.546 | 1.061 | 1.698 |

The convolution component is not independently stable: it falls to lift 1.107
in 2022. Low-dose blending nevertheless keeps every annual lift above 1.30 and
every annual rate inside [1,2]. Its own lift interval is [1.655, 1.884], but
the paired improvement over primary is only +0.021, CI [-0.050, +0.096], with
max-adjusted p=0.284. The regime handoff changes even less: +0.011, CI
[-0.019, +0.045]. Both also inherit sparse individual quarters; minimum quarter
rate is 0.780/0.859, while the control itself is 0.859. Thus the long-run level
is robust, but neither challenger has proven lifecycle superiority.

### Low-dose anchor insurance during the transition

Packets AV--AY ask whether the pre-existing, purely past-range multiscale
anchor can insure the expanding CNY model specifically while the new regime is
immature. The predeclared 75/25 bridge improves 2022 from 1.546 to 1.648 and
2023 from 1.550 to 1.638. On the full lifecycle it reaches lift 1.770, benefit
+66.1 bps, minimum annual lift 1.573 and minimum currency lift 1.704. The paired
shock-period gain is +0.099, but its CI [-0.025, +0.234] still crosses zero.

A post-diagnostic weight plateau shows the effect is broad rather than unique
to 75/25: four of five neighbouring 60--90% CNY weights improve both shock
years and lifecycle minimum-year lift over pure CNY. The 60/40 cell is the best
descriptive compromise:

| Metric | Pure CNY bridge | CNY 60% + anchor 40% |
|---|---:|---:|
| 2022 lift / rate | 1.546 / 1.400 | **1.756 / 1.273** |
| 2023 lift / rate | 1.550 / 1.242 | **1.622 / 1.360** |
| Minimum shock-quarter lift / rate | 1.135 / 1.068 | **1.302 / 0.992** |
| 2017--2026 lift / rate | 1.745 / 1.201 | **1.784 / 1.200** |
| Minimum currency lift | 1.677 | **1.726** |
| Future benefit | +61.2 bps | **+67.0 bps** |

The 60/40 shock lift gain is +0.151; paired bootstrap CI
[-0.013, +0.365]. Its grid-selection-adjusted circular p-value is 0.056, just
outside the predeclared 0.05 gate. The ten-year circular max-adjusted p-value is
0.043, but the paired lifecycle CI [-0.006, +0.090] still crosses zero. It is a
strong retrospective shock-regime challenger, not statistically established
or prospective superiority. The original frozen post-2026 shadow candidates
are unchanged.

## Rejected CNY/CBR basis hypothesis

Packet AN tested whether the gap between the previous MOEX CNY session and the
last official CBR CNY value supplies an additional fair-value surprise. The
features are strictly nested in time: MOEX trade date `< signal_date`, and CBR
date `<=` that MOEX trade date. Aligned basis ExtraTrees reaches combined lift
1.734, but the identical basis delayed 20 target rows reaches 1.841. Because
the stale negative control is stronger, the basis is interpreted as slow regime
state rather than fresh predictive information. It is retained as a failed
hypothesis and does not replace the intraday primary.

## Independent MOEX context and derived microstructure

Packets AO--AP archived four predeclared official MOEX context series: IMOEX,
RGBI, RUSFAR and GLDRUB_TOM. Every feature uses a record with
`TRADEDATE < signal_date`, at most seven calendar days old. The archived payload
and exact ISS requests are in `data/moex_market_context_2010_2026.json` and
`moex_context/data_manifest.json` (SHA-256
`3764a536a15af9fd076223df3325bd30a4a8c48fb69edd6f8ca1a2b1d5c21bdc`).
MOEX ISS interface documentation: <https://www.moex.com/a2920>.

Each source gave a superficially useful aligned score, but none beat its own
20-row delayed negative control:

| Context | Aligned combined lift | Delayed-20 combined lift | Verdict |
|---|---:|---:|---|
| IMOEX | 1.797 | 1.825 | stale stronger |
| RGBI | 1.795 | 1.801 | stale stronger |
| RUSFAR | 1.789 | 1.820 | stale stronger |
| GLDRUB_TOM | 1.797 | 1.831 | stale stronger |

The all-context model fell to 1.737, and a fixed 50/50 blend with the primary
fell to 1.769. A low-dose 75/25 IMOEX blend produced 1.871/1.839 by year, but
the minimum-year change versus the primary is about one thousandth and the
source failed the stale control. These fields are rejected as fresh predictors.
For production, MOEX data/index usage rights must also be checked separately;
the research archive does not itself grant redistribution or commercial rights.

Packet AQ derived candle pressure, close/WAP location, body, wick asymmetry,
sign agreement and trailing z-scores from the same completed CNY session.
Aligned ExtraTrees reached 1.770, while the block delayed 20 rows reached 1.860.
Thus the attractive stale score is a regime proxy, not evidence that the new
daily geometry adds fresh information. The block is also rejected.

## Local-currency experts and paired audit

Packet AR tests the user's partial-pooling idea directly. At each quarterly
refit, five local logit/ExtraTrees models are trained separately for AMD, KGS,
KZT, TJS and UZS, using only already resolved post-24.02.2022 rows. The local
model sees the same preceding CNY session, but learns corridor-specific
responses. A global model supplies shrinkage so that a small corridor sample
cannot dominate.

| Model | 2025 lift / rate | 2026 lift / rate | Combined lift / rate |
|---|---:|---:|---:|
| local logit | 1.626 / 1.203 | 1.654 / 1.493 | 1.679 / 1.328 |
| local ExtraTrees | 1.755 / 1.266 | 1.622 / 1.388 | 1.702 / 1.300 |
| global 75% + local ExtraTrees 25% | 1.800 / 1.274 | 1.820 / 1.307 | 1.814 / 1.265 |
| local logit/ExtraTrees consensus | 1.822 / 1.317 | 1.772 / 1.363 | 1.803 / 1.304 |
| **primary 75% + local consensus 25%** | **1.873 / 1.262** | **1.838 / 1.344** | **1.867 / 1.265** |

The last row improves combined benefit from +71.3 to +76.5 bps and minimum
currency lift from 1.538 to 1.592. It swaps 29 primary-only signals, whose
target rate is 31.0% and mean benefit -30.1 bps, for 21 challenger-only
signals, whose target rate is 38.1% and benefit +63.3 bps. Signal Jaccard is
0.912, so the change is deliberately small.

The paired four-week bootstrap does not yet validate replacement: lift
difference +0.021, 95% CI [-0.043, +0.095], one-sided
`P(challenger not better)=0.302`; benefit difference +5.2 bps, CI
[-0.8, +11.9]. This is exactly the sort of plausible refinement that should be
frozen and judged only on new dates rather than tuned further on 2025--2026.

Packet AT compresses partial pooling into one auditable linear model. It adds
70 L2-shrunk currency-by-feature interactions to 14 shared numerical effects
and five currency intercepts. The hierarchical logit alone gives lift
1.667/1.657 at rate 1.258/1.468 in 2025/2026. A fixed 50/50 blend with the
global logit improves the explainable fallback to 1.676/1.659, combined 1.697.
However, blending it into the primary reduces 2025 lift to 1.777 and combined
lift to 1.829. Explicit linear shrinkage is therefore useful for explanation,
but it does not reproduce the local nonlinear expert's incremental ordering.

## Smooth additive GAM challenger

Packet AZ fits quadratic quantile splines with five fixed knots and L2 logistic
regression. It uses exactly the same 19 raw past-only inputs as the transparent
logit: eight completed-session CNY fields, six target anchors/returns and five
currency indicators. This makes every feature effect a smooth one-dimensional
curve rather than a tree interaction.

| GAM variant | 2025 lift / rate | 2026 lift / rate | Combined lift / benefit |
|---|---:|---:|---:|
| market-only splines | 1.762 / 1.140 | 1.548 / 1.170 | 1.659 / +68.6 bps |
| all numerical splines | 1.688 / 1.179 | 1.640 / 1.238 | 1.672 / +67.0 bps |
| local per-currency all-spline | 1.673 / 1.179 | 1.630 / 1.257 | 1.662 / +64.3 bps |
| **primary 75% + all-spline GAM 25%** | **1.867 / 1.219** | **1.894 / 1.294** | **1.892 / +80.4 bps** |

The global GAM does not beat ExtraTrees alone, but it contributes different
smooth ordering. The consensus removes 51 primary-only rows with lift 1.318
and mean benefit -13.4 bps, while adding 24 rows with lift 1.711 and benefit
+87.5 bps. Minimum currency lift is 1.639 and minimum quarter frequency 1.005.

Packet BA prevents the attractive point estimate from being oversold. Combined
paired lift gain is +0.046 with CI [-0.046, +0.153]; benefit gain +9.1 bps with
CI [-0.5, +22.1]. The max-adjusted circular difference p-value across the two
predeclared primary/GAM blends is 0.182. The model passes operational gates but
not the predeclared superiority gates, so it remains a research challenger and
does not modify the hashed prospective pair.

## Causal reliability and business-tail surface

Packet BB uses no new market variable and no future label. At each quarterly
refit it searches only already resolved post-2022 examples near today's causal
primary rank, GAM rank and their disagreement. A Jeffreys Beta posterior gives
`mean - 1 sd` hit reliability; forward benefit uses `mean - 1 standard error`.
The local currency estimate is shrunk toward 250 pooled neighbours. A physical
corruption of every still-unresolved outcome leaves earlier scores identical.

| Reliability score | 2025 lift / rate | 2026 lift / rate | Combined lift / benefit | Min currency |
|---|---:|---:|---:|---:|
| pooled hit LCB | 1.940 / 1.191 | 1.764 / 1.213 | 1.856 / +79.1 bps | 1.703 |
| **local/global shrunk hit LCB** | **1.981 / 1.175** | **1.807 / 1.195** | **1.897 / +83.7 bps** | **1.737** |
| shrunk benefit LCB | 1.768 / 1.226 | 1.785 / 1.220 | 1.775 / +73.7 bps | 1.607 |
| hit/benefit equal ranks | 1.873 / 1.167 | 1.849 / 1.207 | 1.867 / +81.7 bps | 1.699 |

The pooled version keeps every quarter at rate >=1.002; the stronger shrunk
version misses only 2025Q1 at 0.987. All quarter lifts stay above 1.52. This is
useful evidence that expert agreement contains a stable local reliability
geometry, and that an uncertainty penalty can outperform direct probability
calibration in the alert tail.

Packet BC nevertheless rejects replacement. Versus primary, shrunk-LCB combined
gain is +0.051 with paired CI [-0.089, +0.268] and max-adjusted p=0.321 across
five BB variants. It improves 2025 strongly but lowers 2026 from 1.850 to 1.807,
so it also fails the non-worse minimum-year gate. The score is retained as a
new independent expert, not blended or retuned on the same years.

## Raw historical-trajectory analogues

Packet BD removes model scores entirely. Every quarter it robust-scales only
resolved training histories and searches for similar raw multiscale target and
completed-session CNY trajectories. Local 80-neighbour hit lower bounds are
shrunk toward 250 pooled neighbours. No learned tree, GAM or primary score is
used in the analogue distance, and corrupting future outcomes leaves earlier
analogue scores identical.

| Analogue space | 2025 lift / rate | 2026 lift / rate | Combined lift / benefit |
|---|---:|---:|---:|
| target path only | 0.832 / 1.325 | 1.250 / 1.493 | 1.063 / +1.2 bps |
| CNY path only | 1.752 / 1.258 | 1.500 / 1.232 | 1.625 / +55.4 bps |
| target + CNY joint | **1.803 / 1.341** | **1.554 / 1.307** | **1.677 / +63.4 bps** |
| primary 75% + joint analogue 25% | 1.852 / 1.238 | 1.794 / 1.325 | 1.835 / +71.9 bps |

The target-only analogue collapses in 2025 and has minimum currency lift below
one. CNY supplies almost all transportable information; adding the target path
helps, but a 25-dimensional Euclidean notion of similarity still loses to the
learned compression in logit/ExtraTrees/GAM. The primary blend also degrades
combined and 2026 lift. This family is retained as an interpretable negative
control rather than promoted or tuned further.

## Completed-session CNY waveform and spectral compression

Packet BE preserves the order of the last 20 completed `CNYRUB_TOM` returns
instead of reducing them to a few horizons or Euclidean neighbours. The fixed
50-field block contains all 20 returns, eight orthonormal DCT-II coefficients,
5/10/20-session mean and volatility, asymmetric volatility, skew,
autocorrelation, sign persistence, path run-up/drawdown and acceleration. Every
input still satisfies `TRADEDATE < signal_date`; corrupting the same-day and
future market payload leaves every past waveform bit-identical.

| Waveform model | 2025 lift / rate | 2026 lift / rate | Combined lift / benefit | Min currency |
|---|---:|---:|---:|---:|
| waveform logit | 1.498 / 1.400 | 1.794 / 1.325 | 1.630 / +58.6 bps | 1.344 |
| **base + waveform ExtraTrees** | **1.782 / 1.187** | **1.827 / 1.332** | **1.827 / +67.1 bps** | **1.567** |
| identical ExtraTrees, waveform stale20 | 1.257 / 1.542 | 1.220 / 1.680 | 1.248 / +24.5 bps | 1.124 |
| primary 75% + waveform logit 25% | 1.864 / 1.230 | 1.831 / 1.319 | 1.860 / +75.8 bps | 1.634 |

Packet BF was frozen before paired differences were calculated. The aligned
ExtraTrees improvement over stale20 is +0.579 lift, paired block-bootstrap CI
[+0.285, +0.937], with a Holm-adjusted one-sided p-value of 0.001 and a fixed
circular-shift p-value of 0.00025. This is the cleanest new evidence since the
original completed-session ablation that the *ordered recent trajectory* is
fresh information rather than a slow regime label.

It does not yet improve the deployment decision. The 75/25 blend is only
+0.013 above primary, CI [-0.086, +0.126], Holm p=0.349. It passes annual lift,
annual cadence, minimum-currency and minimum-quarter-rate gates, but fails both
predeclared superiority gates. The waveform expert is preserved for genuinely
future evaluation; the hashed primary remains unchanged.

## Fixed random-convolution path classifier

Packet BG asks a different question from DCT and trees: whether short ordered
motifs inside the 20-session CNY path add complementary structure. Sixty-four
target-independent kernels were generated once from seed 20260905, with 16
each at lengths 3/5/7/9. Each centered unit-norm kernel contributes only its
maximum response and proportion-positive response. A regularized logistic
model sees those 128 fields, the packet-BE waveform, currency identity and six
past target anchors. This is not the earlier failed target-trajectory ROCKET:
the path here is the previous completed CNY market sessions.

| Convolution candidate | 2025 lift / rate | 2026 lift / rate | Combined lift / benefit | Min currency |
|---|---:|---:|---:|---:|
| convolution logit | 1.216 / 1.550 | 1.222 / 1.437 | 1.209 / +20.4 bps | 1.072 |
| identical logit, CNY block stale20 | 0.963 / 1.550 | 0.804 / 1.524 | 0.884 / -14.9 bps | 0.804 |
| convolution logit 50% + waveform ExtraTrees 50% | 1.670 / 1.341 | 1.600 / 1.338 | 1.635 / +56.6 bps | 1.457 |
| **primary 75% + convolution logit 25%** | **1.990 / 1.215** | **1.819 / 1.257** | **1.911 / +81.1 bps** | **1.624** |

The last blend passes every operational gate and every observed quarter has
lift at least 1.531 and rate at least 1.027. It replaces 95 primary-only rows
(lift 1.729, +27.3 bps) with only 61 challenger-only rows (lift 2.204,
+84.4 bps), which explains the higher point estimate.

Packet BH nevertheless blocks promotion. Aligned convolution versus stale20
has paired gain +0.325, CI [+0.056, +0.625], Holm p=0.022, although its separate
circular-alignment diagnostic is borderline at p=0.058. The blend gain over
primary is +0.065, paired CI [-0.129, +0.298], Holm p=0.219; benefit gain is
+9.8 bps with CI [-5.3, +28.2]. The new score is a high-priority prospective
challenger, not retrospective proof of superiority.

## Resolved-error regime stack

Packet BI operationalizes the proposed "find where each model is wrong" idea
without ever using a future error at decision time. It reconstructs the
own-year OOF scores of primary, waveform ExtraTrees and convolution logit,
converts them to percentiles against only the preceding 250 same-currency
scores, and adds disagreements, currency, six target anchors and the causal
waveform. At every quarterly refit, only 2023-onward rows whose h=5 target has
already resolved may train the stack. A physical future-label flip leaves all
earlier scores bit-identical.

| Regime candidate | 2025 lift / rate | 2026 lift / rate | Combined lift / benefit | Min currency |
|---|---:|---:|---:|---:|
| regime L2 logit | 1.803 / 1.341 | 1.744 / 1.269 | 1.764 / +63.4 bps | 1.468 |
| shallow regime HistGB | 1.848 / 1.270 | 1.699 / 1.238 | 1.770 / +59.8 bps | 1.711 |
| HistGB with auxiliary CNY state stale20 | 1.704 / 1.325 | 1.416 / 1.176 | 1.550 / +54.9 bps | 1.447 |
| **primary 75% + regime logit 25%** | **1.992 / 1.258** | **1.872 / 1.319** | **1.941 / +81.7 bps** | **1.658** |
| primary 75% + regime HistGB 25% | 2.013 / 1.254 | 1.790 / 1.288 | 1.906 / +77.6 bps | 1.628 |

The logit blend removes 64 primary rows with lift 1.634 and benefit +7.1 bps,
then adds 51 rows with lift 2.562 and benefit +99.0 bps. Its minimum observed
quarter lift is 1.600; only 2026Q2 rate, 0.98, narrowly misses the cadence gate.

Packet BJ adjusts across freshness and both predeclared blends. The aligned
HistGB gain over stale20 is +0.220 with paired CI [-0.036, +0.473], so fresh
regime value is suggestive but not formally supported. For the stronger logit
blend, lift gain over primary is +0.094, CI [-0.011, +0.249], Holm p=0.129;
benefit gain is +10.4 bps with CI [+0.5, +24.0]. Thus the business-tail
improvement is supported, but classification superiority and cadence are not.
The fixed score is retained as the highest-priority prospective challenger.

## Low-dose global residual boosting

Packets BM--BN test the user's anchor-plus-global-residual proposal directly.
At each quarterly refit, a small logistic model calibrates the frozen primary
rank on already resolved outcomes. HistGB or ExtraTrees then predicts only
`y - base_probability` from the causal regime matrix, and only 25% of that
residual is added back. Future unresolved labels are physically flipped in the
pipeline; every earlier two-stage score remains bit-identical.

| Residual candidate | 2025 lift / rate | 2026 lift / rate | Combined lift / benefit | Min currency |
|---|---:|---:|---:|---:|
| frozen primary control | 1.838 / 1.305 | 1.850 / 1.325 | 1.846 / +71.3 bps | 1.538 |
| calibrated primary only | 1.782 / 1.337 | 1.823 / 1.244 | 1.787 / +67.4 bps | 1.416 |
| 25% HistGB residual | 1.866 / 1.305 | 1.845 / 1.170 | 1.835 / +69.5 bps | 1.650 |
| identical HistGB residual with CNY state stale20 | 1.847 / 1.290 | 1.659 / 1.257 | 1.750 / +64.6 bps | 1.532 |
| **25% ExtraTrees residual** | **1.864 / 1.345** | **1.864 / 1.276** | **1.853 / +70.4 bps** | **1.573** |

The ExtraTrees point estimate is marginally above primary in both years, but
its paired combined gain is only +0.007, CI [-0.080, +0.124], Holm p=0.698;
benefit is 0.9 bps lower. HistGB aligned-versus-stale gain is +0.084, CI
[-0.019, +0.218], also not supported after multiplicity. Both residual scores
miss the strict minimum-quarter-rate gate. Direct residual boosting is therefore
preserved as a negative result: the richer resolved-error stack captures useful
ranking changes better than adding a generic regression correction.

## Why the boost happened

- The previous completed CNY/RUB trading session provides high-frequency RUB
  state that is absent from the once-daily target series. Close-versus-WAP and
  open-to-close movement are especially informative.
- The small logit supplies a stable low-variance ordering; ExtraTrees captures
  nonlinear interactions. Their partial disagreement is useful, while their
  high-confidence intersection is exceptionally clean.
- The broad FX panel makes the common RUB leg and global-FX residual explicit.
- Direct benefit ranking optimizes the ordering of the actionable tail instead
  of average probability calibration.
- Stacking uses genuinely OOS expert predictions; it can learn when a signal
  family is credible without refitting base models in-sample.
- Resolved-state features carry delayed information only after the exact h=5
  reach date.
- The fixed benefit/stack consensus offsets the stack's 2025 monetary miss
  while retaining its stronger 2026 classification.

## Historical cadence diagnosis before MOEX

Quarterly model refits change the numeric score scale. A standard rolling
threshold can therefore compare a new Q2 model with high Q1 scores from a
different model and nearly stop firing. Resetting threshold history exactly at
the scheduled refit repairs this:

| Policy | 2025 lift / rate / min-quarter-rate | 2026 lift / rate / min-quarter-rate |
|---|---:|---:|
| ordinary rolling-60 winner | 1.502 / 1.254 / 0.764 | 1.314 / 1.406 / 0.296 |
| quarter-reset threshold | **1.289 / 1.668 / 1.400** | **1.446 / 1.493 / 0.933** |
| sparse dual threshold | 1.434 / 1.388 / 0.969 | 1.278 / 1.574 / 0.591 |

The quarter-reset variant missed the annual 2025 lift gate by only 0.011. This
was the state before lagged MOEX features; the new CNY consensus clears both
lift and cadence in every quarter without forced quotas or quarter resets.

## Negative results preserved

- Forced weekly quotas produced smooth rate but reduced lift to about 1.1.
- CBR/broad baseload top-ups filled quiet periods but bought weak rows; the best
  versions fell to 1.18--1.28 in 2026.
- Delayed-feedback Online Hedge stabilized frequency but optimized Brier loss,
  not the alert tail; combined lift stayed near 1.23.
- Activity-aware hand-written gates and weekly caps often removed clustered
  high-confidence primary signals.
- Ordinal/NDCG ranking was strong in one year but did not transfer to 2026.
- Low-dose HistGB/ExtraTrees residual correction was stable annually but added
  no paired lift or benefit over the primary.
- The first packet-P blend output was invalidated because a generic helper
  silently ranked a component against an empty 2023 calibration array. It was
  corrected with strictly expanding 2024 ranks; only corrected files remain.

## Statistical audit

For the new `logit50_extra50` CNY consensus:

| Period | Lift | Four-week block-bootstrap 95% CI | P(lift <= 1) | Circular max-adjusted p within packet AI |
|---|---:|---:|---:|---:|
| 2025--2026 | 1.846 | [1.562, 2.137] | 0.00025 | 0.00025 |

For the previous `stack50_benefit50` baseline:

| Period | Lift | Four-week block-bootstrap 95% CI | P(lift <= 1) | Circular max-adjusted p within packet P |
|---|---:|---:|---:|---:|
| 2025 | 1.502 | [1.187, 1.858] | 0.0005 | 0.0485 |
| 2025--2026 | 1.421 | [1.103, 1.794] | 0.0067 | 0.0227 |

The block bootstrap preserves adjacent weeks and the five currencies of a
date. The circular-shift audit adjusts across the seven recorded packet-P
candidates. It does not correct across every idea tried in all six research
rounds, and the period is retrospective; the statistical result is promising
but must be frozen and replicated prospectively.

## Leakage and reproducibility checks

- Broad-panel future values were physically multiplied after a cutoff; every
  earlier feature remained exactly equal.
- Unresolved future h=5 labels were flipped; earlier Online Hedge/router scores
  and decisions remained equal.
- Future expert scores were corrupted; earlier stack features and refit-aligned
  threshold decisions remained equal.
- Every quarterly training log satisfies `last_resolved < refit_quarter`.
- Target currencies never enter broad-panel selection.
- MOEX same-date/future rows are physically corrupted in a unit test; the
  earlier feature row remains bit-identical.
- The four independent MOEX context histories and the derived CNY
  microstructure block have the same physical future-corruption checks.
- The 20-row stale CNY negative control cannot propagate future values
  backward within a currency.
- The 20-session waveform has its own physical same-day/future corruption test.
- The fixed random-convolution transform has the same physical corruption test.
- The regime stack physically flips unresolved future outcomes and verifies
  that earlier OOF predictions remain bit-identical.
- The lifecycle handoff test verifies that regime scores cannot alter any
  pre-2024 decision.
- Full suite: **103 tests passed** on 2026-09-05. The suite now also verifies
  the byte hashes of every frozen prospective model input, the noon/15:30
  spot archives and their physical cutoff/future corruption invariance.
- NBG and NBRB loaders verify aligned RUB/USD/CNY calendars, normalized units
  and physical future-corruption invariance. The new exponential threshold and
  weekly-cap policies have separate causal state-transition tests.

The complete precommit sequence is in `research/round6_protocol.md`; every
packet was written before its corresponding later-period result was read.
Model outputs, screen grids, breakdowns, bootstraps and multiplicity audits are
stored under this directory and intentionally include failed experiments.

## Next research direction

Keep these frozen prospective candidates:

1. `geometry75_cba_consensus_basis25`, rate 22%, rolling 20 -- strongest
   five-horizon official-scorecard candidate.
2. `logit50_extra50`, rate 22%, rolling 20 -- clean CNY consensus baseline.
3. `primary75_local_consensus25`, rate 22%, rolling 20 -- low-dose local
   challenger; do not tune its 75/25 weight on the existing years.
4. `market_anchor_logit`, rate 22%, rolling 20 -- small explainable fallback.
5. `wave_extra`, rate 22%, rolling 20 -- independent ordered-path expert whose
   fresh-versus-stale value is supported; do not tune the 20-session/DCT schema.
6. `primary75_rocket25`, rate 22%, rolling 20 -- complementary convolution
   challenger; fixed seed/kernel bank and 75/25 weight, pending future evidence.
7. `primary75_regime_logit25`, rate 22%, rolling 20 -- high combined h=5
   point estimate and positive benefit-difference CI; one cadence and both
   lift-superiority gates remain unpassed.

The next packets should freeze a truly prospective time block, test whether
older pre-2022 CNY regimes support the same mechanism, and compress ExtraTrees
into a monotone/additive market-state model. The objective remains annual
future-only lift >=1.30 at 1--2 alerts per currency-week; no further tuning on
2025--2026 can substitute for future prospective evidence.

The prospective block is now formally frozen at the 2026-09-03 historical
cutoff. Its information boundary, immutable settings, logging schema and
non-optional reporting checkpoints are in
`research/round6_prospective_shadow_protocol.md`; byte hashes are in
`prospective_freeze/manifest.json`. The primary and local challenger must run
in shadow on every later date, and historical rows cannot be pooled into their
prospective interval.

## Continued packets T--BN

Research continued after the first report checkpoint. These packets are
protocol-controlled but still retrospective:

| Packet | New idea | Best useful observation | Verdict |
|---|---|---|---|
| T | target-free within-quarter/day score normalization | combined lift 1.319, min-quarter rate 0.954 | uniform cadence bought weak 2026 rows |
| U | learning-to-rank inside currency-week queries | 2024 lift 1.471, 2025 1.261, 2026 0.874 | severe regime overfit |
| V | one shared horizon-conditioned model for five barriers | ExtraTrees alone reached 1.58 in 2026; 50/50 primary blend 1.408 | genuinely complementary, weak in 2025 |
| W/X | low-dose horizon consensus and fixed primary policy | **1.397 / 1.374** at rate **1.23 / 1.42** | strongest minimum-year classification challenger; 2025 benefit -2.6 bps |
| Y | benefit-heavy horizon consensus | combined lift at most 1.360 | did not restore 2025 benefit |
| Z | geometric/harmonic/min expert agreement | geometric 1.345 / 1.312 | annual lift pass, 2025 benefit -8.7 bps |
| AA/AB | tail-only causal meta-labeling and low-dose variants | 50/50 Extra meta: 2025 lift 1.463, benefit +15.5; 2026 lift 1.246 | cadence improved, 2026 precision failed |
| AC/AD | target-currency cross-sectional panel and low-dose blend | 1.405 / 1.316 annual pass | filled 2026Q2 with weak signals; did not replace primary |
| AE/AF | official lagged MOEX FX and matched ablation | CNY-only 1.664 / 1.836 | new information source; strict `TRADEDATE < signal_date` |
| AG | CNY trend/intraday decomposition and stale control | intraday 1.792 / 1.758; stale combined 1.315 | aligned market state, not feature identity, carries gain |
| AH | 19-feature CNY + anchor logit | **1.658 / 1.643** | strong explainable compression |
| AI/AJ | fixed logit/ExtraTrees consensus and policy plateau | **1.838 / 1.850**, rate **1.305 / 1.325** | all quarters pass; 14/15 neighbouring policies pass |
| AK | pre-2022 transport, 2017--2021 | combined **1.845 / rate 1.147** | mechanism survives all five pre-shock years |
| AL | 2022 shock bridge | expanding 1.546 / 1.550 in 2022/2023 | reset is initially data-starved; 2022Q3 is real break |
| AM | full history vs post-2022 weighting/reset | reset **1.838 / 1.850**; weight-x3 1.713 / 1.822 | reset wins after enough post-shock data accumulates |
| AN | MOEX-versus-CBR CNY basis | aligned 1.734, stale20 1.841 | rejected: delayed control is stronger |
| AO/AP | IMOEX, RGBI, RUSFAR, gold context and pairwise stale controls | aligned 1.789--1.797; every stale control 1.801--1.831 | rejected as fresh information |
| AQ | derived CNY candle microstructure | aligned 1.770, stale20 1.860 | rejected: slow regime proxy |
| AR | five local-currency CNY experts with global shrinkage | 75/25 challenger **1.873 / 1.838**, rate **1.262 / 1.344** | useful partial-pooling candidate |
| AS | paired primary/challenger audit | lift gain +0.021, CI [-0.043, +0.095] | freeze challenger; primary not replaced |
| AT | one-stage hierarchical interaction logit | linear blend 1.676 / 1.659; primary blend combined 1.829 | useful explainable fallback, not a primary boost |
| AU | one causal 2017--2026 training-memory lifecycle | **1.745**, min year **1.546**, min annual rate **1.061** | sample-size handoff at 2,000 resolved rows is stable |
| AV/AW | fixed 75/25 CNY/anchor shock bridge and paired audit | lifecycle 1.770; shock gain +0.099, CI [-0.025, +0.234] | useful but not significant |
| AX/AY | seven-weight plateau and multiplicity audit | 60/40 lifecycle **1.784**, min shock quarter **1.302** | broad plateau; adjusted shock p=0.056, keep retrospective |
| AZ/BA | regularized global/local spline GAM and paired audit | primary/GAM **1.867 / 1.894**, combined **1.892** | best point estimate; paired CI crosses zero, no promotion |
| BB/BC | causal neighbour Beta/benefit lower-confidence surfaces | shrunk hit LCB **1.981 / 1.807**, combined **1.897** | strongest combined point; weaker 2026 and paired uncertainty prevent promotion |
| BD | raw target/CNY historical trajectory analogues | joint **1.803 / 1.554**, target-only 0.832 / 1.250 | CNY carries the portable state; raw path distance is too crude |
| BE/BF | ordered 20-session CNY waveform, DCT and paired freshness audit | ExtraTrees **1.782 / 1.827**, combined **1.827**; aligned-stale gain +0.579 | fresh path is supported, but primary blend gain +0.013 is unproven |
| BG/BH | fixed random-convolution motifs and paired audit | primary 75% + convolution 25% **1.990 / 1.819**, combined **1.911** | highest point; aligned beats stale, but primary gain CI crosses zero |
| BI/BJ | causal resolved-error stack over primary/wave/convolution ranks | primary 75% + regime logit 25% **1.992 / 1.872**, combined **1.941** | new highest point and positive benefit CI; lift CI/cadence gates fail |
| BK/BL | full 2017--2026 convolution/regime lifecycle and paired audit | 75/25 convolution lifecycle **1.766**, min year **1.522** | annual gates pass; paired gain +0.021 is unproven |
| BM/BN | global HistGB/ExtraTrees correction of primary residuals | Extra residual **1.864 / 1.864**, combined **1.853** | annual pass, but paired gain +0.007 and benefit -0.9 bps |
| BO--CA | states, shadow nowcast, survival models and resolved routers | several useful independent experts; no reliable promotion | preserved as negative/diagnostic evidence |
| CB--CF | official local-CB cross-rates for TJS and AMD | CBA geometry blend h5 pooled lift **1.947** | CBA is fresh; Armenia overlay becomes official-scorecard leader |
| CG--CJ | exact Q&A metric, five-horizon weights and uncertainty | leader min/mean lift **1.623 / 1.855** | all horizon lifts and symmetric benefits pass Holm |
| CK | separate long-horizon learner | min lift 1.456 standalone, 1.597 blended | weaker than incumbent across the official scorecard |
| CL--CN | official Uzbekistan, Kazakhstan and Kyrgyzstan local-CB archives | each screen retained incumbent | local cross-rates did not transfer robustly |
| CO--CP | resolved multi-horizon routers | screen retained incumbent | routing added variance, not stable discrimination |
| CQ | official Georgian cross-rates | selected raw min lift 1.077; later 0.974 | strong rejection; no blend promoted |
| CR/CS | official Belarus cross-rates and dense blend screen | later 10% point min lift 1.625; frozen 30% choice 1.564 | tiny posthoc point only; screen-selected blend overfit |
| CT | corridor-specific panel across five local central banks | screen panel min lift 1.105 | direct test of per-currency experts failed |
| CU | rolling versus exponential causal thresholds | selected later min/mean 1.594 / 1.819 | higher rate, lower robustness than default |
| CV | pre-2024 calendar prior | later min lift 1.611 | no transferable seasonal boost |
| CW | weekly confidence policy with cap two | later min/mean 1.553 / 1.684 | cadence and future benefit deteriorated |
| CX | post-gap regime modifier | later min/mean 1.605 / 1.866; lifecycle min 1.549 | descriptive gap effect did not survive formal transport |
| CY | seven-bank shadow-RUB consensus | selected later min/mean 1.544 / 1.773 | robust cross-bank dispersion hurts short horizons; no promotion |
| CZ | causal cross-bank uncertainty/veto | selected later min/mean 1.612 / 1.826 | low-dispersion confirmation nearly preserves h1 but trails overall |
| DB | quarterly joint target+CBA+cross-bank ML stack | fresh logit screen min 1.378 versus stale20 1.181 | fresh information exists, but all joint learners trail the incumbent |
| DC | within-source cross-bank revision dynamics | best feasible raw screen min/mean 1.005 / 1.047 | 2024 selection retains incumbent; common short-term revisions are not predictive |
| DD | within-source normalized latent RUB factor | raw screen min/mean 1.233 / 1.339; selected later 1.593 / 1.759 | structural normalization helps the raw factor, but its 10% blend still dilutes the leader |
| DE | one-sided alpha-beta target state | selected later min/mean 1.591 / 1.845 | h10/h20 and symmetric benefits improve, but h1/h3/h5 weaken; no overall promotion |
| DF | nonlinear incumbent/state agreement | selected later min/mean 1.598 / 1.809; stale 1.576 / 1.794 | fresh state adds timing information, but the geometry dilutes future-only precision |
| DG/DH | lagged MOEX CNYRUBF/USDRUBF perpetual futures | selected later min/mean 1.534 / 1.648; stale 1.064 / 1.123 | strong fresh independent expert; incumbent remains stronger |
| DI/DJ | incumbent/futures minimum geometry and paired audit | point min/mean 1.659 / 1.834; min-lift gain CI [-0.183, 0.098] | new Pareto point, but superiority is unresolved; incumbent retained |
| DK/DL | delayed five-horizon online weighting of futures expert | point min/mean 1.643 / 1.845; gain CI [-0.144, 0.081] | causal regime adaptation, but no incumbent promotion |
| DM/DN | official hourly perpetual futures, frozen 12:00 MSK cutoff | noon HistGB later min/mean 1.636 / 1.793 | strong fresh short-horizon state; stale20 min 0.776 |
| DO/DP | label-free incumbent/noon rank consensus and paired audit | point min/mean **1.714 / 1.892**; gain CI [-0.137, 0.202] | new point leader, statistical promotion unresolved |
| DQ | fixed noon/state balance | 2024 retained noon consensus | long-state correction adds no robust scorecard value |
| DR | shared official-horizon noon learner | best new screen min 1.543; noon consensus 1.566 | direct multi-horizon pooling does not improve selection |
| DS/DT | official hourly MOEX spot and quarterly ML | best spot/perpetual screen min 1.528 | full spot ML trails the simple consensus |
| DU | signed noon partial-fixing basis | screen 1.644; later min 1.604; stale 0.844 | fresh but regime-dependent standalone signal |
| DV/DW | delayed online noon/spot weighting and paired audit | later min/mean 1.714 / **1.898**; mean-gain CI [-0.045, 0.076] | point mean improves, statistical promotion fails |
| DX | label-free noon/spot agreement geometry | 2024 retained signed spot | agreement formulas do not fix transport |
| DY/DZ | methodology-aligned 15:30 spot nowcast | later min/mean **1.708 / 1.883**; h20 1.927 | strong explainable timed challenger, not overall leader |
| EA | label-free 12:00/15:30 consensus | 2024 retained raw fixing basis | strong components do not improve through static agreement geometry |
| EB/EC | candle-level fixing proxies and paired audit | fresh-vs-stale min gain **+0.926**, CI **[+0.638, +1.167]** | freshness proved; no lift superiority over noon; true VWAP unavailable |
| ED/EE | market-availability fixing/noon router and paired audit | later min/mean **1.780 / 1.956**; h5 2.096/1.994 at rate 1.19/1.28 | new point leader; paired gain unresolved and one quarter rate 0.955 |
| EF | cadence-first router policy plateau | selected rolling-20%/20 reaches min 1.785 but quarter rate 0.875 | default 22%/20 retained; 2024 cadence does not transport perfectly |
| EG/EH | fixed 2022--2026 fixing lifecycle and SVO-boundary audit | yearly min lift 1.514--1.740; combined min/mean **1.601 / 1.751** | strongest explainable stability evidence; no selector or next CBR fixing |
| EI | fixed pre-publication cutoff frontier | 15:20 ties 15:30 on screen; later raw min/mean 1.712/1.874 | last ten minutes add little; earlier cutoffs weaken more |
| EJ/EK | 15:20 availability router and non-inferiority audit | later min/mean **1.770 / 1.952**; aggregate CIs pass -0.05 margin | h20 CI reaches -0.056, so strict all-horizon non-inferiority fails narrowly |
| EL | global HistGB/ExtraTrees residual correction over fixing anchor | best new screen min 1.558; router 1.607 | user-proposed anchor+global-residual pattern tested directly; no promotion |
| EM | outcome-free fixing-basis regime normalization | screen 1.623 vs 1.607; later min/mean 1.775/1.934 vs 1.780/1.956 | small cadence repair costs precision; no promotion |
| EN | reference-invariant intraday fixing-shape proxies | screen 1.628 vs 1.607; later min/mean 1.761/1.930 | session-persistence filter does not transport; no promotion |

Packets CY--DB were a direct test of whether the official local-central-bank
archives could add a broad, explainable second view beyond Armenia. All local
observations are strictly older than the signal date, domestic units cancel in
the cross-rates, and physical future-corruption tests pass. The seven-bank
median/dispersion family improved the 2024 screen but reversed on the sealed
2025--2026 comparison. A causal low-dispersion confirmation reduced that loss
but still finished below the incumbent at the aggregate objective.

The joint ML layer was refit quarterly from post-24-February-2022 data and
admitted a training row only after its h5 label was fully resolved. Its 32
features combine the primary and geometry ranks, target ranges/returns,
currency identity, 13 Armenian CBA measurements and six cross-bank summaries.
Aligned logit and HistGB features beat their 20-row-stale twins on the 2024
screen, which supports genuine timing information. Nevertheless, the frozen
five-horizon selector retained the incumbent, so no 2025--2026 result was used
to retune or promote this family.

Packet DC additionally removed changing source membership as a possible
confounder by differencing every bank before aggregation. This made the
cross-bank revision family clearly weak on the 2024 screen itself, so the
sealed later selector was the incumbent exactly; no later number was needed
to reject the feature family.

Packet DD addressed a second comparability problem: every bank's level was
converted to a causal percentile and robust z-score against only its own prior
250 signal dates before aggregation. That materially improved the standalone
factor but did not create a transferable addition. Its selected 10% blend
trails the incumbent at every aggregate objective on 2025--2026.

Packet DE moved away from external cross-rates and estimated a causal local
level/slope state for each target series. The 2024-selected 10% correction is
a real horizon trade-off rather than a general failure: on 2025--2026 it raises
h10 from 1.927 to 1.939 and h20 from 1.879 to 1.914, and improves symmetric
benefit at all five horizons. It simultaneously lowers h1 from 1.623 to 1.591,
so the single-trigger official scorecard still favours the incumbent.

Packet DF tested whether state agreement could preserve the incumbent's short
horizon while retaining DE's long-horizon gain. The selected high-state bonus
does beat its 20-row-stale control, but not the incumbent. This separates two
claims: the state is temporally informative, yet it is not incrementally useful
for the single trigger under the all-horizon objective.

Packets DG/DH add the first genuinely new market archive since the CNY spot
packet: 1,107 official MOEX sessions each for perpetual CNY/RUB and USD/RUB
futures from 26 April 2022 through the frozen cutoff. Only the previous trading
day is visible. A quarterly ExtraTrees model selected on 2024 has h5 lift
1.770 in 2025 and 1.645 in 2026 at rates 1.18 and 1.26; its minimum currency
lift is 1.653. Across all five horizons its combined minimum/mean is
1.534/1.648. Delaying the 47 futures fields by 20 rows collapses the same model
to 1.064/1.123, while the fresh h5 bootstrap lower bound remains 1.414. This is
strong evidence of fresh information, though still below the incumbent.

Packet DI applies a conservative agreement rule: rank a row highly only when
both incumbent and fresh futures ExtraTrees do. It raises the point minimum
over h=1/3/5/10/20 from 1.623 to 1.659 and improves h1/h3, while lowering the
mean from 1.855 to 1.834. Its h5 annual lift is 1.959/1.846 at rate 1.21/1.34
and minimum-currency lift 1.795. Packet DJ keeps the same block draws across
horizons: the paired minimum-lift gain is +0.036 with CI [-0.183, +0.098], so
the robust-minimum promotion gate fails. In contrast, fresh-versus-stale lift
differences are positive after Holm at all five horizons. The result is a
frozen challenger, not a replacement for the CBA geometry incumbent.

Packet DK replaces fixed geometry with an auditable online weight. Every
resolved `(row, horizon)` event from h=1/3/5/10/20 contributes Brier loss only
after its own reach date; the best 2024 online family member is global,
250-event memory, eta 5. On 2025--2026 it reaches minimum/mean official lift
1.643/1.845 and h5 lift 2.027/1.843 at rates 1.25/1.25. The incumbent weight
varies causally from about 0.57 in 2024Q1 to 0.39 in 2026Q1. Packet DL finds a
point minimum gain of +0.020 but paired CI [-0.144, +0.081], so it is not a
replacement. Its fresh-versus-stale minimum gain is +0.262 with CI
[+0.111, +0.593], and fresh lift beats stale after Holm at all five horizons.

Packets DM/DN implement the timing requirement from the 05.09 case-owner Q&A.
The frozen public archive contains 16,951 hourly CNYRUBF and 16,953 USDRUBF
candles. A row at date T sees only candles ending before T 12:00 MSK; physical
corruption of the noon candle and all future candles leaves earlier features
unchanged. The 2024-selected `noon_hist` reaches h5 lift 1.830/1.942 at rates
1.30/1.34 in 2025/2026, while its stale20 twin falls below lift one combined.

Packet DO averages the causal percentile ranks of that noon score and the
incumbent. Although a three-view family including the previous-session futures
expert was screened, 2024 selected the simpler two-view arithmetic mean. On
2025--2026 its official lifts are 1.714/1.954/1.961/2.010/1.819, so its minimum
and mean are 1.714/1.892. Annual h5 lift is 2.062/1.873 at rate 1.25/1.47; all
quarters exceed rate 1.09 and combined minimum-currency h5 lift is 1.748.
Packet DP gives paired minimum-lift gain +0.091 with CI [-0.137, +0.202], so
the new point leader remains a challenger. Its matched double-stale minimum is
1.331 and fresh-versus-stale mean-lift gain is +0.430, CI [+0.209, +0.595].

Packets DQ/DR then tested two natural ways to repair the point leader's weaker
h20: a fixed target state-space correction and one shared classifier trained
directly on the five official horizons. Both 2024 screens retained the existing
noon consensus. This is a useful negative result: the long-horizon shortfall is
not repaired by adding a slow target state or by pooling more labels into the
same feature representation.

Packets DS/DT add official hourly `CNYRUB_TOM` and `USD000UTSTOM` candles at the
same noon cutoff. The frozen archive has 13,427 and 8,719 rows respectively;
36 derived features pass physical cutoff/future corruption. Quarterly logit,
HistGB and ExtraTrees variants, including spot+perpetual views, all lose the
2024 selection to the existing noon consensus. The raw economic sign is more
useful: packet DU's maximum spot-to-current-CBR basis reaches screen minimum
1.644, and its stale20 twin collapses below one. But it transports to only
1.604 minimum lift on 2025--2026, so it is a fresh regime-sensitive expert,
not a primary model.

Packet DV applies the previously audited delayed online rule to the noon and
signed-spot experts. Every horizon contributes Brier loss only after its reach
date. The point minimum is unchanged at 1.714, mean lift rises from 1.892 to
1.898, h20 rises from 1.819 to 1.874, and future-only benefit improves at all
five horizons. Packet DW does not confirm the gain: paired mean-lift CI is
[-0.045, +0.076], and no individual lift difference survives Holm. Keep this
as a research point challenger, not a promoted score.

Packets DY/DZ define a separate decision product at 15:30 Moscow, aligned with
the Bank of Russia CNY fixing window rather than tuned on target metrics. The
10-minute archive contains 74,442 CNY and 46,547 USD candles; the next official
CBR rate is never loaded. A label-free positive CNY session-mean basis selected
on 2024 reaches later official lifts 1.708/1.889/1.943/1.948/1.927 for
h=1/3/5/10/20, minimum/mean 1.708/1.883. Annual h5 lift/rate is 1.983/1.38 in
2025 and 1.901/1.34 in 2026; its stale20 minimum is 0.782. It is slightly below
the 12:00 consensus overall but stronger at h20 and highly explainable.

Packets EA--EC close the static fixing-proxy branch. Five predeclared
label-free geometries of the noon consensus and the 15:30 basis all lost the
2024 screen to the raw fixing basis. A more literal transaction-weighted CBR
fixing reconstruction could not be built because every archived MOEX candle
has null `value` and `volume`; no volume was imputed. Five deterministic OHLC
proxies are operationally indistinguishable under the frozen rank policy.
During implementation an apparent min/mean improvement to 1.790/2.040 was
traced to NaN market-closed days being removed from the evaluation denominator
and was rejected. The corrected implementation keeps those dates eligible with
the frozen neutral score and asserts exact equality of mean-close to packet DZ.

Packet EC confirms that the corrected 15:30 basis contains fresh information:
versus stale-20, draw-wise minimum-lift gain is +0.926 with 95% CI
[+0.638, +1.167], and all five lift/benefit differences survive Holm. It does
not replace the noon point leader: minimum difference is -0.006 with CI
[-0.106, +0.178], and mean difference is -0.008 with CI [-0.101, +0.107].
Its symmetric benefit is significantly higher at h=1/3/5/10, so 15:30 remains
an explainable late-decision product rather than a globally superior score.

Packets ED/EE replace the invalid idea of dropping market-closed dates with a
causal fallback: use the 15:30 fixing rank when at least one completed CNY
candle exists, otherwise use the already available noon-consensus rank. Every
target row remains in the denominator. The route wins the frozen 2024 screen
and transfers to official lifts 1.780/2.014/2.053/2.005/1.928, hence
minimum/mean 1.780/1.956. Annual h5 lift is 2.096/1.994 at rate 1.19/1.28;
minimum-currency lift is 1.832 and all future-only benefits are positive.

This is the strongest point score so far, but not a statistical promotion.
Against noon, paired minimum gain is +0.066 with CI [-0.123, +0.238] and mean
gain is +0.065 with CI [-0.061, +0.201]. The 2025Q2 rate is 0.955, narrowly
missing the stricter internal quarter-rate gate, although the requested annual
average 1--2 band passes in both years. Against matched stale fixing, minimum
gain is +1.025 with CI [+0.674, +1.311], so the timing information itself is
strongly supported. Keep the router frozen as a retrospective 15:30
challenger and require new prospective dates before claiming superiority.

Packet EF tests whether a small causal policy plateau can repair the one weak
quarter without touching the score. The 2024 cadence-first screen selects
rolling 20% with memory 20, which transfers to min/mean 1.785/1.952 but lowers
2025Q2 frequency from 0.955 to 0.875. The default 22%/20 policy therefore stays
operationally preferable. No later-period threshold repair is attempted.

Packet EG removes model selection entirely and applies the packet-EB fixing
basis with frozen 22%/20 policy to every available year. The five yearly
minimum lifts for h=1/3/5/10/20 are 1.531, 1.514, 1.578, 1.740 and 1.665 for
2022--2026. Every annual h5 frequency lies between 1.19 and 1.38. Across the
full period, five-horizon minimum/mean is 1.601/1.751, h5 lift is 1.751 with
moving-block 95% CI [1.612, 1.926], minimum-currency h5 lift is 1.649 and the
circular-shift p-value is 0.00025. The first archive date is explicit; earlier
rows are unavailable, while later market-closed days remain neutral eligible
rows rather than disappearing from the denominator.

Packet EH reuses those decisions and splits 2022 mechanically at 24 February.
The pre-boundary block has minimum/mean lift 1.471/1.691 and the post-boundary
block 1.533/1.635, at frequencies 1.47 and 1.27. Only 45 signals occur before
the boundary, so this is descriptive rather than a model-selection claim. It
does show that the signed fixing mechanism is present on both sides of the
shock instead of being created only by the post-2022 regime.

Packet EI measures the price of waiting within the official fixing window.
Eight business cutoffs are frozen before evaluation and every one passes a
physical future-candle corruption test. The 2024 selector chooses 15:20, tied
with 15:30 at minimum/mean 1.578/1.673. On 2025--2026 the raw 15:20 product
gives min/mean 1.712/1.874, compared with 1.708/1.883 at 15:30. The last ten
minutes therefore have little scorecard value, while 10:30--11:30 are visibly
weaker on the screen.

Packets EJ/EK apply the same causal market-availability fallback at 15:20.
The earlier route reaches lifts 1.770/2.006/2.049/2.008/1.928 and hence
min/mean 1.770/1.952; all annual rate, currency, quarter-0.95 and benefit gates
pass. Against the 15:30 route, the paired aggregate minimum difference CI is
[-0.042, +0.025] and mean CI [-0.025, +0.014], both inside the predeclared
-0.05 lift margin. Four of five horizon lift intervals also pass; h20 extends
to -0.056, missing the margin by 0.006. All symmetric and future-benefit
intervals pass their -5 bps margins. Thus 15:20 is a strong ten-minute-earlier
challenger, but strict all-horizon non-inferiority is not claimed.

Packet EL directly implements the proposed simple-anchor plus global-residual
architecture. A quarterly logistic calibrates the fixing rank by currency,
then fixed HistGB and ExtraTrees regressors learn h5 residuals from 36 aligned
15:30 fields, six causal target fields and currency identity. Every training
label has resolved before its refit; flipping unresolved future outcomes leaves
all earlier scores unchanged. The best new model, residual ExtraTrees, reaches
only 1.558 worst-horizon lift on the 2024 screen, and all fixed 75/25 router
mixtures are at or below 1.536 versus 1.607 for the unchanged router. The
screen therefore retains the transparent availability rule without consulting
2025--2026 for a model choice. This is evidence that the strong basis is not
improved merely by adding nonlinear capacity.

Packet EM asks whether the fixing basis needs an adaptive structural baseline.
Seven outcome-free transformations use only the current and strictly earlier
available sessions: innovations, trailing-median excesses, robust z-scores and
two persistence summaries. The 2024 selector chooses a fixed 75/25 mixture of
the router and trailing-60 robust z-score, raising the screen minimum lift from
1.607 to 1.623. The gain reverses on 2025--2026: minimum/mean lift is
1.775/1.934 versus 1.780/1.956 for the unchanged router, while h5 is 2.038
versus 2.059. Minimum-quarter cadence improves only from 0.955 to 0.986. Thus
slow structural normalization slightly shifts alerts but does not add durable
precision; the raw market-to-current-CBR gap remains the better explanation.

Packet EN follows the published CBR fixing mechanism more closely without
pretending that public candles contain unavailable historical trade volumes.
Seven scale- and reference-invariant corrections ask whether the mean basis is
supported by the median, lower tail, all three intraday blocks or a low-noise
session. The 2024 selector chooses a fixed 75/25 router/block-minimum mixture,
raising minimum lift from 1.607 to 1.628. On 2025--2026 it falls back to
1.761/1.930 minimum/mean versus 1.780/1.956 for the raw router; h5 is 2.043
versus 2.059 and minimum-quarter frequency worsens from 0.955 to 0.923. The
path-shape confirmation therefore removes some useful high-basis days rather
than isolating a more stable subset.

The new classification challenger is
`primary75_shared_extra_geomean` under the unchanged 22%/rolling-60 policy:

| Period | Lift | Alerts / currency-week | Future benefit, bps | Min quarter frequency |
|---|---:|---:|---:|---:|
| 2025 | **1.397** | **1.234** | -2.6 | 0.800 |
| 2026 | **1.374** | **1.425** | +78.2 | 0.156 |

That shared-horizon challenger raised the pre-MOEX minimum annual lift from
1.314 to 1.374 but did not replace `stack50_benefit50`. The later MOEX packet
then supplied genuinely new preceding market information and changed the
picture: `logit50_extra50` now replaces both as the strongest retrospective
candidate, including in the formerly quiet 2026Q2 (lift 2.029 at rate 1.011).
