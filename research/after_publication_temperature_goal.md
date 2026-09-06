# Product goal: continuous transfer temperature

Registered 2026-09-06 as a persistent addition to the after-publication CBR
research. This document is a product/evaluation contract, not evidence that the
full intraday widget is already validated.

T19 now supplies the first unified evidence against this contract: 193,400
causal queries across 20 clocks, five currencies and two receipt scenarios.
The route is available everywhere and h5 becomes stably useful during the
market day, but h20 fails every strict paired gate and local currency-year ECE
is not yet production-grade. Availability is implemented; equal quality at
every horizon and regime is not claimed.

T20 tested the first proposed repair and rejected it. A pre-2025 hierarchical
Beta map passed 0/120 state gates on opened 2025--2026 and increased the number
of high-ECE local slices. The frozen probability therefore remains the current
product output; this negative result prevents a cosmetic recalibration from
being mistaken for better forecasting.

T21 then changed discrimination rather than only calibration. Its chosen
HGB+Platt head passed 0/40 h20 gates. A logistic control did improve AUC in all
40 states (mean 0.576 to 0.682), proving useful cross-horizon rank information,
but Brier rose from 0.118 to 0.443. It is therefore not a user-facing
temperature. Any next blend must be fixed on earlier mature data or prospective
outcomes, not tuned on the opened 2025--2026 diagnostic.

T22 tested that small correction. It fails as an all-day replacement, but all
six after-receipt replay states pass the strict AUC/Brier/log-loss/ECE gate:
mean h20 AUC 0.534 to 0.671 and Brier 0.12058 to 0.11379. No-receipt states
worsen. This makes the information event, not a wall-clock threshold, the
candidate boundary. Keep T22 in frozen shadow after actual verified receipts;
retain frozen identity everywhere else until prospective evidence matures.

T23 tested whether monthly delayed recalibration could safely adapt other
states. It passes 0/40 gates and worsens mean h20 AUC/Brier. Mature labels alone
are not enough when a short trailing screen repeatedly selects noisy mappings.
Do not shorten the update interval or relax the gate on the opened period.
The frozen receipt-only T22 shadow remains the next prospective candidate.

T24 supplied the missing pre-receipt discrimination candidate. A 41-feature
history-only logistic ranker reached AUC 0.742 on selection-2024 and 0.702 on
opened 2025--2026 versus 0.374 early identity. Its selection-period Platt map
failed Brier/ECE, so identity correctly remains the user-facing temperature.
Freeze the compact rank separately and test only a predeclared mapping that
preserves the anchor's probability level.

T25 tested that predeclared anchor-preserving family. A 40% residual-logit
blend selected on 2024-H2 improves open AUC 0.374 to 0.563 and Brier 0.12429 to
0.11825, but the 20/50-date Brier intervals cross zero. It remains a premarket
shadow. The divergence between 2025 and 2026 says the missing component is a
slow, causally delayed base-rate level, not more retrospective rank tuning.

T26 tested the first such delayed update and retained T25. A 30-date window
improved 2024-H2 Brier/AUC but failed ECE; all other candidates failed the joint
screen. A 250-date rolling diagnostic is attractive on opened 2025--2026, but
could not be distinguished from expanding before 2025 and is therefore not a
selected temperature. The next experiment must create earlier rolling-origin
probability history or move to prospective shadow.

T27 built that earlier history from 247 quarterly OOS T4/compact publication
dates in 2023. It rejected w250 decisively on 2024-H2, proving the open-period
gain is not a stable preselected rule. w30 remains the only direction that
improves pre-2025 Brier/AUC, but full strength fails ECE; only a preregistered
weak blend is a defensible next calibration experiment.

T28 tested that weak blend with a nested Q3/Q4 decision. Q3 selected 50%, but
Q4 rejected it because AUC dropped 0.03794; even 10% exceeded the rank-loss
allowance. Daily delayed levels are not rank-neutral across dates. Retain T25;
the next defensible update must be frozen for a coarse period or prospective.

T29 froze the update for a month or quarter. Q3 selected a full monthly w30
correction, but disjoint Q4 lost 0.04486 AUC despite better proper scores.
Coarse updates therefore do not solve cross-period rank drift. Retain T25 and
move this calibration question to a genuinely prospective shadow; do not keep
searching weights on the opened history.

T30 rebuilt rank before 2022 and fitted only a probability map on a fixed
post-SVO 2022 window. The 2023 screen rejected every candidate, while the same
all-history family became very strong in 2024--2026 (open AUC 0.750, Brier
0.11827). This is evidence of a regime reversal, not a selectable winner.
Freeze `all_platt_b050` only as a prospective challenger; production remains
unchanged until new outcomes arrive.

T31 tested whether delayed mature feedback can identify that reversal without
a calendar switch. A fixed-share Hedge mixed three frozen T30 experts and used
only h20 outcomes mature before each query minus embargo. No setting passed the
pre-registered 2023 screen, so the selected output remains identity. An
unselected no-share fast learner later reaches open AUC 0.750, Brier 0.11881
and ECE 0.03357, but this is now a frozen prospective control, not a promoted
temperature. The causal mechanism is promising; the current evidence is not a
fresh holdout.

T32 removed T31's cold start with a disjoint Q4-2022 OOS warm-up. It made the
2023 screen worse, not better: maximum candidate AUC fell to 0.552 versus
0.588 identity and 0.604 in T31. Mature Q4 feedback concentrated weight on the
recent2y expert just before the regime changed again. Reject warm-start
promotion; recent outcomes alone are not a stable regime state.

T33 held mixture weights fixed for each quarter. This recovered most of the
rank lost by daily adaptation: 2023 AUC reached 0.607 and proper scores improved
strongly, but the frozen +0.02 AUC gate was missed by 0.00160. Do not relax the
gate after inspection. Freeze `qstack_w125_r100` only as a prospective control;
quarterly stabilization is supported as a mechanism, not promoted as runtime.

T34 repaired the observable cold-start state without another parameter grid.
Before 20 fully mature feedback publication batches exist at a quarter origin,
the system now keeps identity instead of treating equal expert weights as
knowledge. The single candidate passes both 2023 screen (AUC 0.688, Brier
0.18325) and disjoint 2024 validation (AUC 0.620, Brier 0.15645). It also beats
identity on the open period with paired intervals, but not T25 significantly.
Because the repair was proposed after inspecting T33 and later periods are
already open, T34 is a frozen retrospective shadow with
`production_promoted=false`, not a fresh runtime promotion.

## User experience

For every corridor and every requested `as_of` moment, return the latest score
that could really have been computed by then:

- `temperature_0_100`: calibrated attractiveness of transferring now;
- `probability_now_best_h`: probability that the current effective CBR fixing
  is no higher than every fixing in the next `h` publications;
- `expected_future_bps_h`: separately calibrated expected advantage against
  the mean of the next `h` fixings;
- `horizon_publications`: the horizon the user is viewing;
- `score_as_of`, `last_source_at`, `age_minutes`, and a freshness band;
- `phase`: before-new-CBR, after-new-CBR, weekend/holiday, or stale;
- `confidence`: enough mature history / limited history / unavailable;
- `push_now`: a separate sparse binary decision from the frozen push policy.

The widget may say "похожие исторические условия чаще совпадали с удачным
моментом". It must not say "лучше подождать", promise a future course, or turn
the probability into a customer instruction. It must not translate CBR basis
points into rubles saved at the bank until executable customer quotes, fees,
limits and quote validity are present.

## Meaning of temperature

Temperature is not the raw model rank and not the push threshold. For a selected
horizon it starts from a causally calibrated probability. A value near 80 means
that comparable historical model states resolved favourably about 80% of the
time after calibration; it does not mean an 80% guaranteed return. A composite
view may use the fixed geometric mean of calibrated h=3/5/10/20 probabilities,
but the four horizon values must remain inspectable.

Expected future benefit is a second output because two moments with the same
success probability can have different monetary magnitude. It uses the
future-only target, never the symmetric +/-h metric, and stays in CBR basis
points until bank execution data are available.

## Information-time contract

Every feature, peer quote, model, calibration mapping and freshness label must
be a function only of records with `received_at <= as_of`. Model training may
use only labels whose full horizon matured before the fit origin, with the
existing embargo. Corrupting every future feature, target, timestamp and source
must leave the returned historical prefix unchanged.

The current historical CBR replay has calendar-assumed receipt events rather
than certified publication timestamps. Therefore its after-publication widget
is a daily latest-valid-snapshot prototype. T18 now gates production-style
queries on a caller-supplied verified same-day receipt and shifts later dependent
snapshots to that event; only an explicit research flag permits the old calendar
assumption. Between valid updates the score is held constant and becomes
progressively stale; time passing alone must not invent a fresh prediction.

A truly varying intraday temperature requires timestamped information such as
completed MOEX candles or executable bank quotes. It must be replayed at fixed
pre-registered slices or event times and grouped by day in validation. A whole
day's high/low/close/volume cannot be used before that day ends. Missing trading
and weekends produce an explicit stale state, not imputed current prices.

## Model plan

1. Keep frozen AP49/T17 probability as the anchor, T22 after verified receipt
   and T25 before receipt as shadow ranks. T26's delayed selector is rejected;
   T27 rejects w250, T28 rejects a daily weak w30 blend, T29 rejects the
   month/quarter-held correction, T30 rejects the retrospective regime selector
   T31 rejects promotion of a mature-only Hedge on its frozen 2023 screen, and
   T32 rejects disjoint recent-data warm-up, T33 supports quarterly-frozen
   stacking, and T34 proves that missing mature feedback must retain identity
   instead of activating an arbitrary equal mixture. Freeze T25 plus the
   declared T30/T31/T33/T34 controls and wait for prospective
   outcomes before reconsidering level adaptation.
2. Fit a separate robust causal regressor for future-only basis-point benefit
   at each horizon. Report error and calibration by predicted-benefit bins.
3. Implement `score_as_of(currency, timestamp, horizon)` that selects the latest
   admissible snapshot and returns freshness/phase metadata.
4. Add intraday market snapshots only after their timestamps and availability
   conventions are verified. Keep CBR-only and market-enhanced scores separate.
5. Preserve the sparse AP37/AP49-family push router. Push firing must not be
   required for the widget to return a score.

The concrete time-of-day routing design and known coverage gaps are frozen in
`research/transfer_temperature_router_design.md`. In particular, a 15:30 model
may be held after 15:30 as a stale-but-causal anchor, but it is not silently
treated as an updated 17:30 estimate; post-window candles need their own causal
correction and validation.

## Acceptance evidence

- chronological quarterly OOS predictions with mature labels and embargo;
- Brier score and log-loss against a train-only constant prior;
- reliability table/plot and expected calibration error overall, by currency,
  horizon, year and phase;
- discrimination (ROC AUC and average precision) reported separately from
  calibration;
- future-only expected-bps MAE and bin calibration;
- timestamp/staleness unit tests and future-prefix corruption audit;
- coverage for weekdays, weekends, missing data and a user arriving hours after
  the push;
- no claim of real bank savings until quote/fee data pass an execution audit.

Push and widget can therefore improve independently: the push takes only a few
top opportunities to maximize lift, while the widget remains defined and
honestly qualified for every `as_of`.
