# AP2 frozen packet: evening market information beyond the announced fixing

Registered 2026-09-06 before computing AP2 model/target scores. Previous goal
turn was substantive progress: AP1 completed, 127 tests and two PDFs saved,
commit 1aeab71 pushed. AP2 advances actual prediction, not just applicability.

## Information clock and target

Primary convention ONLY publication-reference: start from last announced fixing,
predict next h=1/3/5/10/20 observations. Keep effective-reference AP1 intact.
Decision at 18:30 Moscow on each own-currency announcement event. CBR receipt
remains CALENDAR-ASSUMED effective_date minus one day at 18:00. The extra half
hour is a frozen operational buffer, NOT a claim of guaranteed release by then;
live use must first verify receipt and delay/abstain if missing.

MOEX candles: last-trade end strictly before cutoff AND begin+10min <= cutoff.
Use 10:00 session start consistently, calculate full-session and 15:30-onward
states. Fixed 15:30 boundary is an explanatory fixing-window proxy, not a
timestamp fitted to 2024-2026 returns. Exact CBR methodology changed over years;
the 15:30-to-evening return is a market feature, not a reconstructed official
fixing formula. Rows without fresh market data remain in evaluation and use
CBR fallback; no outcome-based venue selection or date filtering.

CNY main source, quote normalized by verified FACEVALUE=1. Direct currencies
use archived metadata FACEVALUE and SHA manifests. TOM preferred when present,
otherwise TOD; no best-price venue choice. USD excluded from this bounded packet:
official MOEX 2026 innovations show non-deliverable contract regime changes.
No assertion that current USD entries are corrupt. Source https://www.moex.com/s3933.

## Fixed features and scores

Reuse AP1 causal CBR-prefix features. Add CNY and local: last and geometric mean
basis relative to latest announced CBR, mean post15:30 basis, return from last
bar completed by15:30 to evening, intraday range/count/age/quality. CNY reference
is independently cut by assumed receipt, never the old effective rate. Direct
pair quality = min(count/24,1)*exp(-age_minutes/120); no session -> quality0.
Simple scores normalized by CBR historical20-return volatility with1bp floor.

Simple policies: AP1 known_change_z; CNY last basis, post15:30 mean basis, late
return; direct quality-weighted mixtures of last basis with CNY weights75/50/0%;
CBR known_change_z mixed25% with CNY last basis. Missing CNY -> CBR score; missing
direct -> CNY score. Stale20 CNY-last control has its own causal score history.

Model packet, quarterly refits from 2022Q3, expanding market training since2022:
- CBR-only HistGB h5 classifier, and identical HistGB with market features;
- global standardized Logit C=.1, ExtraTrees200/depth8/minleaf30, full features;
- HistGB regressor on mean five-horizon favourable indicators;
- HistGB quantile.25 regression of future h5 minimum log-change;
- per-currency standardized Ridge alpha100 of h5 minimum log-change using own
  announced/effective history, known change, calendar and market features;
- local Ridge plus global HistGB residual, weights.25/.5/1. Residual targets
  MUST use earlier quarterly OOS local predictions, not in-sample fitted anchors.
HistGB fixed160iters, learning_rate.05,15leaves,minleaf40,l2=5, no early stopping.
Seed20260906. Global minimum train400, local60, residual200; cold start uses
only matured training mean (zero if no train) or no residual correction.
All train rows require mature20 < origin-2 calendar days, including h5 fits.

All simple and model scores use strictly prior250 scores/warmup40, rates25/35%,
and separate top35%+3-calendar-day cooldown. Strict ties, same as AP1. Stale20
control top35% only, never eligible for selection. No weekly retrospective max.

## Evaluation and selection frozen before running

Early selection uses ONLY2023, with allh20 matured strictly before2024-01-01.
Choose overall and simple separately: maximize minimum adjusted lift across h
minus2*cadence penalty minus max(-minimum_symmetric_bps/100,0); tie mean lift,
then stable insertion order. Cadence penalty same as AP1 per-currency min/max.
Selection recorded before any2024-2026 scorecard. Opened2024-2026 remains
retrospective, not a fresh holdout; year/currency slices and uncertainty cannot
remove multiple-research bias. Predicting later rows does not require labels.

Compare all candidates on identical own-announcement dates and target. Preserve
full output arrays, log train cutoffs, source hashes and availability summaries.
Compute paired1000 circular20-date-block bootstrap versus AP1 change_z_r25 and
the selected simple; show benefit and communication clustering as well as lift.
Tests: post-cutoff candle corruption, nominal bar end, new/old CBR basis,
missing data retention, OOS residual maturity, causal thresholds/cooldown.
