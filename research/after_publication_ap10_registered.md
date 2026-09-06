# AP10 frozen protocol: observed price-path shape and robust evening prices

STATUS: PAUSED BEFORE ANY FIT on 2026-09-06 by the user's clarification of the
reference price. No outcome scorecard from this protocol exists. The next actual
model packet is after_publication_effective_information_registered.md: matched
today-effective targets, without/with the received tomorrow fixing. Preserve
this unexecuted design, do not label it an evaluated negative model result.

Registered 2026-09-06 before fitting or viewing new target scorecards. Prior
turn AP9 was progress: code, all results, 195 tests and checked PDF pushed as
b4d9b5b. Goal remains active; no hourly automation.

## Data discovery changes the originally proposed volume experiment

AP9's source probe checked column names, not values. A complete source-only
scan now shows ALL volume/value fields are null for CNY and all ten direct
pair archives. Three frozen CETS spot checks (183 bars) are also all null.
A fresh public CETS CNY request for 2026-09-02 returns both metadata types
`undefined` and all 55 values null. No VWAP/turnover model can be evaluated
from this archive; null is not zero, candle count is not trading volume.
This is a correction of preliminary availability inference, not a negative
predictive result for real volume. No paid access or account actions.

Instead run the planned price-only alternative on observed OHLC. Keep the
source limitation and exact counts/hashes in an audit artifact. Do not mutate
old source files or historical AP9 results. USD2026 remains excluded.

## Fixed information, target and evaluation

AP8/AP9 panel and 133 original features, same5755 events and all early/later
support. Decision18:30 MSK, completed candles from10:00 with20min delay:
end+20<decision AND begin+10+20<=decision. Same latest ANNOUNCED CBR reference;
all future horizons unknown. CBR receipts remain CALENDAR-ASSUMED, not a
certified18:00 availability guarantee. No bank execution claims.

Fullmature20 before quarterly origin minus2days; same17 origins, training2022+.
Same HistGB160/.05/15leaves/minleaf40/L2=5/noearlystopping, directh5 and coarse
first-cheaper survival. Refit original controls exactly; direct rawX and hazard
AP4 CSVroundtripX remain distinct. No tuning or new regime/clock selection.

## Feature packet and simple rules

Two new matrices: original+CNY shape; original+CNY shape+own pair shape.
Own instrument TOM if any completed bar, otherwise TOD, same old routing.
For each instrument add 18 current-session fields: median-last3 and EW30min
bases relative to announced price, open-to-last return, time slope projected
across observed span, signed path efficiency, realized close-return scale,
positive and zero return fractions, final jump, high-low position, drawdown,
rebound, early/middle/late segment returns (10-12,12-15:30,15:30-cutoff), late
realized-variation share, available-grid density, and path-missing indicator.
Log-return magnitudes divided by the corresponding last20 published-return
standard deviation (floor1bp). No full-session future normalizers. EW uses
elapsed actual candle-end time, not hypothetical complete future intervals.
Density is observed price-bar coverage, NOT liquidity/volume/aggressor flow.
Missing session gives zeros except path_missing=1 and position=.5; missing
simple estimators use old CNY known-change fallback. No overnight price fill.

Three simple CNY estimates (original last, median last3, exponential weighted
log-close half-life30min), each raw or50/50 previous63 CDF mixed with original
directHist or survivalh5:9 policies. Two new feature matrices, each directh5,
survivalh5, survivalmean, CNY50direct, CNY50survival:10 policies. Three original
raw model controls plus2 named AP3/AP4 controls:24 total inclduplicates.
Old urgent-cap2 controller unchanged,40warmup, no retrospective top-k.

Choose only by unchanged2023 joint selector, save selection BEFORE later
scorecard. Later2024-26 already repeatedly studied, NOT a fresh holdout.
All h1/3/5/10/20, adjusted corridor/period baseline, sym/future utility, cadence,
year-currency slices, paired date-block bootstrap20 and selected50. Prespecified
pairs: AP4 versus all; smoothed vs last in same mixture; shapeCNY vs original;
shapeBoth vs shapeCNY for same model/policy. Conditional CIs, not search-adjusted.

## Validation

Source hashes and full missingness scans; independent reconstruction of time
eligibility, segment boundaries, price estimators, normalizers and feature
values; frozen old forecasts/targets/support; training masks/risk counts and
strict mature dates. Tests future bars/days, nominal vs actualend, empty/one
bar/irregular gaps, price rescaling invariance, volume-null notzero, prefixes,
and actual Hist training future/immature labels corruption. Preserve all
negative results, PDF render/visual QA, summary/checkpoint/tests/checkpush only
ivan-experiments. Goal is not complete at packet end.
