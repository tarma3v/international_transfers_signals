# After-publication research - active priority from 2026-09-06

The user explicitly redirected the indefinite research loop to KNOWLEDGE OF THE
ALREADY ANNOUNCED next CBR fixing. Past-only/MOEX work is now control work, not
the principal search. Preserve all old results. Never call unpublished future
values available just because the experiment is labelled after18:00.

## Case interpretation and open questions

Original case-owner Q&A Sep5 p5 allows information available at signal time T
and explicitly notes tomorrow's rate can be published today. This supports
the source's use, not every implementation. Sep5 p7 says application quotes are
real-time and linked to liquidity providers. Keep technical CBR quality separate
from actual savings. See publication_applicability/report-source.md and its
primary-source references. No new blanket prohibition is to be invented.

Two independently labelled target conventions must be studied:

1. **Effective-price reference:** at decision T, start from the rate effective
   on T, with the next h new effective-rate observations as outcomes. At a new
   announcement event the first following observation may already be known.
   Disclose that part of target; compare with the same-information simple rule.
2. **Publication-indexed reference:** start from the latest announced rate, and
   predict the next h not-yet-announced observations. This is a different target,
   not a silent correction to the first. Do not combine their denominators.

Direct original-page recheck on2026-09-06: see after_publication_tz_decision.md.
The page says to use the latest published rate and explicitly allows a public
reference distinct from bank execution. Publication-reference is the conservative
case-facing interpretation; effective-reference is separately labelled. Bank
quote history is not a prerequisite to continue the case's model experiments.

The official scorecard covers h=1/3/5/10/20, signal cadence approximately1-2 per
currency/week, symmetric benefit as stated and future-only benefit separately.
No direction-specific bank execution/price-lock assertion without evidence.

## AP0: availability contract, before scoring a new model

Keep effective_date, received_at, publication evidence type and decision_at
separate. The as-of snapshot receives only rate records known by the cutoff.
An already effective rate and the most recently announced future-effective rate
are separate fields. A weekend without an announcement has no new next-fixing
feature; either the old valid state is used or the frozen policy abstains.

Actual first-seen timestamps take priority. Existing daily CBR archives lack
those timestamps. A sensitivity study may explicitly infer announcement date
as effective_date minus one calendar day and assume an intraday time, but it
must be labelled CALENDAR-ASSUMED, not verified18:00 live. Never expose actual
future values through missing-data flags, calendars, normalization or peer rows.

Required unit checks: a Saturday snapshot cannot see the Tuesday rate whose
receipt is Monday; one microsecond before receipt excludes a record; timezones
are respected; records received after the cutoff cannot change retained values;
duplicate/same-currency asynchronous announcements are deterministic; missing
publication evidence cannot silently become verified; mixed currencies rejected
by a single-currency snapshot.

## AP1: causal baseline and first model packet (freeze before evaluation)

Build new event-indexed and/or explicit daily decision panels from AP0. Do not
merely remove147 bad final alerts from old outputs: training/calibration/ranks/
thresholds and denominators must all use the reconstructed timeline.

First benchmark: no ML. Known-change sign with three/four-calendar-day cooldown;
strictly past rolling ranks of announced change and volatility-normalized
announced change, with predeclared windows and rates. Include no-new-announcement
handling. Build features directly from available prefixes, not a blind i+1 copy
of calendar features whose date is tomorrow. Same-currency and cross-currency
sources independently respect received_at.

Only then test compact global logit, HistGB, ExtraTrees and local/global
residual models. Prefer stable cross-horizon and cross-period gains; no demand
that a complicated model win against a simple known-information rule.

Chronological selection: preregister a bounded grid; select on early folds
(2017-2020 and2022-2023 where suitable). Mature allh20 labels before a boundary
when selecting on all horizons; apply embargo. The already viewed2024-2026
remains retrospective and cannot be sold as a fresh holdout. Model retraining
may use only previously matured labels under a frozen chronological schedule.

Evaluate equal decision dates and target conventions. Report pooled and
currency-year adjusted lift, count/cadence, yearly/currency tables, negative
controls and paired date-block bootstrap versus the simple known-change rule.
Use untouched future prospective data for a new performance claim.

## Active target and persistence

The user explicitly removed the hourly automation and requested an active target.
Continue substantive experiments sequentially without waiting for a schedule;
keep the next stage explicit in after_publication_next_steps.md. Continue until
user stops. Do not
launch duplicate runs or change the Codex agent's model settings without instruction. Existing
push authorization applies only to checked results in ivan-experiments; never
main, force-push, bank transfers or client communications. Update PDF at major
completed stages, not with unverified new scores.
