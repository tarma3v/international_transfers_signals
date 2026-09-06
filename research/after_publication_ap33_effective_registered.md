# AP33-E frozen protocol: calendar-aware AP23 fallback around AP26 core

Registered 2026-09-06 before AP33 signals or scorecards. AP32 showed that
decision-level routing preserves more of the AP26 specialist than score blending.
AP33 tests exactly one narrower and product-interpretable fallback schedule; no
threshold grid or late-period selector is allowed.

At each currency/date, take the frozen AP26 `y20_shrink200` decision first. If
it does not fire, allow the frozen AP23 `soft730_pace_competence` decision only
when all of the following are true:

1. the meta-policy's own trailing-365 selected rate before the current date is
   below one signal per ISO week;
2. at least 84 calendar days have elapsed since the first eligible date for the
   currency;
3. either no combined signal has fired in the current ISO week and the current
   date is Thursday/Friday, or the currency has had no combined signal in the
   preceding 10 calendar days.

A new sequential max2/currency/ISO-week cap is applied after routing. All state
uses only earlier combined decisions. Source decisions are already causal and
retain mature-only fits, known-down veto, month rescue and their original caps.
Reason1 is AP26 core, reason2 is late-week AP23 fallback and reason3 is AP23
fallback after causal silence. Core keeps priority even when fallback conditions
also hold.

The sole fresh candidate is selected on early 2023 using the unchanged joint
gates. If it fails, the frozen AP32 decision-router is the registered fallback.
Opened 2024-2026 is only a retrospective diagnostic. Audit exact source streams,
trailing rate, silence, week state, reasons, cap, future-source prefix invariance
and paired 20/50-date uncertainty. Acceptance remains min h3/h5/h10/h20 lift
above 2.4, min currency rate at least1, zero empty complete months and max2/week.
H1 is known after receipt and excluded. Historical publication timestamps remain
calendar-assumed and bank execution is not validated.
