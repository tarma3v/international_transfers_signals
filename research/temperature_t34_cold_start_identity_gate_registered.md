# T34 registration: identity fallback before mature quarterly feedback

Registered before running T34 or inspecting any T34 metric.

## Question

T33 used equal weights when a quarterly origin had fewer than 20 eligible
feedback publication batches. That is an unsafe cold start: equal weight is a
model choice even though no sufficiently mature outcomes exist to support it.
Does an explicit observable availability gate remove this cold-start damage
without changing the already registered quarterly stack after feedback exists?

## Frozen input and candidate

- Reuse the immutable T33 publication-event predictions and state log.
- Use only the already declared `qstack_w125_r100` quarterly expert mixture.
- At a quarter origin, count earlier publication batches for which the maximum
  contributing h20 maturity is strictly before `origin - 2 days`.
- If that count is below 20, output `identity_early` for every publication in
  the quarter.
- Otherwise output the existing `qstack_w125_r100` value unchanged for the
  whole quarter.
- The threshold 20 is not searched in T34. It is inherited from T33's minimum
  sample rule and means that the stack cannot activate in a state where T33
  itself refused to estimate weights.
- No clock, currency, year, probability threshold, weight, or ridge parameter
  is selected in T34. Push policy and the runtime router do not change.

The only candidate is named `cold_identity_qstack_w125_r100`.

## Chronological protocol

- screen: mature rows in calendar 2023;
- validation: mature rows in calendar 2024;
- open diagnostic: 2025--2026, which has already been inspected and cannot be
  used for selection;
- baseline: `identity_early`; T25 is an additional open-period comparator;
- all grouping and bootstrap resampling remains by publication date.

The candidate passes a stage only if, against identity on that stage:

- AUC delta is at least +0.020;
- Brier delta is below zero;
- log-loss delta is below zero;
- ECE delta is no larger than +0.005.

`historical_protocol_passed=true` requires both screen and validation to pass.
Because the cold-start idea was formed after examining T33 and all 2024--2026
periods are open retrospective data, even a pass creates only a frozen
prospective shadow. `production_promoted` is fixed to false in this packet.

## Required evidence

- exact reconstruction from T33 predictions and quarterly state rows;
- one state per quarter and candidate;
- fallback iff eligible feedback batches are below 20;
- future-target corruption cannot change the gate or any prediction;
- screen, validation, open metrics overall and by currency/year;
- paired date-block bootstrap against identity and T25 on the open diagnostic;
- source hashes and a standalone audit.
