# T46 frozen protocol: local FX temperature with CNY fallback

Registered on 2026-09-07 before fitting T46 or viewing any T46 scorecard.

## Question

At the 15:30 pre-receipt state, can a simple model use the target currency's
own MOEX/RUB pair when that market is dense and fresh, while falling back
exactly to the existing CNY-based T5 temperature when it is not? This tests the
user-requested per-currency hierarchy. It does not tune or replace sparse
AP37 push and does not use the tomorrow CBR fixing.

## Frozen baseline and targets

- Baseline probability: T5 `cutoff_1530` for each of
  `h = 1, 3, 5, 10, 20`.
- Binary target: the existing future-only send-now event
  `v[t] <= min(v[t+1:t+h+1])`.
- Screen: 2023. Validation: 2024. Open 2025-2026 only for a horizon that passes
  every screen and validation gate. The opened period never changes weights.
- Minimum training date: 2022-02-24. Each quarterly fit uses only same-currency
  labels whose horizon is mature before origin minus a two-calendar-day
  embargo.

## One deliberately compact model

Fit one L2 logistic regression per currency, horizon and quarter on hard-quality
own-pair rows only. Hard quality is frozen from round 7: at least six completed
10-minute candles by 15:30 and the last candle no older than 60 minutes. A fit
requires at least 150 mature same-currency rows; otherwise the quarter uses the
CNY baseline exactly.

Features are fixed before results:

- causal CNY 15:30 raw basis and past-only percentile rank;
- own-pair mean, last and previous basis versus today's CBR;
- own-pair intraday return and high-low range;
- log candle count, candle age, continuous quality, TOM count and TOD count.

Basis inputs are clipped to `[-5000, 5000]` bps, age to `[0, 720]`, counts use
`log1p`, missing values become zero after explicit eligibility gating. Fit is
`StandardScaler + LogisticRegression(C=0.05, L2)` with deterministic seed and
730-day half-life times source-quality weights. No tree model or parameter grid
is allowed.

On a hard-quality query with a fitted local head, blend baseline and local
probability in log-odds using the already observed quality `q`:

`logit(candidate) = (1-q) logit(T5) + q logit(local)`.

On every other row candidate must equal T5 bit-for-bit. This is the literal
fallback to CNY and prevents sparse KGS/UZS/TJS markets from fabricating a
currency-specific update.

## Gates

For each horizon, both 2023 and 2024 must satisfy all conditions:

1. pooled Brier delta below zero;
2. pooled log-loss delta not above zero;
3. pooled AUC delta at least -0.005;
4. Brier delta below zero on rows where a local head is actually active;
5. AMD and KZT each have at least 30 active rows and Brier delta not above zero;
6. 95% paired circular moving-block Brier-delta upper bound not above zero for
   both 20- and 50-date blocks, 2000 deterministic draws.

Passing horizons are prospective shadows only. `production_promoted=false`
regardless of the opened result because these periods have been repeatedly
inspected. Required audits: source hashes, exact fallback, maturity/embargo,
quarterly fit reconstruction, and future-prefix corruption of direct features
and targets.
