# T32 preregistration: out-of-sample warm start for mature-only Hedge

Registered before computing T32 outputs. T31 started its online ensemble on
2023-01-01 with uniform weights and no past feedback. T32 tests whether a
strictly earlier, disjoint warm-up stream can remove that artificial cold start
without using any 2023--2026 outcome for initialization.

## Frozen chronology

1. Rank models are exactly the T30 `all` and `recent2y` logistic families,
   trained only on h20 labels mature before the 2022 origin minus two-day
   embargo.
2. Positive Platt maps are refit only on rows from 2022-04-01 up to 2022-08-01
   whose h20 labels mature before the warm-up origin minus embargo.
3. T4 has no saved OOS anchor before 2023. For the warm-up only, reproduce its
   frozen feature set and HGB specification in one fit at 2022-10-01, using
   rows since 2022-02-24 whose h20 labels mature before origin minus embargo.
   From 2023 onward use the immutable saved T4 quarterly OOS anchor.
4. The fixed experts are this causally extended `identity_early`, half-mapped
   `all_platt_b050` and fully mapped `recent2y_platt_b100`.
5. 2022-10-01 through 2022-12-31 is an out-of-sample online warm-up. August and
   September form a deliberate gap after the mapping window. Warm-up is not
   used to fit rank, maps or the Q4 anchor. An outcome is admitted only after
   its h20 maturity plus embargo, exactly as in T31.
6. Screen is mature 2023, validation is mature 2024, and opened 2025--2026 is
   diagnostic only.

Rows are real CBR publication/currency events. Weekend holds are not duplicated.

## Online candidates

Reuse the audited T31 arithmetic mixture and update order. Start uniform on the
first warm-up publication. Before every query, consume each prior publication
batch once iff its maximum h20 `maturity_ord` is strictly below
`(query_date - 2 days).toordinal()` and its publication date is earlier.
Update with mean cross-currency Bernoulli log-loss, learning rate `eta`, then
fixed share `gamma`.

Frozen grid is unchanged to isolate the warm-start effect:

- `eta in {0.25, 0.50, 1.00, 2.00}`;
- `gamma in {0.00, 0.01, 0.05, 0.10}`;
- deterministic priority low eta, then low gamma.

## Selection and gates

On 2023, a candidate is feasible only if versus identity it has AUC delta at
least +0.02, lower Brier and log-loss, and ECE delta at most +0.005. Select
minimum Brier, then frozen priority. Validate that exact candidate on 2024 with
the same gates; do not substitute a second-best row.

If either stage fails, `t32_selected` equals identity. Save the screen candidate
separately. Open 2025--2026 cannot select the map window, warm-up, expert set,
eta, gamma, gate or fallback. A passing historical result is still only a
prospective shadow because all later periods have been inspected previously.

## Required audit

- source hashes, T4 alignment and unique publication/currency keys;
- rank labels mature before 2022 origin;
- mapping, Q4-anchor training and warm-up are causal; map ends before a
  two-month gap, and every training label matures before its fit origin;
- every online feedback batch earlier than its query and fully mature;
- weights finite, non-negative, normalized and one state per date/candidate;
- independent rebuild of all 16 mixtures from saved states and expert values;
- exact screen winner, validation verdict and identity fallback;
- changing targets not yet mature at a prefix cannot change that prefix;
- complete overall/year/currency/currency-year metrics and 20/50-date paired
  intervals against identity and T25;
- explicit `selection_on_open_2025_2026=false` and `fresh_holdout=false`.
