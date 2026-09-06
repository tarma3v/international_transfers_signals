# AP26-E frozen protocol: cold-start shrinkage for pace specialists

Registered 2026-09-06 before AP26 scores, signals or scorecards. AP25 specialist
scores improve late accuracy and future-only benefit but the hard-pool contains
fewer than100 mature rows until 2024, causing constant cold-start predictions and
an early cadence failure. AP26 does not refit or retune those models. It combines
them with frozen global CatBoost utility using only the specialist training count
known at each quarterly origin.

For each row, map the AP25 `n_specialist_train` from its OOS quarter. Five fixed
pace scores are declared:

1. `y20_hard100`: global CatBoost while n<100, AP25 y20 specialist when n>=100;
2. `y20_shrink100`: weight n/(n+100) on y20 specialist, remainder on CatBoost;
3. `y20_shrink200`: weight n/(n+200) on y20 specialist;
4. `mean_shrink100`: weight n/(n+100) on expanding mean-utility specialist;
5. `future5_rank_shrink100`: blend causal per-currency ranks of CatBoost and the
   future5 specialist with weight n/(n+100), avoiding incompatible raw scales.

Counts and scores are frozen OOS artifacts; they contain no evaluation labels at
the current row. Hard100 is exactly the AP21 rolling/Cat policy throughout 2023,
because all 2023 specialist counts are below100. Every score is used only as the
pace expert with frozen rolling primary and exact AP21 thresholds, reserve gate,
month24 rescue, known-down veto and max2/week.

Selection and all gates remain early2023; opened2024-2026 is diagnostic only.
Compare AP21, AP23, AP24 and AP25 controls. Audit exact quarter-count mapping,
mixtures, causal ranks, state, future-score/count corruption prefix invariance and
paired20/50-date uncertainty. Target is strict min h3/h5/h10/h20 lift>2.4 with
every currency at1..2 signals/week, no empty complete months and improved
future-only benefit. H1 is excluded; receipt timestamps remain calendar-assumed
and bank execution is unvalidated.
