# AP29-E frozen protocol: mature-competence backstop for the y20 specialist

Registered 2026-09-06 before AP29 signals or scorecards. Error analysis on the
already-open late period showed that AP26's y20-shrink200 core is useful, while
the raw-CatBoost backstop used by strict AP27 is the weak component. This makes
AP29 a retrospectively motivated challenger, never a fresh holdout claim.

Test exactly one policy and no threshold grid. Keep AP27's strict r60 state
machine unchanged: rolling ExtraTrees is primary; AP26 y20-shrink200 is the pace
expert; a backstop is considered only after84 days, when pace did not fire and
trailing365 currency rate is below.95. Replace only the backstop score: use the
AP23 soft730_pace mature-competence score instead of raw global CatBoost.
Require its prior250 same-currency rank>.60 and the unchanged reserve rank>.70.
Month24 rescue, known-down veto and max2/ISO-week remain exact.

All three component scores are already quarterly OOS or causal. AP23 competence
uses only expert outcomes whose publication h20 matured at least two days before
the decision. Current/future labels never enter a rank, weight or state update.
The sole fresh AP29 policy is selected on early2023 with unchanged joint gates;
opened2024-2026 is diagnostic. Frozen controls are AP29's AP26 core, AP23, AP27
r60/r70, AP21, AP18 and AP17.

Audit exact source hashes, four causal ranks, trailing-rate state, reasons,
known-down/month/cap behavior, future-score prefix invariance and paired20/50-date
uncertainty. Acceptance is min h3/h5/h10/h20 lift>2.4, min currency rate>=1,
zero empty complete months and max2/week. H1 is known after receipt and excluded.
Receipt timestamps remain calendar-assumed and bank execution is unvalidated.
