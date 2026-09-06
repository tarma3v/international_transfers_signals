# AP25-E frozen protocol: specialist model for deficit-pace candidates

Registered 2026-09-06 before AP25 fits, scores, signals or scorecards. AP21-AP24
consistently show that rolling-730 ExtraTrees is the robust primary selector;
quality is lost when the cadence controller must add non-primary signals. AP25
therefore keeps the primary expert frozen and trains specialist models only on
the hard reserve pool, instead of asking one global model to solve both roles.

Compute frozen past-250 per-currency ranks for rolling ExtraTrees and AP12
known70/hazard30 reserve. The specialist training pool is outcome-free:
eligible known-next-not-lower rows with finite ranks, rolling rank<=.70 and
reserve rank>.70. It does not use trailing future decisions or labels. At each
quarterly OOS origin, training additionally requires the shared publication-h20
maturity mask with two-day embargo and complete unknown-horizon response.

Five fixed specialists use the AP13 feature matrix:

1. `pace_cat_mean_gate_full`: CatBoost regression of mean(y3,y5,y10,y20),
   expanding hard-pool history;
2. `pace_cat_mean_gate_roll3`: the same with trailing1095-day hard-pool history;
3. `pace_extra_mean_gate_full`: ExtraTrees regression of mean utility;
4. `pace_cat_y20_gate_full`: CatBoost classification of y20 survival;
5. `pace_cat_future5_gate_full`: CatBoost regression of clipped effective
   future-only h5 benefit, clipped at training 2nd/98th percentiles and divided
   by 200 b.p. before fit.

CatBoost geometry is fixed at AP19 values:320 iterations,depth6,learning rate
.035,L2=10,random strength=.5,Bernoulli subsample=.8,seed42. ExtraTrees uses
400 trees,depth8,min leaf25,max features .6,seed42. Every specialist is used
only as the pace expert with frozen rolling primary under the exact AP21 policy:
outer causal ranks,primary>.70,pace>.55 while trailing365 rate<1 and reserve>.70,
month24 rescue,known-down veto,max2/ISO-week. No threshold changes.

Compare five fresh policies with AP21 strict, AP23 competence-pace, AP24 best
ranker-pace and frozen accuracy controls. Select only on early2023 using the same
h3/h5/h10/h20, benefit, cadence and calendar gates. Opened2024-2026 remains a
diagnostic. Audit exact hard-pool membership, maturity, target clipping, every
fit and score, future feature/label corruption prefix invariance, policy state
and paired20/50-date uncertainty. Target is a materially higher strict minimum
lift without reducing future-only benefit. H1 remains known and excluded;
historical receipt time is calendar-assumed and bank execution is unvalidated.
