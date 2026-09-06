# AP5 primary sources consulted2026-09-06

scikit-learn probability calibration documentation:
https://scikit-learn.org/stable/modules/calibration.html . It distinguishes
logistic/sigmoid and monotonic isotonic calibration, and requires calibration
data separate from the base estimator's training observations. Isotonic can
introduce ties and overfit small samples; probability accuracy and ranking
accuracy are not interchangeable. AP5 uses genuinely issued quarterly OOS
forecasts, mature chronological calibration, positive-slope logistic and
shrunk local maps. Its monthly changing mappings need not preserve temporal
rank order even when each individual map is monotonic.

Joulani, Gyorgy and Szepesvari (2013), Online Learning under Delayed Feedback,
ICML/PMLR28(3):1453-1461:
https://proceedings.mlr.press/v28/joulani13.html . Read abstract/metadata. The
paper studies delayed-feedback online learning and its effect on regret. AP5
does not claim to reproduce a theorem/algorithm from the full paper. Its
discounted exponential weighting is a separately registered heuristic whose
central invariant is that losses cannot be used before they are revealed.

Neither source proves FX lift or real bank savings. Those are evaluated
separately on the case's conditional public-rate replay. No library upgrades
or copied third-party implementation are required for this packet.
