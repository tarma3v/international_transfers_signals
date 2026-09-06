# AP6 source and implementation distinction

Read the official scikit-learn StackingClassifier documentation on 2026-09-06:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html

The documented distinction between base fitting and out-of-fold meta training,
and the warning about training a prefit stack on its base models' own train,
motivate a strict issued-prediction design. AP6 does not call the default
StackingClassifier CV: it reuses quarterly-issued AP3/AP4 scores, then trains
monthly meta models with fully matured h20 and a two-day embargo. Source-only
controls and exact earlier policies distinguish useful information from merely
adding layers. No theorem or published FX performance is claimed.

The nonnegative logistic objective, bounded same-currency pair construction,
small HistGB and local/global OOS residual variants are explicitly registered
experiments, not replications of a paper. Scaling and residual availability are
reconstructed in after_publication_ap6_audit.py. The outcome was no confirmed
new superiority. See the report and complete saved negative experiments.
