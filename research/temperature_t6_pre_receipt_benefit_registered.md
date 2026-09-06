# Temperature T6 frozen protocol — expected benefit before CBR receipt

Registered 2026-09-06 before T6 metrics were computed.

T6 adds expected future-only CBR basis points to every validated pre-receipt
temperature state: the T4 history-only premarket expert, all eight T5 market
cutoffs, and the T3 16:30/17:30 updated market bridges. The target at horizon h
is the signed basis-point difference between the current effective CBR rate and
the mean of the next h publications. No symmetric past window and no tomorrow
CBR rate may enter the features.

Fit one independent quarterly Ridge model per state and h=1/3/5/10/20. Features
are the frozen T4 compact CBR vector plus that state's raw causal rank/probability
and its already causal calibrated P(now best at h). Use StandardScaler,
Ridge(alpha=10), train-only 1st/99th target winsorisation, 730-day half-life,
fixed 2022-02-24 minimum training date, mature horizon before origin minus two
days. Do not select states or hyperparameters from T6 outcomes.

Report MAE/RMSE against the weighted train-mean prior, Spearman correlation,
sign accuracy and predicted-versus-actual means on 2024 and the already-open
2025--2026 period. Preserve all upstream candle/CBR corruption evidence and
prove future state/feature/target/maturity corruption cannot change earlier
predictions. Units remain CBR basis points; this is not executable bank savings.
