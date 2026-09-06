# Temperature T2 frozen protocol — 15:30 to receipt market bridge

Registered 2026-09-06 before T2 scores or metrics were computed. This packet
tests the user's proposed bridge between the 15:30 market model and actual CBR
receipt. It does not use tomorrow's rate and assumes both fixed query clocks are
still pre-publication.

Evaluate exactly 16:30 and 17:30 Moscow. At both clocks start with the frozen
15:30 score: mean completed CNY/RUB TOM close in 10:00--15:30 divided by the
current effective CBR CNY rate. The hold control keeps this score unchanged.
The sole update rule adds one-for-one the log return from the last completed
CNY candle available before 15:30 to the last completed candle available before
the query clock. If no new valid candle exists, delta is zero. Admit only
completed same-day candles with nominal ten-minute end and observed end strictly
before the cutoff. Do not screen delta weights, clocks, signs or instruments.

Convert each raw score to a same-currency causal percentile using the frozen
250-row window/minimum 20. Preserve the rolling-22%/20 push policy only as a
diagnostic; report all h=1/3/5/10/20 on 2024 and opened 2025--2026 without
selecting a winner. Separately calibrate each continuous rank quarterly with the
AP50 mature-only calibrator and compare Brier/log-loss/ECE against its own hold
control and train prior.

Physically corrupt every candle at/after a future date-cutoff and require every
earlier score to remain identical. The historical CBR receipt time is not
certified; 16:30/17:30 therefore test a pre-receipt information bridge, not a
claim that publication never occurred by those clocks.
