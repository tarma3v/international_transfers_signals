# AP2-D20: full replay with a delayed public market feed

Registered after original AP2 results, before evaluating any delayed-feed scores.
Original AP2 is retained unchanged in results/research/after_publication/ap2.
This is a follow-up sensitivity experiment, NOT a preregistration of the already
observed instant-feed numbers and NOT a fresh holdout.

Primary evidence checked2026-09-06: https://www.moex.com/a8531 describes paid
real-time versus free delayed ISS data (including chart/candle data). The exchange
defines delayed streaming data as at least15min in
https://www.moex.com/files/414zjnha8jdfpvpdezmjms1b4t . Neither source certifies
the historical exact receipt of every archived candle. Do not equate trade end
or bar completion with client receipt. Original AP2 assumes zero extra feed
latency and is therefore not a validated free-ISS18:30 replay.

Freeze a20-minute market delay:15min customary delay plus5min operational margin.
This is a conservative MODEL ASSUMPTION, not a guaranteed service-level bound.
Decision still18:30 MSK; both actual candle end+20min <18:30 and nominal
begin+10min+20min <=18:30 must hold. CBR receipt still assumed18:00; no change
to target/index/dates or claimed actual release knowledge. Live receipt logging
and abstention on absent data remain mandatory. No subscription is purchased.

Recompute the entire market feature history, every model fit, genuinely OOS
local residual, causal score rank, signal and selection under this delay.
All55 policies and model settings from AP2 unchanged. Selection remains2023
with h20 resolved before2024; 2024-2026 remains already-opened retrospective.
Store independently in results/research/after_publication/ap2_delay20. Do not
choose the better delay after testing; use20min as the case-facing public-feed
scenario, zero-delay only as a separate sensitivity reference.
