# AP10-E amendment: strong market-information control

2026-09-06, BEFORE fitting the new past_market control. First AP10-E nine-model
packet already ran: Hist CBR-only effective target0.997804 without new fixing vs
1.975266 with it. Its original artifacts/registration remain unchanged. The user
expects a meaningful gain versus existing strong approaches, so a CBR-only
information ablation is insufficient to establish that stronger claim.

Add ONE information set at the same18:30: current-effective CBR features plus
exactly the SAME completed market candles as the published+market model, but
ALL CNY/local market bases and vol normalizers reconstructed relative to
current-effective CBR, not announced CBR. No explicit future fixing from any
currency may enter this control. Market may naturally react to publication;
this is direct-information-withholding at18:30, not the actual before-release
15:30 product. A comparison to earlier-time production belongs separately.

Fit the same three models/4 outputs on the exact same frozen train/query masks.
No known-next gate in this information set. Add4 policies to the original23;
reuse all prior predictions without modifying them. Same early joint selection,
fallback and later support. Prespecified pairs now include market vs past_market
for each output, plus new published+market gate vs past_market. Separately
measure improvements of learned methods vs known-sign and known-change rules.
No late winner selection. Save extended results separately from first AP10-E.
