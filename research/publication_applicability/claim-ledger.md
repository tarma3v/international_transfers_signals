# Internal evidence and gap ledger - 2026-09-06

Audience: project team. Decision: whether to retain an after-CBR-publication signal,
under what information and execution assumptions, and whether the old reported
result is defensible. Primary-source research; no transactions or bank login.
This ledger is internal provenance, not the report manuscript.

## Consequential claims and sources

| Claim | Source / location / provenance | Confidence, contradiction, remaining gap |
|---|---|---|
| T signal may use only information available at T; tomorrow CBR can be published today | Case-owner Q&A 20260905, p5; supplied local PDF. Same answer 20260904 p4. Main read extracted text and rendered Sep5 p5. | High on statement. Conditional interpretation on permission for a particular time/index; no blanket approval of implementation. |
| App price is real-time and linked to liquidity suppliers | Case-owner Q&A 20260905 p7; main read full relevant page and image. | High. Contradicts assumed day-fixed app execution. Exact rail/quote-lock contract still unknown. |
| Forecast model not mandatory; all horizons and product interpretation needed | Same Q&A pp5-7,14; Sep4 corroboration. | High. Does not guarantee acceptance or business value of any particular signal. |
| 18:00 is typical upper time, not guaranteed release clock | CBR FAQ, “В какое время Банк России устанавливает и публикует официальные курсы...”, update date not displayed; https://cbr.ru/Reception/TopicalMessage/Page/2661 ; parent web native refs turn110view0, turn128view0, wordlim200 | High; directly re-opened by coordinator. Need event timestamp rather than clock constant. |
| Next-calendar-day effect; course lasts until next fixing | CBR FAQ, “Каковы сроки действия официальных курсов...”, date not displayed; https://www.cbr.ru/Reception/TopicalMessage/Page/2656 ; refs turn110view1, turn128view1, wordlim200 | High on ordinary rule. Does not identify actual intraday timestamp. |
| CBR does not undertake exchange at official rate | CBR instruction 6956-U dated02.12.2024, registered19.12.2024 No80631, §1; https://www.cbr.ru/queries/unidbquery/file/90134/6223 | High; official normative document, coordinator/worker inspected. No assertion that CBR is retail provider. |
| XML date is insufficient first-seen provenance | CBR XML docs https://www.cbr.ru/development/SXML/ ; GetLatestDateTime schema https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx?op=GetLatestDateTime | High on missing local first-seen log. Worker observed date-only midnight result and query-date fallback, not used as a historical proof. |
| Routes have distinct pricing; general CBR clause needs scoped clarification | Alfa DКБО approved28.04.2026 No1761, 298PDF pages, printed pp36,79-80,103,261; https://alfabank.servicecdn.ru/site-upload/f7/8e/1869/dogovor_cbo_1052026.pdf?previewDocument=true ; parent turn121view0 wordlim200 | High on text; not established latest at Sep6 or applicable to user's particular transfer. General §10.4.1.4 vs exceptions §10.4.1, §3.2.2.1, special §11.16 and appendix19. Main downloaded, extracted exact clauses and visually inspected printedp80. |
| Ordinary app exchange is dynamic and amount-dependent | Alfa “Обмен валюты” https://alfabank.ru/lp/retail/info/exchange/ ; primary search snippet turn121search0, direct open502 | Medium/high on quoted public description; not universal cross-border rail. Not treated as authenticated quote. |
| Saved2.459 reads next-effective row without announcement gate | research/round4_research.py::_publication_matrix; retained output pkl and CSV; new research/publication_semantics_audit.py | High, source inspected and replayed. 3260rows/732signals; reproduced2.458731785063752;147signals later-inferred publication. Calendar screen only. |
| No executable old-price window established | Synthesis of case Q&A + bank exceptions + absence of quote history | Bounded negative: not found/established, not proof none exists. Need specific route and timing evidence. |

Original attachments, available locally but not embedded in report:

- /Users/jeck5iv/Downloads/Q&A для команд — сводка 20260904.pdf
  SHA256 73a09d7e352fb72e0ee284566f6b3d509b6dc222ea3f24052c30e295b2213e3b.
- /Users/jeck5iv/Downloads/Q&A для команд — сводка 20260905.pdf
  SHA256 fcb92c37d0fb756febe8c3d2abc3c84bc7f2d7cff37b51948e2f872132b59273.

All sources accessed06.09.2026 (local timezone); originals have dates in names.
Keep imported primary-source summaries within source word limits; the report's
contract discussion is deliberately short. No long bank/CBR verbatim extracts.

## Discovery and targeted follow-up log

1. Located repository docs/source and original case Q&A; reconciled supplied
   original text against prior project notes (omitted real-time quote statement).
2. CBR lane: exact publication-time and effective-date FAQ, current6956-U,
   older6290-U/calendar continuity, 2022 trading windows, current XML/SOAP
   schemas. Checked weekends and distinction between effect and publication.
3. Alfa lane: official transfer and exchange pages, DКБО route clauses,
   cross-border SBP, card/phone, beneficiary conversion; examined May2026 and
   November2025-approved editions. Did not claim exact operational transition
   date or latest Sep2026 contract.
4. Disconfirming follow-up: general §10.4.1.4 explicitly mentions CBR conversion;
   read full context instead of discarding it. Scoped against execution delay,
   opening exception and special-route provisions. Coordinator verified primary.
5. Repository follow-up: unconditional i+1 matrix, stored candidate replay,
   calendar mismatch count and concrete TJS Saturday example. No refit, no new
   performance claim, no post-filtered score advertised as fixed.
6. Public PDF screenshot twice returned cache errors; downloaded same official
   PDF and rendered locally. The ordinary bank-exchange page returned502;
   primary indexed snippet supports only its narrow dynamic-exchange claim.

Searches comprised these bounded claim families and exact-source reads; further
general search stopped because it cannot establish private executable quote
history or reconstruct missing historical first-seen timestamps. An affirmative
old-price window or a repaired after18:00 lift needs new evidence, not inference.

## Gap matrix at synthesis

| Missing fact | What is known | Next action if pursuing implementation |
|---|---|---|
| Exact historical release timestamps for all features | Effective dates and ordinary calendar rule | Obtain timestamped archives or label calendar assumptions; create prospective event log |
| Quote valid after customer action | Public terms distinguish routes | Confirm concrete route, lock moment, TTL, conversion day with bank |
| Monetary benefit over waiting | CBR hit-rate proxy, no quote panel | Authorized quote panel with matched amount/recipient/channel and fee-inclusive target |
| Corrected after-publication score | Old score reproducible but availability-inconsistent | Rebuild train/calibration/test replay, simple known-change baseline, fresh prospective test |

## Deliverable and verification

Canonical manuscript: report-source.md. One PDF rendered from it by
research/build_publication_applicability_report.py. Exact final verification is
recorded in results/research/publication_applicability/verification.json.
The plan tool was not exposed by ALL_TOOLS; scope and single-in-progress stages
were maintained in functions session store through discovery/follow-up/synthesis.
