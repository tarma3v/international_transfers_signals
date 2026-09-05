# Round 7 verification

2026-09-05, branch `ivan-experiments`.

- Full repository test suite: 112 tests passed.
- Three new checks cover nominal candle completion, missing weekend data,
  and invariance to changes in unresolved future labels.
- Direct-feature future-price corruption and residual-model future-outcome
  corruption checks passed during the experiment runs (with positive controls).
- Data loading checks the 10 direct-pair files against saved SHA-256 hashes.
- Matched-date audit and paired four-week-block bootstrap completed for all
  five horizons. Intervals are conditional on the frozen candidates; they do
  not erase historical research selection bias.
- The new report contains 11 pages. All pages were rendered with Poppler and
  visually inspected. PDF text checks found no out-of-page text and no Unicode
  replacement characters.
- No new candidate replaced the frozen push reference. Widget calibration is
  evaluated only at 15:30; no live widget or bank-transfer execution is shipped.

Report: `output/pdf/ivan_direct_pairs_and_widget_report.pdf`.
