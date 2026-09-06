# Temperature T14 frozen protocol — horizon-aware 09:00 perpetual route

Registered 2026-09-06 after the audited T13 selection and before the T14
snapshot artifact was built.

T14 extends T12 with a 09:00 Moscow snapshot only when T13 has a physically
completed same-day CNYRUBF candle and a finite calibrated prediction. The T13
screen-only rule selected `perp_basis_rank` for h1 and h3. It rejected h5/h10/
h20 at 09:00 and every perpetual replacement at 10:00. T14 must implement that
selection exactly; no 2025–2026 result may alter it.

At 09:00, h1 and h3 use the T13 routed probabilities. h5/h10/h20 retain the T4
history-only probabilities. Expected future-only basis points remain the mature
premarket prior at every horizon because T13 did not validate a new magnitude
head. T10, T5/T3, T11, AP51/T7B and AP37 push behavior remain unchanged.

Because only two horizons consume the new candle, T14 introduces optional
horizon-specific `source_at_h*`, `source_kind_h*`, `phase_h*`, `confidence_h*`
and `availability_evidence_h*` fields. The lookup must prefer these fields and
fall back to the legacy row-level provenance for older artifacts. Thus an h1
query at 09:15 may correctly say `moex_perpetual_prefix`, while the h5 query on
the same row remains `cbr_history` with limited confidence.

Audit exact T12 identity outside the new rows, the frozen T13 selection,
horizon-specific provenance, `source_at_h <= valid_from`, finite bounded heads,
every-day/currency queryability, weekend stale behavior, future-snapshot
invariance and unchanged push counts. Historical exchange and CBR receipt
timestamps remain uncertified; no bank economics are claimed.
