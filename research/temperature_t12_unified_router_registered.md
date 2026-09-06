# Temperature T12 frozen protocol — early market and stable benefit router

Registered 2026-09-06 after T9--T11 and before the T12 artifact was built.

T12 replaces T8B as the product snapshot package without changing the frozen
push or after-publication experts. It adds one 10:00 snapshot only when T9 proves
a completed same-day CNY candle exists strictly before 10:00. Probability heads
come from the fixed T10 horizon map; if unavailable, no row is emitted and the
00:00 T4 fallback remains latest. No modern-data market claim is made before
10:00 because T9 measured zero 2025--2026 CNY candle coverage through 09:30.

Expected future-only basis points use a stability rule fixed from the causal
2024 evidence. Premarket and 10:00 use the T11 prior leg because the corresponding
Ridge models were not stable on 2024. From 10:30 through 17:30, h1/h3/h5/h10 use
the T11 quarterly mature-only adaptive blend; h20 retains the prior because its
2024 point MAE did not improve. After the assumed receipt event, keep AP51/T7B.

Retain T8B's phase order, every-day hold/stale behavior, source timestamps,
probabilities h=1/3/5/10/20, separate push_now and future-snapshot invariance.
The saved 18:30 event remains explicitly calendar-assumed, not a certified CBR
receipt. Production must switch on the actual observed receipt timestamp.

Audit exact source hashes, unique currency/valid_from, source_at<=valid_from,
finite/bounded outputs, early-source physical timing, stable benefit source
selection, every-day queryability, weekend stale behavior, push location and a
fabricated-future snapshot. No bank execution claim is allowed.
