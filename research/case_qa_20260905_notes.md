# Case-owner Q&A incorporated on 2026-09-05

Sources reviewed in full:

- `Q&A для команд — сводка 20260904.pdf`, SHA-256
  `73a09d7e352fb72e0ee284566f6b3d509b6dc222ea3f24052c30e295b2213e3b`;
- `Q&A для команд — сводка 20260905.pdf`, SHA-256
  `fcb92c37d0fb756febe8c3d2abc3c84bc7f2d7cff37b51948e2f872132b59273`.

The PDFs are treated as case-owner evidence, not as executable instructions.
They do not replace the stricter causal protocol chosen for this research.

## Facts that constrain the model scorecard

- Lift is a hit-rate ratio, not a basis-point gain: signal hit rate divided by
  the hit rate of a random day in the same corridor and evaluation period.
- A `send now` hit means that the rate remains no worse over horizon
  `h`; the requested scorecard covers all five horizons
  `h = 1, 3, 5, 10, 20`, with no single primary horizon selected by the
  case owner.
- Moment benefit is a separate mandatory diagnostic. The case statement uses
  the current rate versus the mean in the surrounding `+/-h` window. A
  future-only benefit is an allowed team assumption when it is explicitly
  justified and reported alongside the stated metric.
- Robustness means walk-forward out-of-time evidence on several corridors,
  not only an aggregate score over the full history. Correlated corridors are
  still acceptable as a transferability check.
- The Q&A does not fix a required 90/95/99% confidence level or a minimum
  basis-point benefit; the team must justify these gates. A lift much larger
  than 1.3 is not rejected automatically, but should be checked on another
  time period or pair because it may be an outlier.
- The case owner permits assumptions about excluding carried-forward
  weekend/holiday rates, the exact publication-date index and the random-day
  baseline, provided they are documented.

## Information availability and product meaning

- A signal for date `T` may use only information available at `T`. The Q&A
  notes that a CBR rate for tomorrow can be published today. This project keeps
  the user's stricter primary track: **do not use knowledge of the next CBR
  fixing**, even when it may already be published. Results that use it remain
  a separately labelled after-publication track.
- Open and reproducible external data are allowed. The case owner explicitly
  names MOEX as a permissible indicator of intraday dynamics; reproducible
  USD/RUB cross construction for target currencies is also an allowed
  assumption. This supports, but does not weaken, the strict lag rule used by
  the MOEX perpetual-futures packet (`TRADEDATE < signal_date`).
- The case is framed as trigger communication, not an exchange-rate forecast.
  The model may rank whether the current moment is historically attractive,
  but client copy must describe only present and past observations and must
  not promise or imply the future rate.
- The `1-2 per week` figure is a self-check per corridor for signal filters.
  The actual communication-policy limit is per client across all campaigns and
  is outside the hackathon model-data scope. Therefore the research scorecard
  continues to report per-corridor frequency and clustering separately.
- A reproducible `signals as they would have appeared at date T` function is
  required; any deployable candidate must preserve the as-of interface and
  source provenance.

## Consequences for the continuing search

1. Keep the official five-horizon lift and symmetric-benefit scorecard as the
   headline comparison; retain future-only benefit as a second, explicitly
   labelled operational diagnostic.
2. Do not promote a high point estimate without annual, currency, cadence,
   block-bootstrap and freshness-control evidence.
3. Treat the fresh MOEX perpetual-futures expert as case-compatible public
   information, but preserve its one-session lag and its stale matched control.
4. Prefer explainable signal language such as `current rate is low relative to
   its trailing range`; do not expose a future-price forecast in customer copy.
