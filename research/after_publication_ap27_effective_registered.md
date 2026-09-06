# AP27-E frozen protocol: sparse causal backstop for pace specialists

Registered 2026-09-06 before AP27 signals or scorecards. AP26 y20-shrink200
improves late min lift to2.417 and h5 to2.483 but misses the all-currency cadence
floor by about3%. AP27 keeps that score and all model fits frozen. It adds a
second, global CatBoost-utility backstop only when the specialist pace rule does
not fire and the currency is causally behind schedule.

Five fixed policies are declared:

1. `s200_cat95_r70`: y20-shrink200 pace; Cat rank>.70 when trailing365 rate<.95;
2. `s200_cat100_r70`: Cat rank>.70 when trailing rate<1.00;
3. `s200_cat95_r60`: Cat rank>.60 when trailing rate<.95;
4. `s100_cat95_r70`: y20-shrink100 pace with the first backstop;
5. `s200_cat_silence14_r70`: Cat rank>.70 only after14 calendar days without a
   selected signal.

The backstop also requires reserve rank>.70, at least84 days of history, current
eligibility, available weekly capacity and failure of the specialist pace test.
Primary rolling rank>.70, specialist pace rank>.55 while rate<1, month24 rescue,
known-down veto and max2/ISO-week remain exact. Backstop decisions are reason3;
month rescue becomes reason4. All ranks use only prior250 same-currency scores.

Selection is still early2023 with the unchanged joint gates; opened2024-2026 is
diagnostic. Compare AP26, AP23, AP21 and older strict controls. Audit every rank,
trailing-rate/last-signal state, reason, prefix invariance and paired20/50-date
uncertainty. Target is min h3/h5/h10/h20 lift>2.4, min currency rate>=1, zero
empty complete months and max2/week. H1 remains excluded; receipt timestamps are
calendar-assumed and actual bank execution is unvalidated.
