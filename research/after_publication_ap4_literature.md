# AP4 primary-source note,2026-09-06

Gensheimer and Narasimhan (2019), *A scalable discrete-time survival model for
neural networks*, PeerJ7:e6257. DOI https://doi.org/10.7717/peerj.6257 .
Primary indexed abstract/method excerpt retrieved at
https://pmc.ncbi.nlm.nih.gov/articles/PMC6348952/ . Direct article opening was
blocked by a browser challenge/403; do not claim a full-paper read.

Author implementation inspected:
https://github.com/MGensheimer/nnet-survival/blob/master/nnet_survival.py
(surv_likelihood and make_surv_array). The per-interval likelihood distinguishes
survived intervals and the failure interval, excluding intervals after failure.
Its architecture is neural; AP4 borrows only the discrete hazard factorization,
implemented independently with classical pooled logistic regression and HistGB.

Our mapping: first strictly cheaper CBR observation is the event; surviving to h
is exactly our favourable-price target. Conditional probabilities combine via
products of interval survival. We require complete matureh20 rows, so no ad-hoc
censoring midpoint rule is imported. This statistical construction does not
demonstrate FX performance, bank savings, source availability, or causal effects.
Those must be measured separately in our replay. No third-party model code copied.
