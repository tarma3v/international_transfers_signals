# AP9 source read before fitting

Suresh K., Severn C., Ghosh D. (2022),
[Survival prediction models: an introduction to discrete-time modeling](https://link.springer.com/article/10.1186/s12874-022-01679-6).
Read Methods: Discrete-time survival models, equations1-2/person-period
construction and Alternative approaches/censoring, on2026-09-06.

The paper explains the product of observed survival terms plus a failure term,
with no contributions after the last observed interval. This admits binary
classification on person-period rows. It also discusses bias from counting
incomplete intervals as fully survived and the non-informative censoring
assumption. It does not establish predictive success in our FX case.

Our adaptation: each new fixing observation is one discrete step. Fine AP9
uses only steps actually published before fitcutoff; coarse AP9 discards the
unfinished coarse interval for BOTH observedearlyfailures and survivors.
Partial followup is not an invented fullh20 survival label. Administrative
cutoff and calendar/regime dependence remain limitations; use chronological
controls and dateblock uncertainty rather than iid interval standard errors.
