# AP3 literature note (primary paper read2026-09-06)

Pablo Montero-Manso and Rob J Hyndman, *Principles and Algorithms for Forecasting
Groups of Time Series: Locality and Globality*, arXiv:2008.00444v3 (2021).
Primary full text: https://arxiv.org/html/2008.00444v3 ; read especially sections
5.6 and6.4 on normalization, and discussion of local/global modeling.

The paper does not prescribe universal scaling. Its scale experiment compares
unscaled data, normalized data, and normalized data with scale restored as a
feature. Scale can contain useful information; benefits depend on data/loss.

AP3 interpretation, not a replicated result of the paper: test normalized
future-floor targets while keeping known volatility features in X, and retain
raw versus prediction-only-rescaled versus refitted normalized controls. This
addresses AP2's empirical score-scale shifts without assuming normalization
must improve FX lift. The paper studies different benchmark tasks and does not
establish our after-publication availability, trading benefit, weekly cadence,
or the effectiveness of our residual models.
