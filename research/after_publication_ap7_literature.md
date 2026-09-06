# AP7 reading and implementation scope

Read the abstracts on 2026-09-06, not the complete PDFs:

- Meinshausen (2006), Quantile Regression Forests:
  https://www.jmlr.org/papers/v7/meinshausen06a.html
- Cevid et al., Distributional Random Forests, revised2022:
  https://arxiv.org/abs/2005.14458

These motivate using weighted outcomes as an estimated conditional distribution
instead of reducing the model to a mean. The latter paper describes an MMD
distributional splitting criterion. AP7 does NOT implement that criterion or
claim to replicate either paper. Its smaller temporal split ensemble uses
multioutput squared-error tree splits on five normalized path coordinates,
then separate, later mature paths in each leaf. No asymptotic guarantee or
published FX return is transferred to this custom experiment.

The nearest-neighbor libraries, chronological Ridge residual distributions,
Gaussian comparator and nonlinear scenario-by-scenario utilities are locally
registered modeling choices. Results did not establish an improvement. Exact
library membership and source availability are checked in the audit; repeated
retrospective evaluation and assumed receipt timestamps remain limitations.
