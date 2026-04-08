# occurrence

Pipeline to estimate occurrence rates of orbital companions (number of companions per star) using user-provided injection–recovery completeness maps and companion posterior samples.

**What it does**
- Builds per-star completeness maps from per-system `recoveries.csv` outputs.
- Aligns / interpolates maps to create an average completeness map and saves per-system interpolators.
- Samples companion posterior draws, computes interim priors, and attaches completeness values for each draw.
- Bins parameter space (semi-major axis × mass) into cells and runs an MCMC (via `emcee`) to infer occurrence rates per cell.
- Produces diagnostic plots (corner plots, completeness maps annotated with inferred rates).

**Key files**
- `main.py` — high-level workflow functions (map prep, post sampling, MCMC entrypoints, plotting).
- `completeness_utils.py` — make single-system maps, average maps, and interpolators.
- `sampling_utils.py` — sample posteriors, compute priors, attach completeness.
- `occurrence_utils.py` — assign samples to cells, likelihood, and run MCMC.
- `plotting_utils.py` — plotting helpers.

Requirements
- Python 3.8+ and these typical packages:
  - numpy, scipy, pandas, astropy, emcee, matplotlib, corner