# occurrence

Pipeline to estimate occurrence rates of orbital companions (companions per star) from
injection-recovery completeness maps and companion posterior samples. It fits a censored Poisson
point-process likelihood to the unbinned posterior samples, with a piecewise-constant
(histogram) model or smooth parametric models.

## Workflow
Results are organized in three tiers beneath an output directory:

1. **Tier 1** (`mtrue`, `msini`, `qtrue`, `qsini`): per-star completeness maps in the chosen mass
   or mass-ratio coordinate, built from each system's `recoveries.csv`. The `_P` variants
   (`mtrue_P`, `msini_P`, `qtrue_P`, `qsini_P`) use orbital period in days instead of
   semi-major axis; give period edges in `a_edges`. Injections use their injected periods,
   and companion draws are converted with Kepler's third law (host mass from the star
   catalog plus each draw's companion mass).
2. **Tier 2** (one folder per stellar sample): the sample's average completeness map and its
   companion posterior draws, each with its interim prior and completeness.
3. **Tier 3** (one folder per run): MCMC fits, summaries, and plots for the chosen bins and models.

`run.run_multiple()` runs any combination of tiers. Analysis projects (for example
`paper_mass_function_stellar_properties`) call it with their own configuration.

## Models
- **Histogram**: piecewise-constant occurrence-rate density, optionally with a Gaussian-process
  prior on the bin heights.
- **Parametric** (`mcmc_powerlaw.MODEL_REGISTRY`): sigmoid, log-Gaussian (`logG`), log-linear,
  escarpment, and broken power law (`bpl`).

## Modules
- `run.py`: entry point; builds the three tiers and runs the fits.
- `main.py`: steps that `run.py` calls (prepare recoveries and maps, sample posteriors, fit, plot).
- `completeness_utils.py`: single-system and average completeness maps and interpolators.
- `rvsearch_borrowed.py`, `radvel_borrowed.py`: the parts of RVSearch and radvel the maps need,
  copied so the package does not depend on them.
- `sampling_utils.py`: posterior samplers, interim priors, and per-sample completeness.
- `fit_utils.py`: catalog and exposure-grid inputs for the fits; saves and loads `fit_data.npz`.
- `likelihood.py`: model densities and the cached Poisson likelihoods.
- `mcmc.py`: `emcee` sampling for the histogram and parametric models, and the model plots.
- `mcmc_powerlaw.py`: registry of parametric models (density, parameter labels, plot colors).
- `occurrence_utils.py`: posterior summaries (medians and highest-density intervals).
- `convergence.py`: autocorrelation-time convergence checks for saved chains.
- `plotting_utils.py`: completeness maps, catalog plots, occurrence histograms, and tick formatters.
- `post_fit_analysis.py`: publication products from finished fits (LaTeX variables and tables,
  ΔBIC, CDF comparisons, companion and stellar-parameter plots).
- `matplotlibrc`: plot style used by every figure.
- `cls_files/`: California Legacy Survey star and companion tables.

## Requirements
Python 3.8+ with numpy, scipy, pandas, astropy, emcee, matplotlib, and corner.

## Tests
```bash
python -m pytest tests
```
