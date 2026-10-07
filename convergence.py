"""Autocorrelation-based convergence summaries for saved MCMC chains.

Each fit's production chain (burn-in already discarded) is compared with
its integrated autocorrelation time ``tau``; a chain spanning at least
``target`` (by default 50) autocorrelation times in every sampled parameter
is considered converged.  :func:`write_convergence_summary` records one
experiment's chains in ``saved_chains/convergence.txt``, and
:func:`summarize` does so for every experiment beneath a results directory.
"""

import logging
from pathlib import Path

import emcee
import numpy as np

CONVERGENCE_FILENAME = "convergence.txt"
DEFAULT_TARGET = 50

# Names of the parameters each smooth model samples, in sampler order.
# Amplitudes and widths are sampled as natural logarithms.
SMOOTH_PARAMETER_NAMES = {
    "sigmoid": ["ln B1", "ln B2", "log10 x_t", "ln W"],
    "logG": ["ln A", "mu", "ln sigma"],
    "escarpment": ["ln C1", "ln C2", "log10 x_t1", "log10 x_t2"],
    "loglinear": ["ln D1", "ln D2"],
    "bpl": ["ln C", "log10 x_break", "beta", "gamma"],
}


def _integrated_time(samples):
    """Return emcee's autocorrelation times without its short-chain warning."""
    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        return np.asarray(
            emcee.autocorr.integrated_time(samples, quiet=True), dtype=float
        )
    finally:
        logging.disable(previous)


def _model_name(path):
    stem = Path(path).stem
    return stem[len("chains_"):].split("_bin")[0]


def chain_convergence(path, target=DEFAULT_TARGET):
    """Return the convergence statistics of one saved chain file.

    The result maps ``path``, ``model``, ``sampled`` (the array that was
    sampled), ``tau_source``, ``n_steps``, ``n_walkers``, ``parameters``
    (``(name, tau, steps_per_tau)`` rows), ``steps_per_tau`` (the minimum),
    ``required_steps`` (for ``target``), and ``converged``.
    """
    path = Path(path)
    model = _model_name(path)
    with np.load(path) as chain:
        if model == "piecewise":
            latent = chain["latent_chains"] if "latent_chains" in chain else None
            if latent is not None and latent.size:
                samples, sampled = latent, "latent_chains (GP parameterization)"
                n_bins = samples.shape[2] - 4
                names = ["total occurrence", "ln GP amplitude",
                         "ln length scale (a)", "ln length scale (M)"] + [
                    f"bin {index} (latent)" for index in range(n_bins)
                ]
            else:
                samples, sampled = chain["chains"], "chains (independent bins)"
                names = [f"bin {index} height"
                         for index in range(samples.shape[2])]
            stored_tau = None
        else:
            key = ("transformed_chains" if "transformed_chains" in chain
                   else "chains")
            samples, sampled = chain[key], key
            names = SMOOTH_PARAMETER_NAMES.get(model)
            stored_tau = (chain["autocorrelation_time"]
                          if "autocorrelation_time" in chain else None)
        samples = np.asarray(samples, dtype=float)
    if samples.ndim != 3 or samples.shape[0] < 2:
        raise ValueError(f"{path} does not hold a (steps, walkers, parameters) "
                         "chain")
    n_steps, n_walkers, n_parameters = samples.shape
    if names is None or len(names) != n_parameters:
        names = [f"parameter {index}" for index in range(n_parameters)]
    if stored_tau is not None and np.size(stored_tau) == n_parameters:
        tau, tau_source = np.asarray(stored_tau, dtype=float), "saved by the fit"
    else:
        tau, tau_source = _integrated_time(samples), "computed from the chain"
    ratios = n_steps/tau
    return {
        "path": str(path),
        "model": model,
        "sampled": sampled,
        "tau_source": tau_source,
        "n_steps": n_steps,
        "n_walkers": n_walkers,
        "parameters": [(name, float(value), float(ratio))
                       for name, value, ratio in zip(names, tau, ratios)],
        "steps_per_tau": float(ratios.min()),
        "required_steps": int(np.ceil(target*tau.max())),
        "converged": bool(ratios.min() >= target),
        "target": target,
    }


def format_convergence(result):
    """Return a readable text block describing one chain's convergence."""
    target = result["target"]
    lines = [
        f"Model: {result['model']}  (sampled array: {result['sampled']})",
        f"  Chain file: {Path(result['path']).name}",
        f"  Production steps: {result['n_steps']} per walker x "
        f"{result['n_walkers']} walkers (burn-in already discarded)",
        f"  Autocorrelation times (tau, in steps; {result['tau_source']}):",
        f"    {'parameter':<22} {'tau':>8} {'steps/tau':>10}  status",
    ]
    for name, tau, ratio in result["parameters"]:
        status = "ok" if ratio >= target else "TOO SHORT"
        lines.append(f"    {name:<22} {tau:8.1f} {ratio:10.0f}  {status}")
    if result["converged"]:
        verdict = (f"converged: spans {result['steps_per_tau']:.0f} "
                   f"autocorrelation times (target {target})")
    else:
        verdict = (f"NOT converged: spans {result['steps_per_tau']:.0f} "
                   f"autocorrelation times (target {target}); run at least "
                   f"{result['required_steps']} production steps")
    lines.append(f"  Verdict: {verdict}")
    return "\n".join(lines)


def write_convergence_summary(chain_dir, target=DEFAULT_TARGET):
    """Write ``convergence.txt`` for every chain in one ``saved_chains`` dir.

    Returns the :func:`chain_convergence` results, or an empty list when the
    directory holds no chains.
    """
    chain_dir = Path(chain_dir)
    paths = sorted(chain_dir.glob("chains_*.npz"))
    results = [chain_convergence(path, target) for path in paths]
    if not results:
        return results
    experiment = "/".join(chain_dir.parent.parts[-3:])
    failing = [result for result in results if not result["converged"]]
    header = [
        f"Convergence summary for {experiment}",
        f"Rule: every sampled parameter's chain must span at least {target} "
        "autocorrelation times (steps/tau).",
        f"Result: {len(results) - len(failing)} of {len(results)} fits "
        "converged" + (": " + ", ".join(result["model"] for result in failing)
                       + " did not" if failing else ""),
    ]
    text = "\n".join(header) + "\n\n" + "\n\n".join(
        format_convergence(result) for result in results
    ) + "\n"
    (chain_dir / CONVERGENCE_FILENAME).write_text(text, encoding="utf-8")
    return results


def convergence_warning(results):
    """Return a one-paragraph warning naming unconverged fits, or ``None``."""
    failing = [result for result in results if not result["converged"]]
    if not failing:
        return None
    target = failing[0]["target"]
    items = [
        f"{'/'.join(Path(result['path']).parent.parent.parts[-3:])} "
        f"({result['model']}, {result['steps_per_tau']:.0f} tau; needs "
        f">= {result['required_steps']} steps)"
        for result in failing
    ]
    return (f"WARNING: {len(failing)} of {len(results)} fits span fewer than "
            f"{target} autocorrelation times: " + "; \n".join(items) +
            f". See each experiment's saved_chains/{CONVERGENCE_FILENAME}.")


def summarize(results_dir, target=DEFAULT_TARGET):
    """Write convergence summaries for every experiment under ``results_dir``.

    Linked ``saved_chains`` folders (runs that reuse another run's fits) are
    skipped, since their chains are summarized with their source run.
    Returns all results and prints a warning naming unconverged fits.
    """
    results = []
    for chain_dir in sorted(Path(results_dir).glob("*/*/*/saved_chains")):
        if chain_dir.is_symlink():
            continue
        results.extend(write_convergence_summary(chain_dir, target))
    warning = convergence_warning(results)
    print(warning or f"All {len(results)} fits span at least {target} "
          "autocorrelation times.")
    return results
