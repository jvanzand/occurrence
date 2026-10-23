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
    "loglinear": ["ln C1", "ln C2"],
    "escarpment": ["ln D1", "ln D2", "log10 x_t1", "log10 x_t2"],
    "bpl": ["ln C", "log10 x_break", "beta", "gamma"],
    "beta": ["ln A", "ln alpha", "ln beta"],
    "rayleigh": ["ln A", "ln sigma_e"],
    "ecc_mixture": ["ln A", "f1", "ln sigma1", "ln sigma2"],
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


def _experiment(result):
    """Return ``tier1/tier2/tier3`` for a chain's experiment folder."""
    return "/".join(Path(result["path"]).parent.parent.parts[-3:])


def _chain_label(result):
    """Return the model name, with the stack bin for binned smooth fits."""
    stem = Path(result["path"]).stem[len("chains_"):]
    return stem.replace("_bin", " bin ") if "_bin" in stem else stem


def _slowest_parameter(result):
    return min(result["parameters"], key=lambda row: row[2])


def format_duration(seconds):
    """Return a short duration such as ``45 s``, ``12.3 min``, or ``2.1 h``."""
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    return f"{seconds/3600:.1f} h"


def convergence_table(results):
    """Return a table with one row per fit, naming its slowest parameter.

    Columns give the production steps, the largest autocorrelation time, the
    chain length in autocorrelation times, the slowest parameter, the steps
    needed to reach the target (blank when converged), the fit time when the
    result records ``fit_seconds``, and the verdict.
    """
    if not results:
        return ""
    target = results[0]["target"]
    rows = []
    for result in results:
        slowest = _slowest_parameter(result)
        rows.append((
            _experiment(result), _chain_label(result),
            str(result["n_steps"]), f"{slowest[1]:.0f}",
            f"{result['steps_per_tau']:.0f}", slowest[0],
            "" if result["converged"] else str(result["required_steps"]),
            format_duration(result.get("fit_seconds")),
            "ok" if result["converged"] else "SHORT",
        ))
    header = ("experiment", "model", "steps", "tau_max", "N/tau",
              "slowest parameter", "needs", "time", "status")
    widths = [max(len(row[index]) for row in rows + [header])
              for index in range(len(header))]
    right = {2, 3, 4, 6, 7}

    def line(row):
        return "  ".join(
            cell.rjust(width) if index in right else cell.ljust(width)
            for index, (cell, width) in enumerate(zip(row, widths))
        ).rstrip()

    # Name each experiment only on its first row
    shown = [
        ("",) + row[1:] if index and row[0] == rows[index - 1][0] else row
        for index, row in enumerate(rows)
    ]
    return "\n".join(
        [f"Convergence (target: {target} autocorrelation times)", line(header)]
        + [line(row) for row in shown]
    )


def step_recommendations(results):
    """Return one line per run and step setting that needs longer chains.

    Fits are grouped by their run (the Tier 3 folder) and by the setting
    that controls their length: ``piecewise_nsteps`` for the histogram and
    ``nsteps`` for the smooth models. Each line gives the setting that would
    bring every short fit in the group to the target.
    """
    groups = {}
    for result in results:
        run_name = Path(result["path"]).parent.parent.name
        setting = ("piecewise_nsteps" if result["model"] == "piecewise"
                   else "nsteps")
        groups.setdefault((run_name, setting), []).append(result)
    lines = []
    for (run_name, setting), group in groups.items():
        short = [result for result in group if not result["converged"]]
        if not short:
            continue
        kind = ("histogram" if setting == "piecewise_nsteps"
                else "smooth-model")
        needed = max(result["required_steps"] for result in short)
        lines.append(
            f"{run_name}: {len(short)} of {len(group)} {kind} fits span "
            f"fewer than {short[0]['target']} autocorrelation times; set "
            f"{setting} >= {needed}."
        )
    return lines


def report(results):
    """Return the convergence table followed by any step recommendations."""
    if not results:
        return ""
    lines = [convergence_table(results)]
    recommendations = step_recommendations(results)
    lines.append("")
    lines.extend(recommendations or [
        f"All {len(results)} fits span at least {results[0]['target']} "
        "autocorrelation times."
    ])
    return "\n".join(lines)


def summarize(results_dir, target=DEFAULT_TARGET):
    """Write convergence summaries for every experiment under ``results_dir``.

    Linked ``saved_chains`` folders (runs that reuse another run's fits) are
    skipped, since their chains are summarized with their source run.
    Returns all results and prints their convergence table.
    """
    results = []
    for chain_dir in sorted(Path(results_dir).glob("*/*/*/saved_chains")):
        if chain_dir.is_symlink():
            continue
        results.extend(write_convergence_summary(chain_dir, target))
    print(report(results) or "No fits found.")
    return results
