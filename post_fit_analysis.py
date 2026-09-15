"""Collect publication-ready quantities from completed occurrence fits."""

import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import minimize_scalar

from occurrence import direct_fit_utils as dfu
from occurrence import direct_likelihood as dl
from occurrence import mcmc_direct
from occurrence import mcmc_powerlaw


SUMMARY_FILENAME = "summary_dict_direct_piecewise.npz"
DELTA_BIC_FILENAME = "delta_bics.json"

_TIER1_LATEX_PREFIXES = {
    "mtrue": "Mc",
    "msini": "Msini",
    "qtrue": "Q",
    "qsini": "Qsini",
}

_MODEL_PARAMETER_MACROS = {
    "logG": ("LogG", ("A", "Mu", "Sigma")),
    "escarpment": (
        "Escarpment", ("COne", "CTwo", "BPOne", "BPTwo")
    ),
    "sigmoid": ("Sigmoid", ("COne", "CTwo", "Center", "Width")),
}

_MODEL_PARAMETER_LABELS = {
    "logG": (r"$A$", r"$\mu$", r"$\sigma$"),
    "escarpment": (
        r"$C_1$", r"$C_2$", r"$\log_{10}(x_{t,1})$",
        r"$\log_{10}(x_{t,2})$",
    ),
    "sigmoid": (r"$C_1$", r"$C_2$", "Center", "Width"),
}


def _latex_token(value):
    """Turn a directory/type name into a legal, readable command token."""
    parts = re.findall(r"[A-Za-z]+|\d+", str(value))
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _tier1_prefix(tier1_dir):
    name = Path(tier1_dir).name
    return _TIER1_LATEX_PREFIXES.get(name, _latex_token(name))


def _tier2_directories(tier2_type):
    """Return ``(directory, macro suffix)`` pairs for a Tier 2 type."""
    if str(tier2_type).lower() == "allstars":
        return [(str(tier2_type), "")]
    return [
        (f"high{tier2_type}", "High"),
        (f"low{tier2_type}", "Low"),
    ]


def _scalar(summary, key, summary_path):
    if key not in summary:
        raise KeyError(f"{summary_path} does not contain {key!r}")
    value = np.asarray(summary[key])
    if value.size != 1:
        raise ValueError(f"{key!r} in {summary_path} must be a scalar")
    result = float(value.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"{key!r} in {summary_path} is not finite")
    return result


def _command(name, value):
    return rf"\newcommand{{\{name}}}{{\ensuremath{{{value}}}}}"


def _number_word(number):
    """Return a letter-only, zero-based bin label for a LaTeX command."""
    words = (
        "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven",
        "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen",
        "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen",
        "Nineteen",
    )
    if number >= len(words):
        raise ValueError("make_variables supports at most 20 stack bins")
    return words[number]


def _format_rounded(value, decimal_places):
    """Round a number and retain the requested display precision."""
    rounded = round(float(value), decimal_places)
    if rounded == 0:
        rounded = 0.0
    if decimal_places > 0:
        return f"{rounded:.{decimal_places}f}"
    return f"{rounded:.0f}"


def _format_parameter(mode, low, high):
    """Format a central value and asymmetric errors at the tighter precision."""
    low_error = float(mode) - float(low)
    high_error = float(high) - float(mode)
    if (not np.isfinite([mode, low_error, high_error]).all() or
            low_error <= 0 or high_error <= 0):
        raise ValueError("parameter mode and HDI must define two positive errors")
    tighter_error = min(low_error, high_error)
    decimal_places = max(0, -int(np.floor(np.log10(tighter_error))))
    return (
        f"{_format_rounded(mode, decimal_places)}"
        f"^{{+{_format_rounded(high_error, decimal_places)}}}"
        f"_{{-{_format_rounded(low_error, decimal_places)}}}"
    )


def _integrated_occurrence_samples(model_name, samples, model_bounds,
                                   stack_bounds):
    """Integrate posterior model draws exactly as the CDF plotting path does."""
    model_bounds = np.asarray(model_bounds, dtype=float)
    stack_bounds = np.asarray(stack_bounds, dtype=float)
    if (model_bounds.shape != (2,) or stack_bounds.shape != (2,) or
            np.any(model_bounds <= 0) or np.any(stack_bounds <= 0)):
        raise ValueError("model_bounds and stack_bounds must be positive pairs")
    grid = np.logspace(*np.log10(model_bounds), 300)
    log_grid = np.log10(grid)
    density_function = mcmc_powerlaw.get_model_spec(model_name).function
    stack_width = np.log10(stack_bounds[1]/stack_bounds[0])
    return np.asarray([
        np.trapz(density_function(sample, grid), log_grid)*stack_width
        for sample in samples
    ])


def _piecewise_integrated_values(chain_dir, prefix, split_suffix, stack_dim,
                                 n_stack_bins):
    """Collect per-stack integrated occurrence from the piecewise posterior."""
    path = chain_dir / "chains_direct_piecewise.npz"
    if not path.is_file():
        return []
    with np.load(path) as chain:
        if "flat_chains" not in chain or "x_edges" not in chain:
            raise KeyError(f"{path} must contain 'flat_chains' and 'x_edges'")
        y_key = "y_edges" if "y_edges" in chain else "stack_edges"
        if y_key not in chain:
            raise KeyError(f"{path} must contain 'y_edges' or 'stack_edges'")
        samples = np.asarray(chain["flat_chains"], dtype=float)[::10]
        a_edges = np.asarray(chain["x_edges"], dtype=float)
        m_edges = np.asarray(chain[y_key], dtype=float)
    n_a = len(a_edges) - 1
    n_m = len(m_edges) - 1
    if samples.ndim != 2 or samples.shape[1] != n_a*n_m:
        raise ValueError(f"piecewise chain dimensions do not match edges in {path}")
    expected_stack_bins = n_a if stack_dim == "a" else n_m
    if expected_stack_bins != n_stack_bins:
        raise ValueError(f"piecewise stack-bin count does not match summary in {path}")

    areas = np.outer(np.diff(np.log10(m_edges)), np.diff(np.log10(a_edges)))
    occurrence = samples.reshape(-1, n_m, n_a)*areas[None, :, :]
    sum_axis = 1 if stack_dim == "a" else 2
    integrated = occurrence.sum(axis=sum_axis)
    values = []
    for bin_index in range(n_stack_bins):
        low, median, high = np.percentile(
            integrated[:, bin_index], [16, 50, 84]
        )
        name = (
            prefix + "PiecewiseIntOcc" + split_suffix +
            f"Bin{stack_dim.lower()}{_number_word(bin_index)}"
        )
        values.append((name, _format_parameter(median, low, high)))
    return values


def _parametric_values(chain_dir, prefix, split_suffix, stack_dim,
                       n_stack_bins):
    """Collect formatted physical parameters from all saved smooth fits."""
    values = []
    dim = stack_dim.lower()
    for model_name, (model_macro, parameter_macros) in (
            _MODEL_PARAMETER_MACROS.items()):
        paths = [
            chain_dir / f"chains_direct_{model_name}_bin{index}.npz"
            for index in range(n_stack_bins)
        ]
        existing = [path.is_file() for path in paths]
        if not any(existing):
            continue
        if not all(existing):
            missing = [str(path) for path, exists in zip(paths, existing)
                       if not exists]
            raise FileNotFoundError(
                f"incomplete {model_name} stack-bin chains; missing {missing}"
            )

        model_values = []
        for bin_index, path in enumerate(paths):
            with np.load(path) as chain:
                key = "flat_chains" if "flat_chains" in chain else "chains"
                samples = np.asarray(chain[key], dtype=float)
                if "model_bounds" not in chain or "stack_bounds" not in chain:
                    raise KeyError(
                        f"{path} must contain 'model_bounds' and 'stack_bounds'"
                    )
                model_bounds = np.asarray(chain["model_bounds"], dtype=float)
                stack_bounds = np.asarray(chain["stack_bounds"], dtype=float)
            samples = samples.reshape(-1, samples.shape[-1])
            if samples.shape[1] != len(parameter_macros):
                raise ValueError(
                    f"parameter count in {path} does not match {model_name}"
                )
            if samples.shape[0] == 0 or not np.isfinite(samples).all():
                raise ValueError(f"parameter samples in {path} must be finite")
            low, median, high = np.percentile(
                samples, [16, 50, 84], axis=0
            )
            bin_label = _number_word(bin_index)
            for parameter_index, parameter_macro in enumerate(parameter_macros):
                name = (
                    prefix + model_macro + "Param" + parameter_macro +
                    split_suffix + f"Bin{dim}{bin_label}"
                )
                value = _format_parameter(
                    median[parameter_index],
                    low[parameter_index],
                    high[parameter_index],
                )
                model_values.append((name, value))
            cdf_samples = samples[::10]
            if cdf_samples.shape[0] > 10000:
                indices = np.linspace(
                    0, cdf_samples.shape[0] - 1, 10000, dtype=int
                )
                cdf_samples = cdf_samples[indices]
            integrated = _integrated_occurrence_samples(
                model_name, cdf_samples, model_bounds, stack_bounds
            )
            integrated_low, integrated_median, integrated_high = np.percentile(
                integrated, [16, 50, 84]
            )
            model_values.append((
                prefix + model_macro + "IntOcc" + split_suffix +
                f"Bin{dim}{bin_label}",
                _format_parameter(
                    integrated_median, integrated_low, integrated_high
                ),
            ))
        values.append((model_name, model_values))
    return values


def _maximum_likelihood_draw(path, model_name):
    """Return the physical chain draw with the greatest likelihood."""
    with np.load(path) as chain:
        if "flat_chains" not in chain or "flat_log_probs" not in chain:
            raise KeyError(
                f"{path} must contain 'flat_chains' and 'flat_log_probs'"
            )
        samples = np.asarray(chain["flat_chains"], dtype=float)
        log_probabilities = np.asarray(chain["flat_log_probs"], dtype=float)
    samples = samples.reshape(-1, samples.shape[-1])
    log_probabilities = log_probabilities.reshape(-1)
    if samples.shape[0] != log_probabilities.size:
        raise ValueError(f"chain and log-probability lengths differ in {path}")
    # Sampling is performed in transformed coordinates. Remove that
    # transformation's Jacobian to recover the physical-model likelihood.
    log_likelihoods = (
        log_probabilities -
        mcmc_direct._physical_log_jacobian(model_name, samples)
    )
    finite = np.isfinite(log_likelihoods)
    if not np.any(finite):
        raise ValueError(f"{path} contains no finite likelihood draws")
    index = np.flatnonzero(finite)[np.argmax(log_likelihoods[finite])]
    return samples[index]


def _optimize_flat_model(cache):
    """Optimize a positive constant occurrence-rate density."""
    effective_count = float(np.sum(cache.companion_weights))
    exposure = float(np.sum(cache.exposure_weights))
    if effective_count <= 0 or exposure <= 0:
        raise ValueError("flat-model optimization requires positive data and exposure")
    initial_log_amplitude = np.log(effective_count/exposure)

    def objective(log_amplitude):
        amplitude = np.exp(log_amplitude)
        return -dl.cached_smooth_log_likelihood(
            np.array([amplitude]), cache,
            lambda theta, x: np.full_like(x, theta[0], dtype=float),
        )

    result = minimize_scalar(
        objective,
        bounds=(initial_log_amplitude - 20.0, initial_log_amplitude + 20.0),
        method="bounded",
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"flat-model optimization failed: {result.message}")
    return np.exp(result.x), -float(result.fun)


def calculate_delta_bic(direct_fit_path, chain_dir, stack_dim="a"):
    """Compare every saved parametric fit with an optimized flat model.

    The comparison is performed separately in each stack bin.  The returned
    values use ``BIC_flat - BIC_model``; positive values favor the parametric
    model.  The sample size in the BIC penalty is the number of companion
    systems contributing posterior support to that stack bin.

    Returns
    -------
    dict
        Model names mapped to arrays of delta-BIC values in stack-bin order.
    """
    if stack_dim not in {"a", "m"}:
        raise ValueError("stack_dim must be 'a' or 'm'")
    direct_fit_path = Path(direct_fit_path)
    chain_dir = Path(chain_dir)
    companions, exposure = dfu.load_direct_fit_data(direct_fit_path)
    comparisons = {}

    for model_name, (_, parameter_names) in _MODEL_PARAMETER_MACROS.items():
        paths = list(chain_dir.glob(f"chains_direct_{model_name}_bin*.npz"))
        paths.sort(key=lambda path: int(path.stem.rsplit("bin", 1)[1]))
        if not paths:
            continue
        expected_indices = list(range(len(paths)))
        actual_indices = [int(path.stem.rsplit("bin", 1)[1]) for path in paths]
        if actual_indices != expected_indices:
            raise FileNotFoundError(
                f"non-contiguous {model_name} stack-bin chains: {actual_indices}"
            )

        model_deltas = []
        for path in paths:
            with np.load(path) as chain:
                if "stack_bounds" not in chain:
                    raise KeyError(f"{path} does not contain 'stack_bounds'")
                stack_bounds = tuple(np.asarray(chain["stack_bounds"], dtype=float))
            cache = dl.build_smooth_cache(
                companions, exposure, stack_dim, stack_bounds
            )
            _, flat_log_likelihood = _optimize_flat_model(cache)
            draw = _maximum_likelihood_draw(path, model_name)
            if draw.size != len(parameter_names):
                raise ValueError(f"parameter count in {path} does not match {model_name}")
            model_log_likelihood = dl.cached_smooth_log_likelihood(
                draw, cache, mcmc_powerlaw.get_model_spec(model_name).function
            )
            if not np.isfinite(model_log_likelihood):
                raise ValueError(f"maximum-likelihood draw in {path} is invalid")
            n_observations = len(cache.companion_names)
            if n_observations < 1:
                raise ValueError(f"no companion systems contribute to {path}")
            flat_bic = np.log(n_observations) - 2.0*flat_log_likelihood
            model_bic = (
                len(parameter_names)*np.log(n_observations) -
                2.0*model_log_likelihood
            )
            model_deltas.append(flat_bic - model_bic)
        comparisons[model_name] = np.asarray(model_deltas)
    return comparisons


def _result_key(tier1_dir, tier2_dir, tier3_dir):
    """Return the portable relative key used in the delta-BIC cache."""
    return Path(tier1_dir, tier2_dir, tier3_dir).as_posix()


def _table_tier2_parts(tier2_dir):
    """Convert a concrete Tier 2 directory to macro type and split suffix."""
    tier2_dir = str(tier2_dir)
    if tier2_dir.lower() == "allstars":
        return tier2_dir, ""
    for level, suffix in (("high", "High"), ("low", "Low")):
        if tier2_dir.startswith(level) and len(tier2_dir) > len(level):
            return tier2_dir[len(level):], suffix
    return tier2_dir, ""


def _variables_command_names(path):
    """Read the LaTeX command names defined in a variables file."""
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", text))


def make_parameter_table(
        results_dir, t1, t2, t3, models, stack_bin=0, stack_dim="a",
        caption="Derived Model Parameters", output_file=None):
    """Create a two-column LaTeX parameter table for one experiment.

    Every value is a command reference into ``results_dir/variables.tex``.
    ``stack_bin`` is zero-based and defaults to the first fitted stack bin.
    By default, the generated table is saved in ``latex_tables/`` beneath the
    current working directory.  ``output_file`` overrides that location.
    """
    results_dir = Path(results_dir)
    variables_path = results_dir / "variables.tex"
    if not variables_path.is_file():
        raise FileNotFoundError(f"variables file not found: {variables_path}")
    if not isinstance(stack_bin, (int, np.integer)) or stack_bin < 0:
        raise ValueError("stack_bin must be a nonnegative integer")
    if stack_dim not in {"a", "m"}:
        raise ValueError("stack_dim must be 'a' or 'm'")
    if isinstance(models, str):
        models = [models]
    else:
        models = list(models)
    aliases = {"escaprment": "escarpment"}
    models = [aliases.get(model, model) for model in models]
    if not models:
        raise ValueError("models cannot be empty")
    unknown = [model for model in models if model not in _MODEL_PARAMETER_MACROS]
    if unknown:
        raise ValueError(
            f"unknown models {unknown}; choose from "
            f"{list(_MODEL_PARAMETER_MACROS)}"
        )

    defined_commands = _variables_command_names(variables_path)
    tier2_type, split_suffix = _table_tier2_parts(t2)
    base_prefix = _tier1_prefix(t1) + tier2_type
    # Existing parameter variable names use forms such as "BinaZero".
    bin_suffix = "Bin" + stack_dim.lower() + _number_word(stack_bin)

    # A Tier 3 token is present only when make_variables was called with more
    # than one Tier 3 directory. Determine which convention the file uses.
    first_model = models[0]
    first_model_macro, first_parameters = _MODEL_PARAMETER_MACROS[first_model]
    probe_suffix = (
        first_model_macro + "Param" + first_parameters[0] +
        split_suffix + bin_suffix
    )
    prefixes = [base_prefix, base_prefix + _latex_token(Path(t3).name)]
    prefix = next(
        (candidate for candidate in prefixes
         if candidate + probe_suffix in defined_commands),
        None,
    )
    if prefix is None:
        raise KeyError(
            f"no matching commands for {t1}/{t2}/{t3} in {variables_path}"
        )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:model_params}",
        r"\begin{tabular}{lc}",
        r"\hline \hline",
        r"\textbf{Parameter} & \textbf{Value} \\",
        r"\hline",
    ]
    for model in models:
        model_macro, parameter_macros = _MODEL_PARAMETER_MACROS[model]
        labels = _MODEL_PARAMETER_LABELS[model]
        lines.append(
            rf"\multicolumn{{2}}{{l}}{{\textbf{{{model}}}}} \\"
        )
        lines.append(r"\hline")
        for label, parameter_macro in zip(labels, parameter_macros):
            command_name = (
                prefix + model_macro + "Param" + parameter_macro +
                split_suffix + bin_suffix
            )
            if command_name not in defined_commands:
                raise KeyError(
                    f"command \\{command_name} is not defined in {variables_path}"
                )
            lines.append(rf"{label} & \{command_name} \\")
        occurrence_name = (
            prefix + model_macro + "IntOcc" + split_suffix + bin_suffix
        )
        if occurrence_name not in defined_commands:
            raise KeyError(
                f"command \\{occurrence_name} is not defined in {variables_path}"
            )
        lines.append(rf"Occurrence & \{occurrence_name} \\")
        lines.append(r"\hline")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])

    if output_file is None:
        filename = f"model_params_{t1}_{t2}_{t3}.tex"
        output_path = Path.cwd() / "latex_tables" / filename
    else:
        output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def calculate_all_delta_bics(
        results_dir, tier1_dirs, tier2_types, tier3_dirs, stack_dim="a"):
    """Calculate and save delta-BIC values for every requested results folder.

    The arguments and Tier 2 expansion rules match :func:`make_variables`.
    Results are returned as a dictionary keyed by relative results paths and
    are also saved to ``results_dir/delta_bics.json``.  Folders containing no
    recognized parametric chains are represented by an empty dictionary.
    """
    results_dir = Path(results_dir)
    tier1_dirs = list(tier1_dirs)
    tier2_types = list(tier2_types)
    tier3_dirs = list(tier3_dirs)
    if not tier1_dirs or not tier2_types or not tier3_dirs:
        raise ValueError("tier1_dirs, tier2_types, and tier3_dirs cannot be empty")
    if stack_dim not in {"a", "m"}:
        raise ValueError("stack_dim must be 'a' or 'm'")

    all_delta_bics = {}
    for tier1_dir in tier1_dirs:
        for tier2_type in tier2_types:
            for tier2_dir, _ in _tier2_directories(tier2_type):
                for tier3_dir in tier3_dirs:
                    result_dir = results_dir / tier1_dir / tier2_dir / tier3_dir
                    chain_dir = result_dir / "saved_chains"
                    key = _result_key(tier1_dir, tier2_dir, tier3_dir)
                    has_parametric_chains = any(
                        next(chain_dir.glob(
                            f"chains_direct_{model_name}_bin*.npz"
                        ), None) is not None
                        for model_name in _MODEL_PARAMETER_MACROS
                    ) if chain_dir.is_dir() else False
                    if not has_parametric_chains:
                        all_delta_bics[key] = {}
                        continue
                    comparisons = calculate_delta_bic(
                        result_dir / "saved_dicts" / "direct_fit_data.npz",
                        chain_dir,
                        stack_dim,
                    )
                    all_delta_bics[key] = {
                        model_name: np.asarray(values, dtype=float).tolist()
                        for model_name, values in comparisons.items()
                    }

    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "stack_dim": stack_dim,
        "delta_bic_convention": "BIC_flat - BIC_model",
        "results": all_delta_bics,
    }
    (results_dir / DELTA_BIC_FILENAME).write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    return all_delta_bics


def _stack_statistics(summary, stack_dim, summary_path):
    """Calculate effective counts and area-weighted completeness by stack bin."""
    required = ("n_abins", "n_mbins", "cell_weights", "cell_compls",
                "a_m_lims_pairs")
    missing = [key for key in required if key not in summary]
    if missing:
        raise KeyError(f"{summary_path} does not contain {missing!r}")

    n_a = int(_scalar(summary, "n_abins", summary_path))
    n_m = int(_scalar(summary, "n_mbins", summary_path))
    expected_cells = n_a*n_m
    weights = np.asarray(summary["cell_weights"], dtype=float).reshape(-1)
    completeness = np.asarray(summary["cell_compls"], dtype=float).reshape(-1)
    pairs = np.asarray(summary["a_m_lims_pairs"], dtype=float)
    if weights.size != expected_cells or completeness.size != expected_cells:
        raise ValueError(
            f"bin arrays in {summary_path} do not match n_abins*n_mbins"
        )
    if pairs.shape != (expected_cells, 2, 2):
        raise ValueError(f"'a_m_lims_pairs' in {summary_path} has wrong shape")
    if not np.isfinite(completeness).all():
        raise ValueError(f"'cell_compls' in {summary_path} must be finite")

    # Cells are saved in mass-major order by _piecewise_metadata.
    weights = weights.reshape(n_m, n_a)
    completeness = completeness.reshape(n_m, n_a)
    log_widths = np.diff(np.log10(pairs), axis=2).reshape(expected_cells, 2)
    if not np.isfinite(log_widths).all() or np.any(log_widths <= 0):
        raise ValueError(f"bin limits in {summary_path} must be finite and positive")
    areas = np.prod(log_widths, axis=1).reshape(n_m, n_a)

    sum_axis = 0 if stack_dim == "a" else 1
    neff = weights.sum(axis=sum_axis)
    area = areas.sum(axis=sum_axis)
    avg_compl = (completeness*areas).sum(axis=sum_axis)/area
    return neff, avg_compl


def make_variables(
        results_dir, tier1_dirs, tier2_types, tier3_dirs, stack_dim="a"):
    """Write non-model fit statistics to a LaTeX variables file.

    ``results_dir`` is the parent of all Tier 1 directories and is also where
    ``variables.tex`` is written.  ``tier2_types`` contains unsplit names such
    as ``"Mass"`` and ``"FeH"``;
    each is expanded to its ``high`` and ``low`` directories.  ``"allstars"``
    is treated as a single directory.  A completed fit is expected at::

        results_dir/tier1/tier2/tier3/saved_dicts/
        summary_dict_direct_piecewise.npz

    The output contains ``Nstars``, ``Neff`` (the sum of the effective counts),
    and ``AvgCompl`` (the area-averaged completeness across the full ROI), plus
    ``Neff`` and ``AvgCompl`` for every bin along ``stack_dim``.  Stack-bin
    labels are zero-based, matching the legacy analysis (for example,
    ``NeffBinAZero``).
    With one Tier 3 directory, command names retain the legacy form, for
    example ``\\McallstarsNeff`` and ``\\McMassNeffHigh``.  When several Tier
    3 directories are supplied, their names are included to keep commands
    unique.

    Returns
    -------
    pathlib.Path
        Path to the generated file.
    """
    results_dir = Path(results_dir)
    tier1_dirs = list(tier1_dirs)
    tier2_types = list(tier2_types)
    tier3_dirs = list(tier3_dirs)
    if not tier1_dirs or not tier2_types or not tier3_dirs:
        raise ValueError("tier1_dirs, tier2_types, and tier3_dirs cannot be empty")
    if stack_dim not in {"a", "m"}:
        raise ValueError("stack_dim must be 'a' or 'm'")

    include_tier3 = len(tier3_dirs) > 1
    all_delta_bics = calculate_all_delta_bics(
        results_dir, tier1_dirs, tier2_types, tier3_dirs, stack_dim
    )
    blocks = []
    command_names = set()

    for tier1_dir in tier1_dirs:
        tier1_name = Path(tier1_dir).name
        tier1_path = results_dir / tier1_dir
        for tier2_type in tier2_types:
            for tier2_dir, split_suffix in _tier2_directories(tier2_type):
                for tier3_dir in tier3_dirs:
                    summary_path = (
                        tier1_path / tier2_dir / tier3_dir / "saved_dicts" /
                        SUMMARY_FILENAME
                    )
                    if not summary_path.is_file():
                        raise FileNotFoundError(
                            f"no direct piecewise summary found at {summary_path}"
                        )

                    with np.load(summary_path) as summary:
                        nstars = _scalar(summary, "nstars", summary_path)
                        if not nstars.is_integer():
                            raise ValueError(
                                f"'nstars' in {summary_path} must be an integer"
                            )
                        if "cell_weights" not in summary:
                            raise KeyError(
                                f"{summary_path} does not contain 'cell_weights'"
                            )
                        cell_weights = np.asarray(
                            summary["cell_weights"], dtype=float
                        )
                        if (cell_weights.size == 0 or
                                not np.isfinite(cell_weights).all()):
                            raise ValueError(
                                f"'cell_weights' in {summary_path} must be "
                                "nonempty and finite"
                            )
                        neff = float(cell_weights.sum())
                        avg_compl = _scalar(
                            summary, "cell_compl_single", summary_path
                        )
                        bin_neff, bin_avg_compl = _stack_statistics(
                            summary, stack_dim, summary_path
                        )
                        n_stack_bins = len(bin_neff)

                    prefix = _tier1_prefix(tier1_name) + str(tier2_type)
                    if include_tier3:
                        prefix += _latex_token(Path(tier3_dir).name)
                    values = {
                        "Nstars": str(int(nstars)),
                        f"Neff{split_suffix}": f"{neff:.1f}",
                        f"AvgCompl{split_suffix}": f"{avg_compl:.2f}",
                    }
                    if split_suffix:
                        values[f"Nstars{split_suffix}"] = values.pop("Nstars")
                    dim = stack_dim.upper()
                    for bin_index, (neff_value, compl_value) in enumerate(
                            zip(bin_neff, bin_avg_compl)):
                        bin_label = _number_word(bin_index)
                        values[
                            f"Neff{split_suffix}Bin{dim}{bin_label}"
                        ] = f"{neff_value:.1f}"
                        values[
                            f"AvgCompl{split_suffix}Bin{dim}{bin_label}"
                        ] = f"{compl_value:.2f}"

                    block = [
                        "%"*72,
                        f"% {tier1_name} / {tier2_dir} / {tier3_dir}",
                    ]
                    for statistic, value in values.items():
                        name = prefix + statistic
                        if name in command_names:
                            raise ValueError(f"duplicate LaTeX command name: {name}")
                        command_names.add(name)
                        block.append(_command(name, value))

                    chain_dir = (
                        tier1_path / tier2_dir / tier3_dir / "saved_chains"
                    )
                    piecewise_values = _piecewise_integrated_values(
                        chain_dir, prefix, split_suffix, stack_dim,
                        n_stack_bins,
                    )
                    if piecewise_values:
                        block.extend([
                            "", "%"*36, "% Integrated piecewise occurrence"
                        ])
                        for name, value in piecewise_values:
                            if name in command_names:
                                raise ValueError(
                                    f"duplicate LaTeX command name: {name}"
                                )
                            command_names.add(name)
                            block.append(_command(name, value))
                    parametric_groups = _parametric_values(
                            chain_dir, prefix, split_suffix, stack_dim,
                            n_stack_bins)
                    for model_name, model_values in parametric_groups:
                        block.extend(["", "%"*36, f"% Parametric model: {model_name}"])
                        for name, value in model_values:
                            if name in command_names:
                                raise ValueError(
                                    f"duplicate LaTeX command name: {name}"
                                )
                            command_names.add(name)
                            block.append(_command(name, value))
                    if parametric_groups:
                        key = _result_key(tier1_dir, tier2_dir, tier3_dir)
                        delta_bics = all_delta_bics[key]
                        block.extend(["", "%"*36, "% BIC comparisons"])
                        for model_name, deltas in delta_bics.items():
                            model_macro = _MODEL_PARAMETER_MACROS[model_name][0]
                            for bin_index, delta in enumerate(deltas):
                                name = (
                                    prefix + model_macro + "Dbic" + split_suffix +
                                    f"Bin{stack_dim.lower()}{_number_word(bin_index)}"
                                )
                                if name in command_names:
                                    raise ValueError(
                                        f"duplicate LaTeX command name: {name}"
                                    )
                                command_names.add(name)
                                block.append(_command(name, f"{delta:.1f}"))
                    blocks.append("\n".join(block))

    output_path = results_dir / "variables.tex"
    results_dir.mkdir(parents=True, exist_ok=True)
    contents = "% Auto-generated by post_fit_analysis.make_variables\n\n"
    contents += "\n\n".join(blocks) + "\n"
    output_path.write_text(contents, encoding="utf-8")
    return output_path
