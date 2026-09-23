"""Collect publication-ready quantities from completed occurrence fits."""

import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from occurrence import fit_utils as dfu
from occurrence import likelihood as dl
from occurrence import mcmc
from occurrence import mcmc_powerlaw


SUMMARY_FILENAME = "summary_dict_piecewise.npz"
DELTA_BIC_FILENAME = "delta_bics.json"
PAPER_ITEMS_DIRNAME = "paper_items"

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
    "bpl": ("BPL", ("C", "LogBreak", "Beta", "Gamma")),
    "loglinear": ("LogLinear", ("CLow", "CHigh")),
}

_MODEL_PARAMETER_LABELS = {
    "logG": (r"$A$", r"$\mu$", r"$\sigma$"),
    "escarpment": (
        r"$C_1$", r"$C_2$", r"$\log_{10}(x_{t,1})$",
        r"$\log_{10}(x_{t,2})$",
    ),
    "sigmoid": (
        r"$C_1$", r"$C_2$", r"$\log_{10}(x_t)$", r"$W$"
    ),
    "bpl": (r"$C$", r"$\log_{10}(x_0)$", r"$\beta$", r"$\gamma$"),
    "loglinear": (r"$C_{\rm low}$", r"$C_{\rm high}$"),
}

_APPENDIX_MODEL_ORDER = (
    "sigmoid", "escarpment", "logG", "loglinear",
)

_THREE_PARAMETER_TIER2_DIRS = (
    "highMstarhighFeHhighAct", "highMstarhighFeHlowAct",
    "highMstarlowFeHhighAct", "highMstarlowFeHlowAct",
    "lowMstarhighFeHhighAct", "lowMstarhighFeHlowAct",
    "lowMstarlowFeHhighAct", "lowMstarlowFeHlowAct",
)

_THREE_PARAMETER_DEFAULT_CUTS = {
    "Mstar": 1.0,
    "feh": 0.0,
    "age": 5.0,
}


def _latex_token(value):
    """Turn a directory/type name into a legal, readable command token."""
    parts = re.findall(r"[A-Za-z]+|\d+", str(value))
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _tier1_prefix(tier1_dir):
    name = Path(tier1_dir).name
    return _TIER1_LATEX_PREFIXES.get(name, _latex_token(name))


def _tier2_command_token(tier2_dir):
    """Return a command token that directly mirrors a Tier 2 directory."""
    name = Path(tier2_dir).name
    return "allstars" if name.lower() == "allstars" else _latex_token(name)


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


def _plot_companions_for_result(
        results_path, stellar_parameter, output_file=None,
        catalog_path=None, tier1_name=None):
    """Plot one result's median companion mass and SMA by host property.

    ``results_path`` may be a Tier 3 experiment directory, its
    ``saved_dicts`` directory, or the corresponding ``fit_data.npz`` file.
    The included hosts are inferred from the saved companion names.  By
    default the CLS companion catalog is loaded from this repository's
    ``cls_files`` directory, and the plot is written beneath the experiment's
    ``plots`` directory.
    """
    from matplotlib import pyplot as plt
    import pandas as pd
    from occurrence import plotting_utils as pu

    plt.style.use(str(Path(__file__).resolve().parent / "matplotlibrc"))

    parameter_configs = {
        "mstar": ("Mstar", r"$M_{\star}$ [$M_{\odot}$]", "Blues"),
        "mass": ("Mstar", r"$M_{\star}$ [$M_{\odot}$]", "Blues"),
        "feh": ("feh", "[Fe/H]", "seismic"),
        "metallicity": ("feh", "[Fe/H]", "seismic"),
        "age": ("age", "Age [Gyr]", "Greens"),
    }
    parameter_key = re.sub(r"[^a-z]", "", str(stellar_parameter).lower())
    try:
        column, colorbar_label, colormap = parameter_configs[parameter_key]
    except KeyError:
        raise ValueError(
            "stellar_parameter must be one of 'Mstar', 'FeH', or 'Age'"
        )

    supplied_path = Path(results_path)
    candidates = []
    if supplied_path.is_file():
        candidates.append(supplied_path)
    else:
        candidates.extend([
            supplied_path / "saved_dicts" / "fit_data.npz",
            supplied_path / "fit_data.npz",
        ])
        if supplied_path.name == "saved_chains":
            candidates.append(
                supplied_path.parent / "saved_dicts" / "fit_data.npz"
            )
    fit_path = next((path for path in candidates if path.is_file()), None)
    if fit_path is None:
        raise FileNotFoundError(
            f"could not find saved_dicts/fit_data.npz beneath {supplied_path}"
        )

    with np.load(fit_path, allow_pickle=False) as saved:
        if "companion_names" not in saved:
            raise KeyError(f"{fit_path} does not contain 'companion_names'")
        companion_names = [str(name) for name in saved["companion_names"]]
        x_bounds = (
            tuple(np.asarray(saved["x_bounds"], dtype=float))
            if "x_bounds" in saved else None
        )
        y_bounds = (
            tuple(np.asarray(saved["y_bounds"], dtype=float))
            if "y_bounds" in saved else None
        )
    if not companion_names:
        raise ValueError(f"{fit_path} contains no companions")

    host_names = {
        name.rsplit("_", 1)[0].strip().lower() for name in companion_names
    }
    catalog_path = (
        Path(__file__).resolve().parent / "cls_files" /
        "cls_all_comps_with_stellar_params.csv"
        if catalog_path is None else Path(catalog_path)
    )
    catalog = pd.read_csv(catalog_path)
    required = {
        "cps_identifier", "Mtrue_post_med", "a_post_med",
        "Msini_pre_med", "a_pre_med", column,
    }
    if parameter_key == "age":
        required.add("Mstar")
    missing_columns = sorted(required - set(catalog.columns))
    if missing_columns:
        raise KeyError(f"{catalog_path} lacks columns {missing_columns}")

    catalog_hosts = catalog["cps_identifier"].astype(str).str.strip().str.lower()
    selected = catalog.loc[catalog_hosts.isin(host_names)].copy()
    if selected.empty:
        raise ValueError(
            f"no catalog companions match the hosts saved in {fit_path}"
        )
    numeric_columns = {
        "a_post_med", "a_pre_med", "Mtrue_post_med", "Msini_pre_med",
        column,
    }
    if parameter_key == "age":
        numeric_columns.add("Mstar")
    for value_column in numeric_columns:
        selected[value_column] = pd.to_numeric(
            selected[value_column], errors="coerce"
        )
    selected["a_post_med"] = selected["a_post_med"].fillna(
        selected["a_pre_med"]
    )
    selected["Mtrue_post_med"] = selected["Mtrue_post_med"].fillna(
        selected["Msini_pre_med"]
    )
    valid_coordinates = (
        np.isfinite(selected["a_post_med"]) &
        np.isfinite(selected["Mtrue_post_med"]) &
        (selected["a_post_med"] > 0) &
        (selected["Mtrue_post_med"] > 0)
    )
    if not valid_coordinates.all():
        invalid_names = selected.loc[
            ~valid_coordinates,
            "CPS Name" if "CPS Name" in selected else "cps_identifier"
        ].astype(str).tolist()
        raise ValueError(
            "selected companions have missing or nonpositive plotting values: "
            f"{invalid_names}"
        )
    if parameter_key != "age" and not np.isfinite(selected[column]).all():
        invalid_names = selected.loc[
            ~np.isfinite(selected[column]),
            "CPS Name" if "CPS Name" in selected else "cps_identifier"
        ].astype(str).tolist()
        raise ValueError(
            f"selected companions have missing {column} values: "
            f"{invalid_names}"
        )

    experiment_dir = (
        fit_path.parent.parent if fit_path.parent.name == "saved_dicts"
        else fit_path.parent
    )
    if tier1_name is None:
        tier1_name = experiment_dir.parent.parent.name
    tier1_name = Path(tier1_name).name
    if tier1_name != "mtrue":
        raise ValueError(
            "Mtrue_post_med points can only be overlaid consistently on an "
            "mtrue completeness map"
        )
    average_map_dir = experiment_dir.parent / "avg_map"
    grid_paths = {
        name: average_map_dir / f"parent_{name}grid.npy"
        for name in ("x", "y", "z")
    }
    missing_grids = [str(path) for path in grid_paths.values()
                     if not path.is_file()]
    if missing_grids:
        raise FileNotFoundError(
            f"average completeness grids are missing: {missing_grids}"
        )
    xgrid = np.load(grid_paths["x"])
    ygrid = np.load(grid_paths["y"])
    zgrid = np.load(grid_paths["z"])
    roi_pairs = (
        [(x_bounds, y_bounds)]
        if x_bounds is not None and y_bounds is not None else None
    )
    figure = pu.completeness_plotter(
        xgrid, ygrid, zgrid, save_path="", title="",
        save_plot=False, a_m_lims_pairs=roi_pairs, zoom=True,
        ycol="inj_mtrue", m_unit="jupiter",
    )
    axis = figure.axes[0]
    completeness_colorbar_axis = figure.axes[1]
    axis.xaxis.label.set_size(24)
    axis.yaxis.label.set_size(24)
    axis.tick_params(axis="both", which="both", labelsize=22)
    completeness_colorbar_axis.yaxis.label.set_size(20)
    completeness_colorbar_axis.tick_params(labelsize=20)
    if parameter_key == "age":
        from matplotlib import colors
        from matplotlib.cm import ScalarMappable

        in_mass_range = (
            np.isfinite(selected["Mstar"]) &
            (selected["Mstar"] >= 0.82) &
            (selected["Mstar"] <= 1.21)
        )
        has_age = np.isfinite(selected["age"])
        colored = in_mass_range & has_age
        missing_age = in_mass_range & ~has_age
        outside_mass_range = ~in_mass_range

        finite_ages = selected.loc[has_age, "age"].to_numpy(dtype=float)
        if finite_ages.size:
            age_min = float(np.min(finite_ages))
            age_max = float(np.max(finite_ages))
            if age_min == age_max:
                age_min -= 0.5
                age_max += 0.5
        else:
            age_min, age_max = 0.0, 1.0
        age_norm = colors.Normalize(vmin=age_min, vmax=age_max)
        if colored.any():
            points = axis.scatter(
                selected.loc[colored, "a_post_med"],
                selected.loc[colored, "Mtrue_post_med"],
                c=selected.loc[colored, "age"], cmap=colormap, norm=age_norm,
                edgecolor="black", s=65, zorder=110,
            )
        else:
            points = ScalarMappable(norm=age_norm, cmap=colormap)
            points.set_array([])
        if missing_age.any():
            axis.scatter(
                selected.loc[missing_age, "a_post_med"],
                selected.loc[missing_age, "Mtrue_post_med"],
                color="gray", edgecolor="black", s=65, zorder=110,
            )
        if outside_mass_range.any():
            axis.scatter(
                selected.loc[outside_mass_range, "a_post_med"],
                selected.loc[outside_mass_range, "Mtrue_post_med"],
                color="black", marker="x", s=65, linewidths=1.5, zorder=111,
            )
    else:
        points = axis.scatter(
            selected["a_post_med"], selected["Mtrue_post_med"],
            c=selected[column], cmap=colormap, edgecolor="black", s=65,
            zorder=110,
        )
    # Match the catalog/completeness layout: completeness colorbar at right,
    # stellar-property colorbar above the main panel.
    axis.set_title("")
    figure.subplots_adjust(top=0.84, left=0.14, right=0.98, bottom=0.14)
    bounds = axis.get_position()
    colorbar_axis = figure.add_axes([
        bounds.x0, bounds.y1, bounds.width, 0.03
    ])
    colorbar = figure.colorbar(
        points, cax=colorbar_axis, orientation="horizontal"
    )
    colorbar.set_label(colorbar_label)
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.xaxis.set_label_position("top")
    if output_file is None:
        output_path = (
            experiment_dir / "plots" /
            f"companions_by_{column.lower()}.png"
        )
    else:
        output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def plot_companions_by_stellar_parameter(
        results_dir, tier1_dirs, tier2_types, tier3_dirs,
        stellar_parameters, catalog_path=None):
    """Create host-colored companion plots for requested result folders.

    The directory arguments follow :func:`make_variables`.  Tier 2 types such
    as ``"Mstar"`` are expanded to their ``highMstar`` and ``lowMstar``
    directories, while ``"allstars"`` remains a single directory.  Every
    requested stellar parameter gets its own plot in the matching Tier 3
    experiment's ``plots`` directory.  A single parameter string is also
    accepted for convenience.

    Returns
    -------
    dict
        Relative ``tier1/tier2/tier3`` keys mapped to generated plot paths.
    """
    results_dir = Path(results_dir)
    tier1_dirs = list(tier1_dirs)
    tier2_types = list(tier2_types)
    tier3_dirs = list(tier3_dirs)
    if isinstance(stellar_parameters, str):
        stellar_parameters = [stellar_parameters]
    else:
        stellar_parameters = list(stellar_parameters)
    if (not tier1_dirs or not tier2_types or not tier3_dirs or
            not stellar_parameters):
        raise ValueError(
            "tier1_dirs, tier2_types, tier3_dirs, and stellar_parameters "
            "cannot be empty"
        )

    outputs = {}
    for tier1_dir in tier1_dirs:
        for tier2_type in tier2_types:
            for tier2_dir, _ in _tier2_directories(tier2_type):
                for tier3_dir in tier3_dirs:
                    experiment_dir = (
                        results_dir / tier1_dir / tier2_dir / tier3_dir
                    )
                    result_key = _result_key(
                        tier1_dir, tier2_dir, tier3_dir
                    )
                    for stellar_parameter in stellar_parameters:
                        key = f"{result_key}/{stellar_parameter}"
                        outputs[key] = _plot_companions_for_result(
                            experiment_dir,
                            stellar_parameter=stellar_parameter,
                            catalog_path=catalog_path,
                            tier1_name=Path(tier1_dir).name,
                        )
    return outputs


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
    stack_width = np.log10(stack_bounds[1]/stack_bounds[0])
    return np.asarray([
        np.trapz(
            mcmc_powerlaw.evaluate_density(
                model_name, sample, grid, model_bounds
            ),
            log_grid,
        )*stack_width
        for sample in samples
    ])


def _piecewise_integrated_values(chain_dir, prefix, stack_dim, n_stack_bins):
    """Collect per-stack integrated occurrence from the piecewise posterior."""
    path = chain_dir / "chains_piecewise.npz"
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
            prefix + "PiecewiseIntOcc" +
            f"Bin{stack_dim.lower()}{_number_word(bin_index)}"
        )
        values.append((name, _format_parameter(median, low, high)))
    return values


def _parametric_values(chain_dir, prefix, stack_dim, n_stack_bins):
    """Collect formatted physical parameters from all saved smooth fits."""
    values = []
    dim = stack_dim.lower()
    for model_name, (model_macro, parameter_macros) in (
            _MODEL_PARAMETER_MACROS.items()):
        paths = [
            chain_dir / f"chains_{model_name}_bin{index}.npz"
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
                    f"Bin{dim}{bin_label}"
                )
                value = _format_parameter(
                    median[parameter_index],
                    low[parameter_index],
                    high[parameter_index],
                )
                model_values.append((name, value))
            if model_name in {"escarpment", "loglinear"}:
                if model_name == "escarpment":
                    denominator = samples[:, 3] - samples[:, 2]
                else:
                    denominator = np.full(
                        samples.shape[0],
                        np.log10(model_bounds[1]/model_bounds[0]),
                    )
                if (not np.isfinite(denominator).all() or
                        np.any(denominator <= 0)):
                    raise ValueError(
                        f"{model_name} slope denominator must be positive in "
                        f"{path}"
                    )
                slope_samples = (samples[:, 1] - samples[:, 0])/denominator
                slope_low, slope_median, slope_high = np.percentile(
                    slope_samples, [16, 50, 84]
                )
                model_values.append((
                    prefix + model_macro + "ParamSlope" +
                    f"Bin{dim}{bin_label}",
                    _format_parameter(
                        slope_median, slope_low, slope_high
                    ),
                ))
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
                prefix + model_macro + "IntOcc" +
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
        mcmc._physical_log_jacobian(model_name, samples)
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


def calculate_delta_bic(fit_path, chain_dir, stack_dim="a"):
    """Compare every saved parametric fit with an optimized flat model.

    The comparison is performed separately in each stack bin and over the
    model-coordinate interval used for the fit.  The returned values use
    ``BIC_flat - BIC_model``; positive values favor the parametric model.  The
    sample size in the BIC penalty is the number of companion systems
    contributing posterior support to that fitted interval.

    Returns
    -------
    dict
        Model names mapped to arrays of delta-BIC values in stack-bin order.
    """
    if stack_dim not in {"a", "m"}:
        raise ValueError("stack_dim must be 'a' or 'm'")
    fit_path = Path(fit_path)
    chain_dir = Path(chain_dir)
    companions, exposure = dfu.load_fit_data(fit_path)
    comparisons = {}

    for model_name, (_, parameter_names) in _MODEL_PARAMETER_MACROS.items():
        paths = list(chain_dir.glob(f"chains_{model_name}_bin*.npz"))
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
                if "stack_bounds" not in chain or "model_bounds" not in chain:
                    raise KeyError(
                        f"{path} must contain stack_bounds and model_bounds"
                    )
                stack_bounds = tuple(np.asarray(chain["stack_bounds"], dtype=float))
                model_bounds = tuple(np.asarray(chain["model_bounds"], dtype=float))
            cache = dl.build_smooth_cache(
                companions, exposure, stack_dim, stack_bounds,
                model_bounds=model_bounds,
            )
            _, flat_log_likelihood = _optimize_flat_model(cache)
            draw = _maximum_likelihood_draw(path, model_name)
            if draw.size != len(parameter_names):
                raise ValueError(f"parameter count in {path} does not match {model_name}")
            density_function = lambda theta, x: mcmc_powerlaw.evaluate_density(
                model_name, theta, x, model_bounds
            )
            model_log_likelihood = dl.cached_smooth_log_likelihood(
                draw, cache, density_function
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


def _variables_command_names(path):
    """Read the LaTeX command names defined in a variables file."""
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", text))


def make_parameter_table(
        results_dir, t1, t2, t3, models, stack_bin=0, stack_dim="a",
        caption="Derived Model Parameters", output_file=None):
    """Create a two-column LaTeX parameter table for one experiment.

    Every value is a command reference into
    ``results_dir/paper_items/variables.tex``.
    ``stack_bin`` is zero-based and defaults to the first fitted stack bin.
    By default, the generated table is saved in ``results_dir/paper_items/``.
    ``output_file`` overrides that location.
    """
    results_dir = Path(results_dir)
    variables_path = results_dir / PAPER_ITEMS_DIRNAME / "variables.tex"
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
    base_prefix = _tier1_prefix(t1) + _tier2_command_token(t2)
    # Existing parameter variable names use forms such as "BinaZero".
    bin_suffix = "Bin" + stack_dim.lower() + _number_word(stack_bin)

    # A Tier 3 token is present only when make_variables was called with more
    # than one Tier 3 directory. Determine which convention the file uses.
    first_model = models[0]
    first_model_macro, first_parameters = _MODEL_PARAMETER_MACROS[first_model]
    probe_suffix = (
        first_model_macro + "Param" + first_parameters[0] +
        bin_suffix
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
                bin_suffix
            )
            if command_name not in defined_commands:
                raise KeyError(
                    f"command \\{command_name} is not defined in {variables_path}"
                )
            lines.append(rf"{label} & \{command_name} \\")
        occurrence_name = (
            prefix + model_macro + "IntOcc" + bin_suffix
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
        output_path = results_dir / PAPER_ITEMS_DIRNAME / filename
    else:
        output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def make_appendix_parameter_table(
        results_dir, tier1_dirs, tier2_types, t3, stack_bin=0,
        stack_dim="a", caption="Fitted Model Parameters",
        label="tab:model_params_appendix", output_file=None,
        include_tier3_in_commands=False):
    """Create a hierarchical appendix table for all smooth fitted models.

    Each experiment contains one row for every smooth model whose requested
    stack-bin chain exists.  Experiment-level cells are populated on the first
    row and left blank on the remaining model rows.  Rows follow the legacy
    experiment order: Tier 1 values in the supplied order, then Tier 2 types
    in the supplied order, with ``allstars`` represented once and every other
    type represented by its high row followed by its low row.  Numerical cells
    reference commands generated by :func:`make_variables`; this function
    neither reads nor validates ``variables.tex``.

    Set ``include_tier3_in_commands`` when ``make_variables`` was called with
    multiple Tier 3 directories and therefore included the Tier 3 token in its
    command prefixes.
    """
    results_dir = Path(results_dir)
    tier1_dirs = list(tier1_dirs)
    tier2_types = list(tier2_types)
    if not tier1_dirs or not tier2_types:
        raise ValueError("tier1_dirs and tier2_types cannot be empty")
    if stack_dim not in {"a", "m"}:
        raise ValueError("stack_dim must be 'a' or 'm'")
    if not isinstance(stack_bin, (int, np.integer)) or stack_bin < 0:
        raise ValueError("stack_bin must be a nonnegative integer")

    bin_word = _number_word(stack_bin)
    parameter_bin_suffix = f"Bin{stack_dim.lower()}{bin_word}"
    statistic_bin_suffix = f"Bin{stack_dim.upper()}{bin_word}"

    lines = [
        r"\begin{longrotatetable}",
        r"\begin{deluxetable*}{cccccccccccc}",
        rf"\tablecaption{{{caption}}}",
        rf"\label{{{label}}}",
        "",
        r"\tablehead{",
        r"\shortstack{Mass\\Parameter} &",
        r"\shortstack{Stellar\\Subsample} &",
        r"$N_\star$ &",
        r"\shortstack{$N_{\mathrm{eff}}$} &",
        r"Completeness &",
        r"Model &",
        r"\shortstack{Integrated\\Occurrence Rate} &",
        r"$\theta_1$ & $\theta_2$ & $\theta_3$ & $\theta_4$ &",
        r"$\Delta\mathrm{BIC}$",
        r"}",
        r"\startdata",
    ]

    for tier1_dir in tier1_dirs:
        tier1_name = Path(tier1_dir).name
        for tier2_type in tier2_types:
            for tier2_dir, _ in _tier2_directories(tier2_type):
                prefix = (
                    _tier1_prefix(tier1_name) +
                    _tier2_command_token(tier2_dir)
                )
                if include_tier3_in_commands:
                    prefix += _latex_token(Path(t3).name)
                experiment_cells = [
                    tier1_name,
                    tier2_dir,
                    f"\\{prefix}Nstars",
                    f"\\{prefix}Neff{statistic_bin_suffix}",
                    f"\\{prefix}AvgCompl{statistic_bin_suffix}",
                ]
                chain_dir = (
                    results_dir / tier1_dir / tier2_dir / t3 / "saved_chains"
                )
                calculated_models = [
                    model_name for model_name in _APPENDIX_MODEL_ORDER
                    if (chain_dir /
                        f"chains_{model_name}_bin{stack_bin}.npz").is_file()
                ]
                for model_index, model_name in enumerate(calculated_models):
                    model_macro, parameter_macros = _MODEL_PARAMETER_MACROS[
                        model_name
                    ]
                    parameter_labels = _MODEL_PARAMETER_LABELS[model_name]
                    parameter_cells = []
                    for parameter_label, parameter_macro in zip(
                            parameter_labels, parameter_macros):
                        command = (
                            prefix + model_macro + "Param" + parameter_macro +
                            parameter_bin_suffix
                        )
                        parameter_cells.append(
                            rf"\shortstack{{{parameter_label}\\\{command}}}"
                        )
                    parameter_cells.extend(
                        [r"\nodata"]*(4 - len(parameter_cells))
                    )
                    occurrence = (
                        prefix + model_macro + "IntOcc" +
                        parameter_bin_suffix
                    )
                    delta_bic = (
                        prefix + model_macro + "Dbic" +
                        parameter_bin_suffix
                    )
                    row = (
                        experiment_cells if model_index == 0 else ["", "", "", "", ""]
                    ) + [
                        rf"\textbf{{{model_name}}}",
                        f"\\{occurrence}",
                        *parameter_cells,
                        f"\\{delta_bic}",
                    ]
                    lines.append(" & ".join(row) + r" \\")
                if calculated_models:
                    lines.append(r"\hline")

    lines.extend([
        r"\enddata",
        r"\end{deluxetable*}",
        r"\end{longrotatetable}",
    ])

    if output_file is None:
        output_path = (
            results_dir / PAPER_ITEMS_DIRNAME /
            f"model_params_appendix_{t3}.tex"
        )
    else:
        output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _three_parameter_levels(tier2_dir):
    """Extract mass, metallicity, and age levels from a subset directory."""
    match = re.fullmatch(
        r"(high|low)Mstar(high|low)FeH(high|low)Act", str(tier2_dir)
    )
    if match is None:
        raise ValueError(
            f"invalid three-parameter Tier 2 directory: {tier2_dir!r}"
        )
    mass, metallicity, activity = match.groups()
    age = "young" if activity == "high" else "old"
    return mass, metallicity, age


def _three_parameter_command_name(t1, t3, levels, statistic="IntOcc"):
    """Return a shared variable-command name for a stellar subset."""
    mass, metallicity, age = levels
    prefix = _tier1_prefix(t1)
    if str(t3) != "stellar3params":
        prefix += _latex_token(Path(t3).name)
    return (
        prefix + mass.capitalize() + "Mstar" +
        metallicity.capitalize() + "FeH" + age.capitalize() + statistic
    )


def _three_parameter_comparisons():
    """Return the twelve comparisons shown in the reordered table."""
    comparisons = []
    for metallicity in ("high", "low"):
        for age in ("old", "young"):
            comparisons.append((
                "Mass", (metallicity, age),
                ("low", metallicity, age),
                ("high", metallicity, age),
            ))
    for mass in ("high", "low"):
        for age in ("old", "young"):
            comparisons.append((
                "FeH", (mass, age),
                (mass, "low", age),
                (mass, "high", age),
            ))
    for mass in ("high", "low"):
        for metallicity in ("high", "low"):
            comparisons.append((
                "Age", (mass, metallicity),
                (mass, metallicity, "young"),
                (mass, metallicity, "old"),
            ))
    return comparisons


def _three_parameter_significance_command_name(
        t1, t3, varied_parameter, fixed_levels):
    """Return the variable-command name for one posterior comparison."""
    prefix = _tier1_prefix(t1)
    if str(t3) != "stellar3params":
        prefix += _latex_token(Path(t3).name)
    first, second = fixed_levels
    if varied_parameter == "Mass":
        fixed_token = first.capitalize() + "FeH" + second.capitalize()
    elif varied_parameter == "FeH":
        fixed_token = first.capitalize() + "Mstar" + second.capitalize()
    elif varied_parameter == "Age":
        fixed_token = (
            first.capitalize() + "Mstar" + second.capitalize() + "FeH"
        )
    else:
        raise ValueError(
            "varied_parameter must be 'Mass', 'FeH', or 'Age'"
        )
    return prefix + varied_parameter + fixed_token + "Significance"


def _three_parameter_dynamic_range_command_name(
        t1, t3, varied_parameter, fixed_levels):
    """Return the variable-command name for one stellar dynamic range."""
    significance_name = _three_parameter_significance_command_name(
        t1, t3, varied_parameter, fixed_levels
    )
    return significance_name[:-len("Significance")] + "DynamicRange"


def _three_parameter_dynamic_ranges(catalog_path=None, cuts=None):
    """Calculate median stellar-property ratios for all table comparisons.

    The catalog is divided using the same mass, metallicity, and age cuts
    represented by the three-parameter Tier 2 directory names.  Ratios are
    always reported as the larger median divided by the smaller median.  The
    metallicity medians are converted from dex to linear abundance first.
    """
    import pandas as pd

    if catalog_path is None:
        catalog_path = (
            Path(__file__).resolve().parent / "cls_files" /
            "cls_all_stars_all_params.csv"
        )
    else:
        catalog_path = Path(catalog_path)
    parameter_cuts = dict(_THREE_PARAMETER_DEFAULT_CUTS)
    if cuts is not None:
        unknown = set(cuts) - set(parameter_cuts)
        if unknown:
            raise ValueError(
                f"unknown three-parameter cut names: {sorted(unknown)}"
            )
        parameter_cuts.update(cuts)

    catalog = pd.read_csv(catalog_path)
    required = set(parameter_cuts)
    missing = sorted(required - set(catalog.columns))
    if missing:
        raise KeyError(f"{catalog_path} lacks columns {missing}")
    for column in required:
        catalog[column] = pd.to_numeric(catalog[column], errors="coerce")

    def level_mask(column, level):
        values = catalog[column]
        cut = parameter_cuts[column]
        if column == "age":
            return values < cut if level == "young" else values >= cut
        return values > cut if level == "high" else values <= cut

    columns = {"Mass": "Mstar", "FeH": "feh", "Age": "age"}
    dynamic_ranges = {}
    for varied_parameter, fixed_levels, low_levels, high_levels in (
            _three_parameter_comparisons()):
        varied_column = columns[varied_parameter]
        medians = []
        for levels in (low_levels, high_levels):
            mass, metallicity, age = levels
            mask = (
                level_mask("Mstar", mass) &
                level_mask("feh", metallicity) &
                level_mask("age", age)
            )
            values = catalog.loc[mask, varied_column].dropna().to_numpy()
            if values.size == 0:
                raise ValueError(
                    f"no finite {varied_column} values for levels {levels} "
                    f"in {catalog_path}"
                )
            medians.append(float(np.median(values)))
        if varied_parameter == "FeH":
            medians = [10**value for value in medians]
        lower, upper = sorted(medians)
        if lower <= 0:
            raise ValueError(
                f"dynamic-range medians must be positive; got {medians} for "
                f"{varied_parameter} with fixed levels {fixed_levels}"
            )
        dynamic_ranges[(varied_parameter, fixed_levels)] = upper/lower
    return dynamic_ranges


def _three_parameter_occurrence_samples(result_dir):
    """Load the integrated piecewise-occurrence posterior for one subset."""
    chain_path = result_dir / "saved_chains" / "chains_piecewise.npz"
    if not chain_path.is_file():
        raise FileNotFoundError(f"piecewise chain not found: {chain_path}")
    with np.load(chain_path) as chain:
        if "flat_chains" not in chain or "x_edges" not in chain:
            raise KeyError(
                f"{chain_path} must contain 'flat_chains' and 'x_edges'"
            )
        y_key = "y_edges" if "y_edges" in chain else "stack_edges"
        if y_key not in chain:
            raise KeyError(
                f"{chain_path} must contain 'y_edges' or 'stack_edges'"
            )
        samples = np.asarray(chain["flat_chains"], dtype=float)[::10]
        a_edges = np.asarray(chain["x_edges"], dtype=float)
        m_edges = np.asarray(chain[y_key], dtype=float)
    cell_areas = np.outer(
        np.diff(np.log10(m_edges)), np.diff(np.log10(a_edges))
    ).reshape(-1)
    if samples.ndim != 2 or samples.shape[1] != cell_areas.size:
        raise ValueError(
            f"piecewise chain dimensions do not match edges in {chain_path}"
        )
    integrated = np.sum(samples*cell_areas[None, :], axis=1)
    if integrated.size < 2 or not np.isfinite(integrated).all():
        raise ValueError(
            f"integrated occurrence samples in {chain_path} must contain "
            "at least two finite values"
        )
    return integrated


def _posterior_difference_significance(low_samples, high_samples):
    """Return P(either sign), its Gaussian Z equivalent, and bound status.

    The probability is evaluated exactly for the two independent empirical
    posteriors: every high-posterior value is compared with every low-posterior
    value.  Exact ties contribute one half to each direction.
    """
    low_samples = np.asarray(low_samples, dtype=float).reshape(-1)
    high_samples = np.asarray(high_samples, dtype=float).reshape(-1)
    if (low_samples.size < 2 or high_samples.size < 2 or
            not np.isfinite(low_samples).all() or
            not np.isfinite(high_samples).all()):
        raise ValueError(
            "each occurrence posterior must contain at least two finite draws"
        )

    sorted_low = np.sort(low_samples)
    positive = np.searchsorted(
        sorted_low, high_samples, side="left"
    ).sum(dtype=np.int64)
    negative = (
        low_samples.size - np.searchsorted(
            sorted_low, high_samples, side="right"
        )
    ).sum(dtype=np.int64)
    total_comparisons = low_samples.size*high_samples.size
    ties = total_comparisons - int(positive + negative)
    probability = (
        max(positive, negative) + 0.5*ties
    ) / total_comparisons
    if probability == 1.0:
        resolution = min(low_samples.size, high_samples.size)
        finite_probability = 1.0 - 1.0/resolution
        return probability, float(norm.ppf(finite_probability)), True
    return probability, float(norm.ppf(probability)), False


def _format_posterior_significance(z_score, lower_bound=False):
    """Format an equivalent Gaussian significance for ``variables.tex``."""
    relation = ">" if lower_bound else ""
    return rf"{relation}{z_score:.1f}\,\sigma"


def _three_parameter_occurrence(result_dir):
    """Return formatted integrated occurrence for one stellar subset."""
    chain_path = result_dir / "saved_chains" / "chains_piecewise.npz"
    if chain_path.is_file():
        integrated = _three_parameter_occurrence_samples(result_dir)
        low, median, high = np.percentile(integrated, [16, 50, 84])
        return _format_parameter(median, low, high)

    summary_path = result_dir / "saved_dicts" / SUMMARY_FILENAME
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"no piecewise chain or summary found beneath {result_dir}"
        )
    with np.load(summary_path) as summary:
        if all(key in summary for key in (
                "mode_OR_single", "hdi_low_OR_single",
                "hdi_high_OR_single")):
            median = _scalar(summary, "mode_OR_single", summary_path)
            low = _scalar(summary, "hdi_low_OR_single", summary_path)
            high = _scalar(summary, "hdi_high_OR_single", summary_path)
        elif (all(key in summary for key in (
                "mode_OR", "hdi_low_OR", "hdi_high_OR")) and
                np.asarray(summary["mode_OR"]).size == 1):
            median = _scalar(summary, "mode_OR", summary_path)
            low = _scalar(summary, "hdi_low_OR", summary_path)
            high = _scalar(summary, "hdi_high_OR", summary_path)
        else:
            raise KeyError(
                f"{summary_path} lacks an integrated occurrence summary"
            )
    return _format_parameter(median, low, high)


def _three_parameter_statistics(result_dir):
    """Collect all quantities needed by the two three-parameter tables."""
    summary_path = result_dir / "saved_dicts" / SUMMARY_FILENAME
    if not summary_path.is_file():
        raise FileNotFoundError(f"piecewise summary not found: {summary_path}")
    with np.load(summary_path) as summary:
        nstars = _scalar(summary, "nstars", summary_path)
        if not nstars.is_integer():
            raise ValueError(f"'nstars' in {summary_path} must be an integer")
        if "cell_weights" not in summary:
            raise KeyError(f"{summary_path} does not contain 'cell_weights'")
        weights = np.asarray(summary["cell_weights"], dtype=float)
        if weights.size == 0 or not np.isfinite(weights).all():
            raise ValueError(
                f"'cell_weights' in {summary_path} must be nonempty and finite"
            )
        completeness = _scalar(summary, "cell_compl_single", summary_path)
    return {
        "Nstars": str(int(nstars)),
        "Neff": f"{np.sum(weights):.1f}",
        "AvgCompl": f"{completeness:.2f}",
        "IntOcc": _three_parameter_occurrence(result_dir),
    }


def make_three_parameter_tables(
        results_dir, t1="mtrue", t3="stellar3params", tier2_dirs=None,
        reordered_caption=(
            "Occurrence Rates with Two Stellar Parameters Held Fixed"
        ),
        reordered_label="tab:three_param_OR_reordered",
        original_caption=(
            "Occurrence Rates by Stellar Mass, Metallicity, and Age"
        ),
        original_label="tab:three_param_OR",
        reordered_output_file=None, original_output_file=None,
        use_latex_variables=True, stellar_catalog_path=None,
        three_parameter_cuts=None):
    """Create reordered and legacy-form tables for three stellar parameters.

    The eight subsets vary stellar mass, metallicity, and activity.  Following
    the legacy convention, high activity is labeled ``young`` and low activity
    is labeled ``old``.  Each occurrence rate appears in all three comparison
    blocks so one parameter can be compared while the other two are fixed.
    By default, cells in both tables reference the same command names that
    :func:`make_variables` emits, without requiring or inspecting
    ``variables.tex``.  Set ``use_latex_variables=False`` to read the saved
    results and write numerical values directly into both tables instead.
    ``stellar_catalog_path`` and ``three_parameter_cuts`` control the stellar
    samples used for the dynamic-range column.
    """
    results_dir = Path(results_dir)
    if tier2_dirs is None:
        tier2_dirs = _THREE_PARAMETER_TIER2_DIRS
    else:
        tier2_dirs = tuple(tier2_dirs)
    if len(tier2_dirs) != 8:
        raise ValueError("tier2_dirs must contain the eight stellar subsets")

    if not isinstance(use_latex_variables, (bool, np.bool_)):
        raise TypeError("use_latex_variables must be a boolean")

    table_values = {}
    for tier2_dir in tier2_dirs:
        levels = _three_parameter_levels(tier2_dir)
        if levels in table_values:
            raise ValueError(f"duplicate three-parameter subset: {levels}")
        if use_latex_variables:
            table_values[levels] = {
                statistic: rf"\{_three_parameter_command_name(t1, t3, levels, statistic)}"
                for statistic in ("Nstars", "Neff", "AvgCompl", "IntOcc")
            }
        else:
            result_dir = results_dir / t1 / tier2_dir / t3
            statistics = _three_parameter_statistics(result_dir)
            table_values[levels] = dict(statistics)
            table_values[levels]["IntOcc"] = f"${statistics['IntOcc']}$"
    expected = {
        (mass, metallicity, age)
        for mass in ("high", "low")
        for metallicity in ("high", "low")
        for age in ("young", "old")
    }
    if set(table_values) != expected:
        missing = sorted(expected - set(table_values))
        raise ValueError(f"three-parameter subsets are incomplete; missing {missing}")

    significance_values = {}
    dynamic_range_values = {}
    posterior_samples = {}
    numerical_dynamic_ranges = None
    if not use_latex_variables:
        numerical_dynamic_ranges = _three_parameter_dynamic_ranges(
            stellar_catalog_path, three_parameter_cuts
        )
        for tier2_dir in tier2_dirs:
            levels = _three_parameter_levels(tier2_dir)
            posterior_samples[levels] = _three_parameter_occurrence_samples(
                results_dir / t1 / tier2_dir / t3
            )
    for varied_parameter, fixed_levels, low_levels, high_levels in (
            _three_parameter_comparisons()):
        key = (varied_parameter, fixed_levels)
        command_name = _three_parameter_significance_command_name(
            t1, t3, varied_parameter, fixed_levels
        )
        if use_latex_variables:
            significance_values[key] = rf"\{command_name}"
            dynamic_range_name = (
                _three_parameter_dynamic_range_command_name(
                    t1, t3, varied_parameter, fixed_levels
                )
            )
            dynamic_range_values[key] = rf"\{dynamic_range_name}"
        else:
            probability, z_score, lower_bound = (
                _posterior_difference_significance(
                    posterior_samples[low_levels],
                    posterior_samples[high_levels],
                )
            )
            significance_values[key] = (
                f"${_format_posterior_significance(z_score, lower_bound)}$"
            )
            dynamic_range_values[key] = (
                f"{numerical_dynamic_ranges[key]:.1f}"
            )

    lines = [
        r"\begin{deluxetable*}{cccccc}",
        rf"\tablecaption{{{reordered_caption}}}",
        rf"\label{{{reordered_label}}}",
        r"\tablehead{",
        r"\colhead{Fixed Parameter 1} &",
        r"\colhead{Fixed Parameter 2} &",
        r"\multicolumn{2}{c}{Occurrence Rate} &",
        r"\colhead{Significance} &",
        r"\colhead{Dynamic Range}",
        r"}",
        r"\startdata",
    ]

    def add_block(fixed_one, fixed_two, low_heading, high_heading, rows):
        lines.append(
            rf"\textbf{{{fixed_one}}} & \textbf{{{fixed_two}}} & "
            rf"\textbf{{{low_heading}}} & \textbf{{{high_heading}}} & "
            r"\textbf{Significance} & \textbf{Dynamic Range} \\"
        )
        lines.append(r"\hline")
        for (first, second, low_value, high_value, significance,
                dynamic_range) in rows:
            lines.append(
                rf"{first} & {second} & {low_value} & {high_value} & "
                rf"{significance} & {dynamic_range} \\"
            )
        lines.append(r"\hline")

    add_block(
        "[Fe/H]", "Age", "Low Mass", "High Mass",
        [
            (metallicity, age,
             table_values[("low", metallicity, age)]["IntOcc"],
             table_values[("high", metallicity, age)]["IntOcc"],
             significance_values[("Mass", (metallicity, age))],
             dynamic_range_values[("Mass", (metallicity, age))])
            for metallicity in ("high", "low")
            for age in ("old", "young")
        ],
    )
    add_block(
        "Mass", "Age", "Low [Fe/H]", "High [Fe/H]",
        [
            (mass, age,
             table_values[(mass, "low", age)]["IntOcc"],
             table_values[(mass, "high", age)]["IntOcc"],
             significance_values[("FeH", (mass, age))],
             dynamic_range_values[("FeH", (mass, age))])
            for mass in ("high", "low")
            for age in ("old", "young")
        ],
    )
    add_block(
        "Mass", "[Fe/H]", "Young", "Old",
        [
            (mass, metallicity,
             table_values[(mass, metallicity, "young")]["IntOcc"],
             table_values[(mass, metallicity, "old")]["IntOcc"],
             significance_values[("Age", (mass, metallicity))],
             dynamic_range_values[("Age", (mass, metallicity))])
            for mass in ("high", "low")
            for metallicity in ("high", "low")
        ],
    )
    lines.extend([
        r"\enddata",
        r"\end{deluxetable*}",
    ])

    if reordered_output_file is None:
        reordered_output_path = (
            results_dir / PAPER_ITEMS_DIRNAME /
            f"three_parameter_OR_reordered_{t1}_{t3}.tex"
        )
    else:
        reordered_output_path = Path(reordered_output_file)
    reordered_output_path.parent.mkdir(parents=True, exist_ok=True)
    reordered_output_path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    original_lines = [
        r"\begin{deluxetable*}{lcccccc}",
        rf"\tablecaption{{{original_caption}}}",
        rf"\label{{{original_label}}}",
        r"\tablehead{",
        r"\colhead{Mass} &",
        r"\colhead{[Fe/H]} &",
        r"\colhead{Age} &",
        r"\colhead{$N_{\star}$} &",
        r"\colhead{$N_{\mathrm{eff}}$} &",
        r"\colhead{Completeness} &",
        r"\colhead{Occurrence Rate}",
        r"}",
        r"\startdata",
    ]
    for tier2_dir in tier2_dirs:
        levels = _three_parameter_levels(tier2_dir)
        mass, metallicity, age = levels
        values = table_values[levels]
        original_lines.append(
            f"{mass} & {metallicity} & {age} & "
            f"{values['Nstars']} & {values['Neff']} & "
            f"{values['AvgCompl']} & {values['IntOcc']} " + r"\\"
        )
    original_lines.extend([
        r"\enddata",
        r"\end{deluxetable*}",
    ])
    if original_output_file is None:
        original_output_path = (
            results_dir / PAPER_ITEMS_DIRNAME /
            f"three_parameter_OR_{t1}_{t3}.tex"
        )
    else:
        original_output_path = Path(original_output_file)
    original_output_path.parent.mkdir(parents=True, exist_ok=True)
    original_output_path.write_text(
        "\n".join(original_lines) + "\n", encoding="utf-8"
    )
    return {
        "reordered": reordered_output_path,
        "original": original_output_path,
    }


def calculate_all_delta_bics(
        results_dir, tier1_dirs, tier2_types, tier3_dirs, stack_dim="a"):
    """Calculate and save delta-BIC values for every requested results folder.

    The arguments and Tier 2 expansion rules match :func:`make_variables`.
    Results are returned as a dictionary keyed by relative results paths and
    are also saved to ``results_dir/paper_items/delta_bics.json``.  Folders
    containing no recognized parametric chains are represented by an empty
    dictionary.
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
                            f"chains_{model_name}_bin*.npz"
                        ), None) is not None
                        for model_name in _MODEL_PARAMETER_MACROS
                    ) if chain_dir.is_dir() else False
                    if not has_parametric_chains:
                        all_delta_bics[key] = {}
                        continue
                    comparisons = calculate_delta_bic(
                        result_dir / "saved_dicts" / "fit_data.npz",
                        chain_dir,
                        stack_dim,
                    )
                    all_delta_bics[key] = {
                        model_name: np.asarray(values, dtype=float).tolist()
                        for model_name, values in comparisons.items()
                    }

    paper_items_dir = results_dir / PAPER_ITEMS_DIRNAME
    paper_items_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "stack_dim": stack_dim,
        "delta_bic_convention": "BIC_flat - BIC_model",
        "results": all_delta_bics,
    }
    (paper_items_dir / DELTA_BIC_FILENAME).write_text(
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
        results_dir, tier1_dirs, tier2_types, tier3_dirs, stack_dim="a",
        three_parameter_t3="stellar3params", stellar_catalog_path=None,
        three_parameter_cuts=None):
    """Write non-model fit statistics to a LaTeX variables file.

    ``results_dir`` is the parent of all Tier 1 directories.  The output is
    written to ``results_dir/paper_items/variables.tex``.  ``tier2_types``
    contains unsplit names such as ``"Mass"`` and ``"FeH"``;
    each is expanded to its ``high`` and ``low`` directories.  ``"allstars"``
    is treated as a single directory.  A completed fit is expected at::

        results_dir/tier1/tier2/tier3/saved_dicts/
        summary_dict_piecewise.npz

    The output contains ``Nstars``, ``Neff`` (the sum of the effective counts),
    and ``AvgCompl`` (the area-averaged completeness across the full ROI), plus
    ``Neff`` and ``AvgCompl`` for every bin along ``stack_dim``.  Stack-bin
    labels are zero-based, matching the legacy analysis (for example,
    ``NeffBinAZero``).
    With one Tier 3 directory, command names mirror the directory hierarchy,
    for example ``\\McallstarsNeff`` and ``\\McHighMassNeff``.  When several
    Tier 3 directories are supplied, their names are included to keep commands
    unique.  Available three-parameter subset results beneath
    ``three_parameter_t3`` are also included.  When both required piecewise
    chains are available, the file also includes the posterior significance
    of every comparison in the reordered three-parameter table.  Dynamic
    ranges use ``stellar_catalog_path`` and ``three_parameter_cuts``; their
    defaults are the repository CLS stellar catalog and cuts at 1 solar mass,
    zero dex, and 5 Gyr.  Missing subsets or comparison chains are reported
    and omitted without interrupting generation of the other commands.

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
            for tier2_dir, _ in _tier2_directories(tier2_type):
                for tier3_dir in tier3_dirs:
                    summary_path = (
                        tier1_path / tier2_dir / tier3_dir / "saved_dicts" /
                        SUMMARY_FILENAME
                    )
                    if not summary_path.is_file():
                        raise FileNotFoundError(
                            f"no piecewise summary found at {summary_path}"
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

                    prefix = (
                        _tier1_prefix(tier1_name) +
                        _tier2_command_token(tier2_dir)
                    )
                    if include_tier3:
                        prefix += _latex_token(Path(tier3_dir).name)
                    values = {
                        "Nstars": str(int(nstars)),
                        "Neff": f"{neff:.1f}",
                        "AvgCompl": f"{avg_compl:.2f}",
                    }
                    dim = stack_dim.upper()
                    for bin_index, (neff_value, compl_value) in enumerate(
                            zip(bin_neff, bin_avg_compl)):
                        bin_label = _number_word(bin_index)
                        values[
                            f"NeffBin{dim}{bin_label}"
                        ] = f"{neff_value:.1f}"
                        values[
                            f"AvgComplBin{dim}{bin_label}"
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
                        chain_dir, prefix, stack_dim, n_stack_bins,
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
                            chain_dir, prefix, stack_dim, n_stack_bins)
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
                                    prefix + model_macro + "Dbic" +
                                    f"Bin{stack_dim.lower()}{_number_word(bin_index)}"
                                )
                                if name in command_names:
                                    raise ValueError(
                                        f"duplicate LaTeX command name: {name}"
                                    )
                                command_names.add(name)
                                block.append(_command(name, f"{delta:.1f}"))
                    blocks.append("\n".join(block))

    missing_three_parameter_results = []
    missing_three_parameter_comparisons = []
    if three_parameter_t3 is not None:
        for tier1_dir in tier1_dirs:
            tier1_name = Path(tier1_dir).name
            occurrence_samples = {}
            available_levels = set()
            for tier2_dir in _THREE_PARAMETER_TIER2_DIRS:
                result_dir = (
                    results_dir / tier1_dir / tier2_dir / three_parameter_t3
                )
                chain_path = (
                    result_dir / "saved_chains" / "chains_piecewise.npz"
                )
                summary_path = result_dir / "saved_dicts" / SUMMARY_FILENAME
                if not chain_path.is_file() and not summary_path.is_file():
                    missing_three_parameter_results.append(
                        _result_key(tier1_dir, tier2_dir, three_parameter_t3)
                    )
                    continue
                levels = _three_parameter_levels(tier2_dir)
                available_levels.add(levels)
                if chain_path.is_file():
                    occurrence_samples[levels] = (
                        _three_parameter_occurrence_samples(result_dir)
                    )
                statistics = _three_parameter_statistics(result_dir)
                block = [
                    "%"*72,
                    f"% {tier1_name} / {tier2_dir} / {three_parameter_t3}",
                ]
                for statistic, value in statistics.items():
                    name = _three_parameter_command_name(
                        tier1_name, three_parameter_t3, levels, statistic
                    )
                    if name in command_names:
                        raise ValueError(f"duplicate LaTeX command name: {name}")
                    command_names.add(name)
                    block.append(_command(name, value))
                blocks.append("\n".join(block))

            comparison_block = [
                "%"*72,
                f"% {tier1_name} / {three_parameter_t3} comparisons",
            ]
            dynamic_ranges = None
            for varied_parameter, fixed_levels, low_levels, high_levels in (
                    _three_parameter_comparisons()):
                significance_name = _three_parameter_significance_command_name(
                    tier1_name, three_parameter_t3,
                    varied_parameter, fixed_levels,
                )
                if (low_levels not in occurrence_samples or
                        high_levels not in occurrence_samples):
                    missing_three_parameter_comparisons.append(
                        significance_name
                    )
                else:
                    probability, z_score, lower_bound = (
                        _posterior_difference_significance(
                            occurrence_samples[low_levels],
                            occurrence_samples[high_levels],
                        )
                    )
                    if significance_name in command_names:
                        raise ValueError(
                            "duplicate LaTeX command name: "
                            f"{significance_name}"
                        )
                    command_names.add(significance_name)
                    comparison_block.append(_command(
                        significance_name,
                        _format_posterior_significance(z_score, lower_bound),
                    ))

                if (low_levels in available_levels and
                        high_levels in available_levels):
                    if dynamic_ranges is None:
                        dynamic_ranges = _three_parameter_dynamic_ranges(
                            stellar_catalog_path, three_parameter_cuts
                        )
                    dynamic_name = (
                        _three_parameter_dynamic_range_command_name(
                            tier1_name, three_parameter_t3,
                            varied_parameter, fixed_levels,
                        )
                    )
                    if dynamic_name in command_names:
                        raise ValueError(
                            f"duplicate LaTeX command name: {dynamic_name}"
                        )
                    command_names.add(dynamic_name)
                    comparison_block.append(_command(
                        dynamic_name, f"{dynamic_ranges[(varied_parameter, fixed_levels)]:.2f}"
                    ))
            if len(comparison_block) > 2:
                blocks.append("\n".join(comparison_block))
    if missing_three_parameter_results:
        print(
            "post_fit_analysis.make_variables: three-parameter occurrence "
            "results have not been calculated for: "
            + ", ".join(missing_three_parameter_results)
        )
    if missing_three_parameter_comparisons:
        print(
            "post_fit_analysis.make_variables: three-parameter significance "
            "could not be calculated without both piecewise chains for: "
            + ", ".join(missing_three_parameter_comparisons)
        )

    paper_items_dir = results_dir / PAPER_ITEMS_DIRNAME
    output_path = paper_items_dir / "variables.tex"
    paper_items_dir.mkdir(parents=True, exist_ok=True)
    contents = "% Auto-generated by post_fit_analysis.make_variables\n\n"
    contents += "\n\n".join(blocks) + "\n"
    output_path.write_text(contents, encoding="utf-8")
    return output_path
