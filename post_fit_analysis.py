"""Collect publication-ready quantities from completed occurrence fits."""

import json
import os
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
PAPER_TABLES_DIRNAME = "paper_tables"

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
        r"$B_1$", r"$B_2$", r"$\log_{10}(x_t)$", r"$W$"
    ),
    "bpl": (r"$C$", r"$\log_{10}(x_0)$", r"$\beta$", r"$\gamma$"),
    "loglinear": (r"$D_1$", r"$D_2$"),
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

_TWO_PARAMETER_TIER2_DIRS = (
    "highMstarhighFeH", "highMstarlowFeH",
    "lowMstarhighFeH", "lowMstarlowFeH",
)

_THREE_PARAMETER_DEFAULT_CUTS = {
    "Mstar": 1.0,
    "feh": 0.0,
    "age": 5.0,
    "Mstar_min": 0.82,
    "Mstar_max": 1.21,
}


_DIGIT_WORDS = (
    "Zero", "One", "Two", "Three", "Four",
    "Five", "Six", "Seven", "Eight", "Nine",
)


def _latex_token(value):
    """Turn a directory/type name into a legal, readable command token.

    LaTeX command names may contain only letters, so each digit is spelled
    out: ``stellar3params_Miyazaki`` becomes ``StellarThreeParamsMiyazaki``
    and ``10`` becomes ``OneZero``.
    """
    parts = re.findall(r"[A-Za-z]+|\d", str(value))
    return "".join(
        _DIGIT_WORDS[int(part)] if part.isdigit()
        else part[:1].upper() + part[1:]
        for part in parts
    )


def _tier1_prefix(tier1_dir):
    name = Path(tier1_dir).name
    return _TIER1_LATEX_PREFIXES.get(name, _latex_token(name))


def _tier2_command_token(tier2_dir):
    """Return a command token that mirrors a Tier 2 directory, e.g. Allstars."""
    return _latex_token(Path(tier2_dir).name)


def _experiment_prefix(tier1_dir, tier2_dir, tier3_dir):
    """Return the command prefix naming one tier1/tier2/tier3 experiment.

    For example, ``mtrue/allstars/paper_bounds`` gives
    ``McAllstarsPaperBounds``.
    """
    return (
        _tier1_prefix(Path(tier1_dir).name) +
        _tier2_command_token(tier2_dir) +
        _latex_token(Path(tier3_dir).name)
    )


def _tier2_directories(tier2_type):
    """Return ``(directory, macro suffix)`` pairs for a Tier 2 type."""
    if str(tier2_type).lower() == "allstars":
        return [(str(tier2_type), "")]
    return [
        (f"high{tier2_type}", "High"),
        (f"low{tier2_type}", "Low"),
    ]


def _requested_tier2_dirs(tier2_types, standalone_tier2_dirs=()):
    """Expand Tier 2 types (``Mstar`` -> high/low) and append standalone dirs."""
    directories = [
        tier2_dir for tier2_type in tier2_types
        for tier2_dir, _ in _tier2_directories(tier2_type)
    ]
    for tier2_dir in standalone_tier2_dirs:
        if tier2_dir not in directories:
            directories.append(str(tier2_dir))
    return directories


def _tier3_stack_dim(tier3_dir, stack_dim, tier3_stack_dims=None):
    """Return the stack dimension for one Tier 3 directory."""
    dim = (tier3_stack_dims or {}).get(Path(tier3_dir).name, stack_dim)
    if dim not in {"a", "m"}:
        raise ValueError(f"stack dimension for {tier3_dir} must be 'a' or 'm'")
    return dim


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


MODEL_TITLES = {
    "piecewise": "Piecewise", "logG": "Log-Gaussian", "sigmoid": "Sigmoid",
    "escarpment": "Escarpment", "bpl": "Broken Power Law",
    "loglinear": "Log-Linear",
}


def model_cdf_samples(chain_path, model_name, n_grid=500, max_samples=2000):
    """Return ``(grid, cdfs)`` for a fitted model normalized over its bounds.

    Each row of ``cdfs`` is one posterior sample's cumulative distribution:
    the occurrence-rate density integrated in ``log10`` of the model
    coordinate from the lower model bound, divided by its integral over the
    full bounds, so every row rises from 0 to exactly 1.  Normalizing removes
    the total occurrence rate, so the spread between rows reflects only the
    uncertainty in the model's shape.  At most ``max_samples`` posterior
    samples, evenly spaced through the chain, are used.
    """
    chain_path = Path(chain_path)
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2")
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    with np.load(chain_path) as data:
        samples = np.asarray(data["flat_chains"], dtype=float)
        model_bounds = tuple(float(bound) for bound in data["model_bounds"])
    if len(samples) > max_samples:
        samples = samples[
            np.linspace(0, len(samples) - 1, max_samples, dtype=int)
        ]
    grid = np.logspace(*np.log10(model_bounds), n_grid)
    densities = np.array([
        mcmc_powerlaw.evaluate_density(model_name, theta, grid, model_bounds)
        for theta in samples
    ])
    steps = (
        0.5*(densities[:, 1:] + densities[:, :-1])*np.diff(np.log10(grid))
    )
    cumulative = np.concatenate(
        (np.zeros((len(samples), 1)), np.cumsum(steps, axis=1)), axis=1
    )
    totals = cumulative[:, -1:]
    if np.any(totals <= 0):
        raise ValueError(
            f"{chain_path} has posterior samples with no positive density"
        )
    return grid, cumulative/totals


def _cdf_tick_values(curve, coordinate):
    """Return the bin edges of a curve's piecewise fit along ``coordinate``.

    These are the ticks the occurrence plots use.  ``None`` is returned when
    the experiment has no piecewise chain.
    """
    path = (Path(curve["results_dir"]) / curve["t1"] / curve["t2"] /
            curve["t3"] / "saved_chains" / "chains_piecewise.npz")
    if not path.is_file():
        return None
    key = "x_edges" if coordinate == "sma" else "y_edges"
    with np.load(path) as data:
        return np.asarray(data[key], dtype=float) if key in data else None


def _cdf_tick_formatter(tick_values, coordinate, tier1_names):
    from occurrence import plotting_utils as pu

    if coordinate == "sma":
        return pu.sma_tick_formatter(tick_values)
    if tier1_names <= {"qtrue", "qsini"}:
        return pu.mass_ratio_tick_formatter(tick_values)
    return pu.mass_tick_formatter(tick_values)


def _set_figure_axis_labels(figure, xlabel, ylabel, axes=None,
                            row_labels=None, top_axes=None):
    """Label a panel grid once along its bottom and left edges.

    The labels use the axis-label font size and are placed in margins kept
    free by the layout, so they never overlap tick labels.  With
    ``row_labels`` (one per row of the 2-D array ``axes``), each row also
    gets a single label centered above it, in the title font size and above
    any panel titles in that row.  ``top_axes`` (same shape as ``axes``)
    holds panels stacked above each row's main panels, if any; the row
    labels then go above those.
    """
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties

    size = FontProperties(
        size=plt.rcParams["axes.labelsize"]
    ).get_size_in_points()
    width, height = figure.get_size_inches()
    # Margins (as figure fractions) just tall enough for one line of label
    # text; tight_layout adds its own padding for the tick labels.
    margin_x = 1.3*size/72/width
    margin_y = 1.3*size/72/height
    if row_labels is None:
        figure.tight_layout(rect=(margin_x, margin_y, 1, 1))
    else:
        row_size = FontProperties(
            size=plt.rcParams["axes.titlesize"]
        ).get_size_in_points()
        # Room for one line of row-label text above each row: extra space
        # at the top of the figure and between rows (h_pad is measured in
        # multiples of the base font size).
        figure.tight_layout(
            rect=(margin_x, margin_y, 1, 1 - 1.6*row_size/72/height),
            h_pad=1.6*row_size/plt.rcParams["font.size"],
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        to_figure = figure.transFigure.inverted()
        gap = 0.3*row_size/72/height
        if top_axes is None:
            top_axes = axes
        for row_axes, label in zip(top_axes, row_labels):
            top = max(axis.get_position().y1 for axis in row_axes)
            for axis in row_axes:
                if axis.get_title():
                    extent = axis.title.get_window_extent(renderer)
                    top = max(top, to_figure.transform(
                        (0, extent.y1))[1])
            left = min(axis.get_position().x0 for axis in row_axes)
            right = max(axis.get_position().x1 for axis in row_axes)
            figure.text((left + right)/2, top + gap, label,
                        fontsize=row_size, ha="center", va="bottom")
    if hasattr(figure, "supxlabel"):
        figure.supxlabel(xlabel, fontsize=size, y=0.01, va="bottom")
        figure.supylabel(ylabel, fontsize=size, x=0.01, ha="left")
    else:  # matplotlib < 3.4
        figure.text(0.5 + margin_x/2, 0.01, xlabel, fontsize=size,
                    ha="center", va="bottom")
        figure.text(0.01, 0.5 + margin_y/2, ylabel, fontsize=size,
                    ha="left", va="center", rotation="vertical")


_CDF_LEGEND_GRAY = "0.35"


def _cdf_difference_significance(first, second):
    """Return the pointwise significance of ``median(first) - median(second)``.

    ``first`` and ``second`` are ``(samples, grid)`` arrays of CDF draws on
    the same grid.  At each grid point the difference of the medians is
    divided by the 16th/84th-percentile errors that face each other, added
    in quadrature (as in :func:`_comparison_significance`).  Points where
    both errors vanish, such as the ends of the grid where every CDF is 0
    or 1, are ``nan``.
    """
    first_low, first_median, first_high = np.percentile(
        first, [16, 50, 84], axis=0
    )
    second_low, second_median, second_high = np.percentile(
        second, [16, 50, 84], axis=0
    )
    difference = first_median - second_median
    error = np.where(
        difference >= 0,
        np.hypot(first_median - first_low, second_high - second_median),
        np.hypot(first_high - first_median, second_median - second_low),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(error > 1e-9, difference/error, np.nan)


def _cdf_draw_difference_significance(first, second, seed=0):
    """Return the pointwise significance from draw-by-draw CDF differences.

    The two posteriors are independent, so draws are paired at random
    (``min`` of the two sample counts, seeded by ``seed``) and differenced
    curve by curve, giving the posterior of ``first - second`` at every grid
    point without assuming Gaussian errors.  The significance is that
    posterior's median divided by its 16th/84th-percentile error on the side
    facing zero.  Points where the difference has no spread, such as the
    ends of the grid, are ``nan``.
    """
    rng = np.random.default_rng(seed)
    count = min(len(first), len(second))
    difference = (
        first[rng.permutation(len(first))[:count]] -
        second[rng.permutation(len(second))[:count]]
    )
    low, median, high = np.percentile(difference, [16, 50, 84], axis=0)
    error = np.where(median >= 0, median - low, high - median)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(error > 1e-9, median/error, np.nan)


_CDF_SIGNIFICANCE_METHODS = {
    "draws": _cdf_draw_difference_significance,
    "quadrature": _cdf_difference_significance,
}


def _cdf_curve_handle(band_style, linestyle, show_line, band_alpha, hatch,
                      hatch_alpha, hatch_edge_width, outline_style,
                      outline_width):
    """Return a gray legend handle showing one CDF curve's band style.

    With ``show_line`` the median's line style is drawn over the band, for
    rows whose medians differ in line style.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    gray = _CDF_LEGEND_GRAY
    if band_style == "fill":
        band = Patch(facecolor=gray, alpha=band_alpha, edgecolor="none")
    elif band_style == "hatch":
        band = Patch(facecolor="none", edgecolor=to_rgba(gray, hatch_alpha),
                     hatch=hatch, lw=hatch_edge_width)
    else:
        band = Patch(facecolor="none", edgecolor=gray, ls=outline_style,
                     lw=outline_width)
    if not show_line:
        return band
    return (band, Line2D([], [], color=gray, lw=2, ls=linestyle))


def plot_model_cdf_comparison(
        results_dir, rows, models, name, title=None, credible=0.68,
        stack_bin=0, n_grid=500, max_samples=2000, xticks=None, xlabel=None,
        ylabel="Cumulative fraction", legend_loc="lower right",
        legend_fontsize=None, panel_size=(5, 3.5), model_colors=None,
        linestyles=("-", "--"), band_styles=("fill", "hatch"),
        band_alpha=0.3, hatch="++++", hatch_alpha=0.6, hatch_linewidth=0.6,
        hatch_edge_width=0.8, outline_style=":", outline_width=1.2,
        row_labels=None, significance_panel=True, significance_height=0.35,
        significance_ylabel=r"$|\Delta|/\sigma$", significance_method="draws",
        significance_pair=(0, 1), significance_seed=0,
        significance_threshold=2, dpi=300,
        output_file=None):
    """Draw a grid comparing the normalized CDFs of fitted samples.

    The grid has one column per entry of ``models`` (e.g. ``["sigmoid",
    "logG"]``) and one row per entry of ``rows``.  Each row is a list of
    curves drawn in every column, typically a pair of stellar samples such
    as high- and low-mass hosts.  A curve is a mapping with ``label`` (legend
    text) and ``t1``, ``t2``, and ``t3`` (the experiment folders), plus an
    optional ``stack_bin`` that overrides the shared ``stack_bin``.  Every
    curve needs a saved chain for every model; all missing chains are listed
    before anything is drawn.  Each curve is the posterior median of
    :func:`model_cdf_samples`, with the central ``credible`` interval shaded.

    ``title`` optionally heads each column, with ``{model}`` replaced by the
    column's model name, so ``"{model} CDF"`` gives "Sigmoid CDF" and
    "Log-Gaussian CDF".  By default the columns are untitled, since the
    top row's legends name each column's model.  Each row's legend appears
    in its first column, with ``legend_fontsize`` defaulting to the size the
    occurrence plots use.  Panels share both axes, so the figure carries a
    single x-axis label along the bottom and a single y-axis label along the
    left edge.  ``xticks`` lists the x-axis tick values; by default they are the
    bin edges of the first curve's piecewise fit, as on the occurrence plots,
    and matplotlib's ticks are used if that fit is missing.  ``xlabel``
    defaults to companion mass or mass ratio, following the Tier 1 folders.

    Each panel is drawn in its model's color from the occurrence plots
    (``mcmc_powerlaw.MODEL_REGISTRY``), which ``model_colors`` may override
    per model.  Curves within a row are told apart by position: the i-th
    curve uses ``linestyles[i]`` for its median and ``band_styles[i]`` for
    its credible interval, one of ``"fill"`` (shaded with ``band_alpha``),
    ``"hatch"`` (hatched with ``hatch`` in lines ``hatch_linewidth`` wide at
    opacity ``hatch_alpha``, bounded by solid edges ``hatch_edge_width``
    wide), or ``"outline"`` (edges drawn with ``outline_style`` and
    ``outline_width``).  By default the first curve has a solid median and
    a filled band, and the second a dashed median and a band hatched as a
    dense square grid.  Both sequences repeat for longer rows.

    Legends avoid repetition: the top-left panel shows its model's band
    color (labeled with the model name) and each curve's style; the rest of
    the top row shows only its model's band color; the rest of the first
    column shows only its row's curve styles.  Curve styles are drawn in
    gray, since they identify curves in every column: the band style, with
    the median's line style when the medians differ.
    ``row_labels`` optionally gives one label per row (e.g. the stellar
    parameter dividing its samples), centered above the row's panels.

    With ``significance_panel`` (the default), a shorter panel of the same
    width sits on top of each main panel, ``significance_height`` times its
    height, showing the significance of the difference between two of the
    row's CDFs across the grid, as an absolute value, so the panel shows how
    far apart the curves are but not which is higher (the main panel shows
    that).  ``significance_pair`` gives their positions in each row (default
    ``(0, 1)``: first minus second).  With
    ``significance_method="draws"`` (the default), randomly paired draws of
    the two CDFs are differenced curve by curve and the significance is the
    median difference over its 16th/84th-percentile error on the side
    facing zero (:func:`_cdf_draw_difference_significance`, seeded by
    ``significance_seed``); ``"quadrature"`` instead divides the difference
    of the medians by the two curves' facing errors added in quadrature
    (:func:`_cdf_difference_significance`).  Both ignore ``credible``.
    These panels share one y-axis starting at zero, labeled
    ``significance_ylabel`` in the first column, with a red dotted line at
    ``significance_threshold`` (default 2) to guide the eye; ``None`` omits
    it.
    ``panel_size`` is the size of each panel in inches.  The
    figure is saved with the catalog plots of
    :func:`plot_companions_by_stellar_parameter`, in the ``plots`` folder of
    the first curve's full-sample experiment
    (``results_dir/<t1>/allstars/<t3>/plots/<name>.png``), unless
    ``output_file`` is given.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch
    from matplotlib.ticker import (
        FixedLocator, FuncFormatter, MaxNLocator, NullLocator,
    )

    results_dir = Path(results_dir)
    rows = [[dict(curve) for curve in row] for row in rows]
    models = [models] if isinstance(models, str) else list(models)
    if not rows or not all(rows):
        raise ValueError("rows must each list at least one curve")
    if not models or len(set(models)) != len(models):
        raise ValueError("models must list at least one distinct model")
    for row in rows:
        for curve in row:
            missing = sorted({"label", "t1", "t2", "t3"} - set(curve))
            if missing:
                raise ValueError(f"CDF curve {curve} is missing {missing}")
    if not 0 < credible < 1:
        raise ValueError("credible must be between 0 and 1")
    if row_labels is not None and len(row_labels) != len(rows):
        raise ValueError("row_labels must give one label per row")
    linestyles = list(linestyles)
    band_styles = list(band_styles)
    if not linestyles or not band_styles:
        raise ValueError("linestyles and band_styles cannot be empty")
    unknown_bands = sorted(set(band_styles) - {"fill", "hatch", "outline"})
    if unknown_bands:
        raise ValueError(
            "band_styles must be 'fill', 'hatch', or 'outline', not "
            f"{unknown_bands}"
        )
    colors = {
        model: mcmc_powerlaw.get_model_spec(model).color for model in models
    }
    colors.update(model_colors or {})

    chain_paths = {}
    missing_chains = []
    for row_index, row in enumerate(rows):
        for curve_index, curve in enumerate(row):
            for model in models:
                path = (
                    results_dir / curve["t1"] / curve["t2"] / curve["t3"] /
                    "saved_chains" /
                    f"chains_{model}_bin{curve.get('stack_bin', stack_bin)}.npz"
                )
                chain_paths[row_index, curve_index, model] = path
                if not path.is_file():
                    missing_chains.append(str(path))
    if missing_chains:
        raise FileNotFoundError(
            "every sample needs a chain for every model; missing: " +
            ", ".join(missing_chains)
        )

    tier1_names = {Path(curve["t1"]).name for row in rows for curve in row}
    with np.load(chain_paths[0, 0, models[0]]) as data:
        coordinate = (
            str(data["model_coordinate"]) if "model_coordinate" in data
            else "mass"
        )
    if xticks is None:
        xticks = _cdf_tick_values(
            dict(rows[0][0], results_dir=results_dir), coordinate
        )
    if xticks is not None:
        xticks = np.asarray(xticks, dtype=float)
        tick_formatter = _cdf_tick_formatter(xticks, coordinate, tier1_names)
    if legend_fontsize is None:
        legend_fontsize = 1.6*plt.rcParams["font.size"]
    if xlabel is None:
        xlabel = (
            r"Mass Ratio [$M_c/M_{\star}$]"
            if tier1_names <= {"qtrue", "qsini"}
            else r"Companion mass [$M_{Jup}$]"
        )
    n_rows, n_columns = len(rows), len(models)
    significance_axes = None
    if significance_panel:
        if significance_height <= 0:
            raise ValueError("significance_height must be positive")
        if significance_method not in _CDF_SIGNIFICANCE_METHODS:
            raise ValueError(
                "significance_method must be one of "
                f"{sorted(_CDF_SIGNIFICANCE_METHODS)}"
            )
        significance_pair = tuple(significance_pair)
        if (len(significance_pair) != 2 or
                len(set(significance_pair)) != 2 or
                any(not isinstance(index, (int, np.integer)) or index < 0
                    for index in significance_pair)):
            raise ValueError(
                "significance_pair must give two distinct curve positions"
            )
        if significance_threshold is not None and not (
                np.isfinite(significance_threshold) and
                significance_threshold > 0):
            raise ValueError("significance_threshold must be positive or None")
        short_rows = [index for index, row in enumerate(rows)
                      if max(significance_pair) >= len(row)]
        if short_rows:
            raise ValueError(
                f"rows {short_rows} have no curves at significance_pair "
                f"{significance_pair}"
            )
        figure = plt.figure(figsize=(
            panel_size[0]*n_columns,
            panel_size[1]*(1 + significance_height)*n_rows,
        ))
        outer = figure.add_gridspec(n_rows, n_columns)
        axes = np.empty((n_rows, n_columns), dtype=object)
        significance_axes = np.empty((n_rows, n_columns), dtype=object)
        for row_index in range(n_rows):
            for column in range(n_columns):
                # Each cell stacks the significance panel on its main panel.
                inner = outer[row_index, column].subgridspec(
                    2, 1, height_ratios=[significance_height, 1], hspace=0,
                )
                axes[row_index, column] = figure.add_subplot(
                    inner[1],
                    sharex=axes[0, column] if row_index else None,
                    sharey=axes[0, 0] if row_index or column else None,
                )
                significance_axes[row_index, column] = figure.add_subplot(
                    inner[0], sharex=axes[row_index, column],
                    sharey=significance_axes[0, 0]
                    if row_index or column else None,
                )
                axes[row_index, column].tick_params(
                    labelbottom=row_index == n_rows - 1,
                    labelleft=column == 0,
                )
                significance_axes[row_index, column].tick_params(
                    labelbottom=False, labelleft=column == 0,
                )
    else:
        figure, axes = plt.subplots(
            n_rows, n_columns, squeeze=False, sharex="col", sharey=True,
            figsize=(panel_size[0]*n_columns, panel_size[1]*n_rows),
        )
    tail = 50*(1 - credible)
    largest_significance = 1.0
    for row_index, row in enumerate(rows):
        for column, model in enumerate(models):
            axis = axes[row_index, column]
            grid_limits = []
            curve_draws = []
            for curve_index, curve in enumerate(row):
                grid, cdfs = model_cdf_samples(
                    chain_paths[row_index, curve_index, model], model,
                    n_grid=n_grid, max_samples=max_samples,
                )
                curve_draws.append((grid, cdfs))
                low, median, high = np.percentile(
                    cdfs, [tail, 50, 100 - tail], axis=0
                )
                color = colors[model]
                linestyle = linestyles[curve_index % len(linestyles)]
                band_style = band_styles[curve_index % len(band_styles)]
                if band_style == "fill":
                    axis.fill_between(grid, low, high, color=color,
                                      alpha=band_alpha, lw=0)
                elif band_style == "hatch":
                    # Hatch lines take the edge color; lw=0 drops the
                    # border, which is drawn instead as solid edges.
                    axis.fill_between(grid, low, high, facecolor="none",
                                      edgecolor=to_rgba(color, hatch_alpha),
                                      hatch=hatch, lw=0)
                    for edge in (low, high):
                        axis.plot(grid, edge, color=color,
                                  lw=hatch_edge_width, ls="-")
                else:
                    for edge in (low, high):
                        axis.plot(grid, edge, color=color, lw=outline_width,
                                  ls=outline_style)
                axis.plot(grid, median, color=color, lw=2, ls=linestyle)
                grid_limits.extend([grid[0], grid[-1]])
            if significance_axes is not None:
                significance_axis = significance_axes[row_index, column]
                if significance_threshold is not None:
                    significance_axis.axhline(significance_threshold,
                                              color="red", lw=1, ls=":")
                first_grid, first = curve_draws[significance_pair[0]]
                second_grid, second = curve_draws[significance_pair[1]]
                if not np.array_equal(first_grid, second_grid):
                    # Compare on the first curve's grid.
                    second = np.array([
                        np.interp(first_grid, second_grid, draw)
                        for draw in second
                    ])
                if significance_method == "draws":
                    significance = _cdf_draw_difference_significance(
                        first, second, significance_seed
                    )
                else:
                    significance = _cdf_difference_significance(
                        first, second
                    )
                significance = np.abs(significance)
                significance_axis.plot(first_grid, significance,
                                       color=colors[model], lw=1.5)
                if np.isfinite(significance).any():
                    largest_significance = max(
                        largest_significance,
                        np.nanmax(significance),
                    )
                if column == 0:
                    significance_axis.set_ylabel(significance_ylabel)
            axis.set_xscale("log")
            axis.set_xlim(min(grid_limits), max(grid_limits))
            if xticks is not None:
                axis.xaxis.set_major_locator(FixedLocator(xticks))
                axis.xaxis.set_major_formatter(FuncFormatter(tick_formatter))
                axis.xaxis.set_minor_locator(NullLocator())
            axis.set_ylim(0, 1)
            axis.set_yticks(np.linspace(0, 1, 6))
            # The mass and separation formatters decide on rotation
            # themselves. Mass-ratio labels (e.g. "3.8e-4") are long, so they
            # are rotated whenever there are more than four of them.
            if (row_index == n_rows - 1 and xticks is not None and
                    getattr(tick_formatter, "rotate_labels",
                            len(xticks) > 4)):
                plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
            handles, labels = [], []
            if row_index == 0:
                # Band color identifies the model; shown once per column.
                handles.append(Patch(facecolor=colors[model],
                                     alpha=band_alpha, edgecolor="none"))
                labels.append(MODEL_TITLES.get(model, model))
            if column == 0:
                # Curve styles identify the samples; shown once per row.
                handles.extend(
                    _cdf_curve_handle(
                        band_styles[index % len(band_styles)],
                        linestyles[index % len(linestyles)],
                        show_line=len(set(linestyles[:len(row)])) > 1,
                        band_alpha=band_alpha, hatch=hatch,
                        hatch_alpha=hatch_alpha,
                        hatch_edge_width=hatch_edge_width,
                        outline_style=outline_style,
                        outline_width=outline_width,
                    )
                    for index in range(len(row))
                )
                labels.extend(curve["label"] for curve in row)
            if handles:
                axis.legend(handles, labels, loc=legend_loc,
                            fontsize=legend_fontsize)
            if row_index == 0 and title:
                axis.set_title(title.format(
                    model=MODEL_TITLES.get(model, model)
                ))
    if significance_axes is not None:
        # From zero, with no tick label at the bottom edge, so it never
        # collides with the main panels' top label; the threshold line
        # always stays inside the panels.
        limit = 1.15*max(largest_significance, significance_threshold or 0)
        significance_axes[0, 0].set_ylim(0, limit)
        significance_axes[0, 0].yaxis.set_major_locator(MaxNLocator(
            nbins=4, steps=[1, 2, 5, 10], prune="lower",
        ))
    _set_figure_axis_labels(figure, xlabel, ylabel, axes, row_labels,
                            top_axes=significance_axes)

    if output_file is None:
        first = rows[0][0]
        output_path = (
            results_dir / first["t1"] / "allstars" / first["t3"] / "plots" /
            f"{name}.png"
        )
    else:
        output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Hatch line widths are read from rcParams when the figure is drawn.
    with plt.rc_context({"hatch.linewidth": hatch_linewidth}):
        figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


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


def _piecewise_stack_occurrence_samples(path, stack_dim):
    """Return piecewise integrated-occurrence draws per stack bin.

    The result has one column per bin along ``stack_dim``; every tenth
    posterior draw is used.
    """
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
    areas = np.outer(np.diff(np.log10(m_edges)), np.diff(np.log10(a_edges)))
    occurrence = samples.reshape(-1, n_m, n_a)*areas[None, :, :]
    sum_axis = 1 if stack_dim == "a" else 2
    return occurrence.sum(axis=sum_axis)


def _piecewise_integrated_values(chain_dir, prefix, stack_dim, n_stack_bins,
                                 percent=False):
    """Collect per-stack integrated occurrence from the piecewise posterior.

    With ``percent``, values are companions per hundred stars, formatted
    with a trailing ``\\%``.
    """
    path = chain_dir / "chains_piecewise.npz"
    if not path.is_file():
        return []
    integrated = _piecewise_stack_occurrence_samples(path, stack_dim)
    if integrated.shape[1] != n_stack_bins:
        raise ValueError(f"piecewise stack-bin count does not match summary in {path}")
    if percent:
        integrated = 100*integrated
    values = []
    for bin_index in range(n_stack_bins):
        low, median, high = np.percentile(
            integrated[:, bin_index], [16, 50, 84]
        )
        name = (
            prefix + "PiecewiseIntOcc" +
            f"Bin{stack_dim.lower()}{_number_word(bin_index)}"
        )
        value = _format_parameter(median, low, high)
        values.append((name, value + r"\%" if percent else value))
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
    ``results_dir/paper_tables/variables.tex``.
    ``stack_bin`` is zero-based and defaults to the first fitted stack bin.
    By default, the generated table is saved in ``results_dir/paper_tables/``.
    ``output_file`` overrides that location.
    """
    results_dir = Path(results_dir)
    variables_path = results_dir / PAPER_TABLES_DIRNAME / "variables.tex"
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
    prefix = _experiment_prefix(t1, t2, t3)
    # Existing parameter variable names use forms such as "BinaZero".
    bin_suffix = "Bin" + stack_dim.lower() + _number_word(stack_bin)

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
        output_path = results_dir / PAPER_TABLES_DIRNAME / filename
    else:
        output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def make_appendix_parameter_table(
        results_dir, tier1_dirs, tier2_types, t3, stack_bin=0,
        stack_dim="a", caption="Fitted Model Parameters",
        label="tab:model_params_appendix", output_file=None):
    """Create a hierarchical appendix table for all smooth fitted models.

    Each experiment contains one row for every smooth model whose requested
    stack-bin chain exists.  Experiment-level cells are populated on the first
    row and left blank on the remaining model rows.  Rows follow the legacy
    experiment order: Tier 1 values in the supplied order, then Tier 2 types
    in the supplied order, with ``allstars`` represented once and every other
    type represented by its high row followed by its low row.  Numerical cells
    reference commands generated by :func:`make_variables`; this function
    neither reads nor validates ``variables.tex``.
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
                prefix = _experiment_prefix(tier1_name, tier2_dir, t3)
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
            results_dir / PAPER_TABLES_DIRNAME /
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
    """Return a shared variable-command name for a stellar subset.

    Piecewise integrated occurrence is named explicitly so it cannot be
    confused with an integrated occurrence from a parametric model.
    """
    mass, metallicity, age = levels
    prefix = _tier1_prefix(t1)
    prefix += _latex_token(Path(t3).name)
    if statistic == "IntOcc":
        statistic = "PiecewiseIntOcc"
    return (
        prefix + mass.capitalize() + "Mstar" +
        metallicity.capitalize() + "FeH" + age.capitalize() + statistic
    )


def _three_parameter_occurrence_command_name(
        t1, t3, levels, occurrence_model="piecewise",
        stack_dim="a", stack_bin=0):
    """Return the integrated-occurrence command for a selected model."""
    if occurrence_model == "piecewise":
        return _three_parameter_command_name(t1, t3, levels, "IntOcc")
    if occurrence_model not in _MODEL_PARAMETER_MACROS:
        raise ValueError(
            "occurrence_model must be 'piecewise' or one of "
            f"{tuple(_MODEL_PARAMETER_MACROS)}"
        )
    if stack_dim not in {"a", "m"}:
        raise ValueError("stack_dim must be 'a' or 'm'")
    if not isinstance(stack_bin, (int, np.integer)) or stack_bin < 0:
        raise ValueError("stack_bin must be a nonnegative integer")
    prefix = _three_parameter_command_name(t1, t3, levels, statistic="")
    model_macro = _MODEL_PARAMETER_MACROS[occurrence_model][0]
    return (
        prefix + model_macro + "IntOcc" +
        f"Bin{stack_dim}{_number_word(int(stack_bin))}"
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
        t1, t3, varied_parameter, fixed_levels,
        occurrence_model="piecewise", stack_dim="a", stack_bin=0):
    """Return the variable-command name for one posterior comparison."""
    prefix = _tier1_prefix(t1)
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
    name = prefix + varied_parameter + fixed_token
    if occurrence_model != "piecewise":
        if occurrence_model not in _MODEL_PARAMETER_MACROS:
            raise ValueError(
                "occurrence_model must be 'piecewise' or one of "
                f"{tuple(_MODEL_PARAMETER_MACROS)}"
            )
        if stack_dim not in {"a", "m"}:
            raise ValueError("stack_dim must be 'a' or 'm'")
        if not isinstance(stack_bin, (int, np.integer)) or stack_bin < 0:
            raise ValueError("stack_bin must be a nonnegative integer")
        name += (
            _MODEL_PARAMETER_MACROS[occurrence_model][0] +
            f"Bin{stack_dim}{_number_word(int(stack_bin))}"
        )
    return name + "Significance"


def _three_parameter_dynamic_range_command_name(
        t1, t3, varied_parameter, fixed_levels):
    """Return the variable-command name for one stellar dynamic range."""
    significance_name = _three_parameter_significance_command_name(
        t1, t3, varied_parameter, fixed_levels
    )
    return significance_name[:-len("Significance")] + "DynamicRange"


def _three_parameter_dynamic_ranges(catalog_path=None, cuts=None):
    """Calculate median stellar-property ratios for all table comparisons.

    The catalog is first restricted to the stellar-mass interval where the
    age-activity relation is valid, then divided using the mass, metallicity,
    and age cuts represented by the three-parameter Tier 2 directory names.
    Ratios are always reported as the larger median divided by the smaller
    median.  The metallicity medians are converted from dex to linear
    abundance first.
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
    required = {"Mstar", "feh", "age"}
    missing = sorted(required - set(catalog.columns))
    if missing:
        raise KeyError(f"{catalog_path} lacks columns {missing}")
    for column in required:
        catalog[column] = pd.to_numeric(catalog[column], errors="coerce")

    mstar_min = parameter_cuts["Mstar_min"]
    mstar_max = parameter_cuts["Mstar_max"]
    if not np.isfinite(mstar_min) or not np.isfinite(mstar_max):
        raise ValueError("Mstar_min and Mstar_max must be finite")
    if mstar_min > mstar_max:
        raise ValueError("Mstar_min cannot exceed Mstar_max")
    valid_age_activity_range = (
        (catalog["Mstar"] >= mstar_min) &
        (catalog["Mstar"] <= mstar_max)
    )

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
                valid_age_activity_range &
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


# Two-row column heads (top, bottom) for the median stellar properties in
# the original-form occurrence tables, keyed by catalog column.
_MEDIAN_COLUMN_HEADS = {
    "Mstar": ("Median", r"$M_{\star}$ ($M_{\odot}$)"),
    "feh": ("Median", r"$\text{[Fe/H]}$ (dex)"),
    "age": ("Median", "Age (Gyr)"),
}


def _two_row_tablehead(columns):
    """Return ``\\tablehead`` lines for two-row column heads.

    ``columns`` lists ``(top, bottom)`` pairs; single-row heads use an
    empty top, so their text sits on the bottom row with the units.
    """
    rows = []
    for row in zip(*columns):
        rows.append(" & ".join(rf"\colhead{{{text}}}" for text in row))
    return [r"\tablehead{", rows[0] + r" \\", rows[1], "}"]
_MEDIAN_DECIMALS = {"Mstar": 2, "feh": 2, "age": 1}


def _round_half_up(value, decimals):
    """Format ``value`` rounded half up, e.g. a median of 0.155 as 0.16.

    Rounding the shortest decimal form avoids binary artifacts that make
    ``f"{0.155:.2f}"`` give ``0.15``.
    """
    from decimal import ROUND_HALF_UP, Decimal

    quantum = Decimal(1).scaleb(-decimals)
    return str(Decimal(repr(float(value))).quantize(quantum, ROUND_HALF_UP))


def _subset_medians(subsets, columns, catalog_path=None, cuts=None,
                    mass_range=False, raw=False):
    """Return each stellar subset's median properties and star count.

    ``subsets`` maps a key to ``{column: level}``, where ``column`` is
    ``"Mstar"``, ``"feh"`` (levels ``"high"``/``"low"``), or ``"age"``
    (``"young"``/``"old"``); the subset is the catalog stars on that side of
    each cut in ``cuts`` (defaults: 1 solar mass, zero dex, 5 Gyr), within
    ``Mstar_min``--``Mstar_max`` when ``mass_range`` is set, as for the fits.
    Returns ``{key: ({column: formatted median}, star count)}`` for the
    requested ``columns``; with ``raw``, the medians are unrounded floats.
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
            raise ValueError(f"unknown cut names: {sorted(unknown)}")
        parameter_cuts.update(cuts)
    catalog = pd.read_csv(catalog_path)
    needed = set(columns) | {
        column for levels in subsets.values() for column in levels
    }
    missing = sorted(needed - set(catalog.columns))
    if missing:
        raise KeyError(f"{catalog_path} lacks columns {missing}")
    for column in needed | {"Mstar"}:
        catalog[column] = pd.to_numeric(catalog[column], errors="coerce")

    results = {}
    for key, levels in subsets.items():
        mask = pd.Series(True, index=catalog.index)
        if mass_range:
            mask &= ((catalog["Mstar"] >= parameter_cuts["Mstar_min"]) &
                     (catalog["Mstar"] <= parameter_cuts["Mstar_max"]))
        for column, level in levels.items():
            values, cut = catalog[column], parameter_cuts[column]
            if column == "age":
                mask &= values < cut if level == "young" else values >= cut
            else:
                mask &= values > cut if level == "high" else values <= cut
        medians = {}
        for column in columns:
            values = catalog.loc[mask, column].dropna().to_numpy()
            if values.size == 0:
                raise ValueError(
                    f"no finite {column} values for subset {key} in "
                    f"{catalog_path}"
                )
            median = float(np.median(values))
            medians[column] = median if raw else _round_half_up(
                median, _MEDIAN_DECIMALS[column]
            )
        results[key] = (medians, int(mask.sum()))
    return results


def _median_ratios(pairs, catalog_path=None, cuts=None, mass_range=False):
    """Return high/low ratios of median stellar mass and metallicity.

    ``pairs`` maps a key to ``(low levels, high levels)``, each in the
    ``{column: level}`` form of :func:`_subset_medians`.  The [Fe/H] ratio
    compares linear abundances, ``10**(high - low)`` of the dex medians.
    Returns ``{key: (mass ratio, metallicity ratio)}``, rounded half up to
    one decimal.
    """
    subsets = {}
    for key, (low, high) in pairs.items():
        subsets[key, "low"], subsets[key, "high"] = low, high
    medians = _subset_medians(subsets, ("Mstar", "feh"), catalog_path, cuts,
                              mass_range, raw=True)
    return {
        key: _format_median_ratios(medians[key, "high"][0],
                                   medians[key, "low"][0])
        for key in pairs
    }


def _format_median_ratios(first, second):
    """Return (mass ratio, metallicity ratio) of two samples' raw medians.

    ``first`` and ``second`` map ``"Mstar"`` and ``"feh"`` to medians; the
    metallicity ratio compares linear abundances, ``10**(first - second)``.
    """
    return (
        _round_half_up(first["Mstar"]/second["Mstar"], 1),
        _round_half_up(10**(first["feh"] - second["feh"]), 1),
    )


def _check_subset_counts(subset_medians, table_values, label):
    """Raise if catalog subsets do not hold the stars the fits used."""
    mismatched = {
        key: (count, table_values[key]["Nstars"])
        for key, (_, count) in subset_medians.items()
        if str(count) != str(table_values[key]["Nstars"])
    }
    if mismatched:
        raise ValueError(
            f"{label} catalog subsets do not match the fitted star counts "
            f"(catalog, fit): {mismatched}; check the cuts and catalog"
        )


# Median stellar properties written to variables.tex: catalog column ->
# command suffix.
_MEDIAN_VARIABLES = {"Mstar": "MedianMstar", "feh": "MedianFeH",
                     "age": "MedianAge"}


def _sample_median_values(sample_catalog, query, nstars, sample):
    """Return ``{suffix: value}`` medians for one sample of the catalog.

    ``query`` selects the sample from ``sample_catalog`` (a DataFrame) the
    way the fits did (``None`` keeps every star); its star count must equal
    the fit's ``nstars``.  Median age is included only when every star lies
    in the mass range where the age-activity relation holds.
    """
    selected = sample_catalog if query is None else sample_catalog.query(query)
    if len(selected) != int(nstars):
        raise ValueError(
            f"sample {sample!r} selects {len(selected)} catalog stars, but "
            f"its fit used {int(nstars)}; check sample_queries and the catalog"
        )
    masses = selected["Mstar"]
    in_age_range = bool(
        (masses >= _THREE_PARAMETER_DEFAULT_CUTS["Mstar_min"]).all() and
        (masses <= _THREE_PARAMETER_DEFAULT_CUTS["Mstar_max"]).all()
    )
    values = {}
    for column, suffix in _MEDIAN_VARIABLES.items():
        if column == "age" and not in_age_range:
            continue
        finite = selected[column].dropna()
        if len(finite):
            values[suffix] = _round_half_up(
                np.median(finite), _MEDIAN_DECIMALS[column]
            )
    return values


def _two_parameter_levels(tier2_dir):
    """Extract mass and metallicity levels from a subset directory."""
    match = re.fullmatch(
        r"(high|low)Mstar(high|low)FeH", str(tier2_dir)
    )
    if match is None:
        raise ValueError(
            f"invalid two-parameter Tier 2 directory: {tier2_dir!r}"
        )
    return match.groups()


def _two_parameter_command_name(t1, t3, levels, statistic="IntOcc"):
    """Return a variable-command name for a mass-metallicity subset."""
    mass, metallicity = levels
    prefix = _tier1_prefix(t1)
    prefix += _latex_token(Path(t3).name)
    if statistic == "IntOcc":
        statistic = "PiecewiseIntOcc"
    return (
        prefix + mass.capitalize() + "Mstar" +
        metallicity.capitalize() + "FeH" + statistic
    )


def _two_parameter_comparisons():
    """Return the four comparisons shown in the reordered table."""
    comparisons = []
    for metallicity in ("high", "low"):
        comparisons.append((
            "Mass", metallicity,
            ("low", metallicity), ("high", metallicity),
        ))
    for mass in ("high", "low"):
        comparisons.append((
            "FeH", mass,
            (mass, "low"), (mass, "high"),
        ))
    return comparisons


def _two_parameter_comparison_command_name(
        t1, t3, varied_parameter, fixed_level, statistic):
    """Return a command name for a two-parameter comparison statistic."""
    prefix = _tier1_prefix(t1)
    prefix += _latex_token(Path(t3).name)
    if varied_parameter == "Mass":
        fixed_token = fixed_level.capitalize() + "FeH"
    elif varied_parameter == "FeH":
        fixed_token = fixed_level.capitalize() + "Mstar"
    else:
        raise ValueError("varied_parameter must be 'Mass' or 'FeH'")
    return prefix + varied_parameter + fixed_token + statistic


def _two_parameter_dynamic_ranges(catalog_path=None, cuts=None):
    """Calculate mass and metallicity median ratios without a mass-range cut."""
    import pandas as pd

    if catalog_path is None:
        catalog_path = (
            Path(__file__).resolve().parent / "cls_files" /
            "cls_all_stars_all_params.csv"
        )
    else:
        catalog_path = Path(catalog_path)
    parameter_cuts = {
        "Mstar": _THREE_PARAMETER_DEFAULT_CUTS["Mstar"],
        "feh": _THREE_PARAMETER_DEFAULT_CUTS["feh"],
    }
    if cuts is not None:
        unknown = set(cuts) - set(parameter_cuts)
        if unknown:
            raise ValueError(
                f"unknown two-parameter cut names: {sorted(unknown)}"
            )
        parameter_cuts.update(cuts)

    catalog = pd.read_csv(catalog_path)
    missing = sorted(set(parameter_cuts) - set(catalog.columns))
    if missing:
        raise KeyError(f"{catalog_path} lacks columns {missing}")
    for column in parameter_cuts:
        catalog[column] = pd.to_numeric(catalog[column], errors="coerce")

    def level_mask(column, level):
        values = catalog[column]
        cut = parameter_cuts[column]
        return values > cut if level == "high" else values <= cut

    columns = {"Mass": "Mstar", "FeH": "feh"}
    dynamic_ranges = {}
    for varied_parameter, fixed_level, low_levels, high_levels in (
            _two_parameter_comparisons()):
        varied_column = columns[varied_parameter]
        medians = []
        for mass, metallicity in (low_levels, high_levels):
            mask = (
                level_mask("Mstar", mass) &
                level_mask("feh", metallicity)
            )
            values = catalog.loc[mask, varied_column].dropna().to_numpy()
            if values.size == 0:
                raise ValueError(
                    f"no finite {varied_column} values for levels "
                    f"{(mass, metallicity)} in {catalog_path}"
                )
            medians.append(float(np.median(values)))
        if varied_parameter == "FeH":
            medians = [10**value for value in medians]
        lower, upper = sorted(medians)
        if lower <= 0:
            raise ValueError(
                f"dynamic-range medians must be positive; got {medians} for "
                f"{varied_parameter} at fixed level {fixed_level}"
            )
        dynamic_ranges[(varied_parameter, fixed_level)] = upper/lower
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


def _parametric_occurrence_samples(
        result_dir, model_name, stack_bin=0):
    """Load and integrate one parametric model's posterior draws."""
    if model_name not in _MODEL_PARAMETER_MACROS:
        raise ValueError(
            f"model_name must be one of {tuple(_MODEL_PARAMETER_MACROS)}"
        )
    if not isinstance(stack_bin, (int, np.integer)) or stack_bin < 0:
        raise ValueError("stack_bin must be a nonnegative integer")
    result_dir = Path(result_dir)
    chain_path = (
        result_dir / "saved_chains" /
        f"chains_{model_name}_bin{int(stack_bin)}.npz"
    )
    if not chain_path.is_file():
        raise FileNotFoundError(f"parametric chain not found: {chain_path}")
    with np.load(chain_path) as chain:
        key = "flat_chains" if "flat_chains" in chain else "chains"
        if key not in chain:
            raise KeyError(
                f"{chain_path} must contain 'flat_chains' or 'chains'"
            )
        if "model_bounds" not in chain or "stack_bounds" not in chain:
            raise KeyError(
                f"{chain_path} must contain 'model_bounds' and 'stack_bounds'"
            )
        samples = np.asarray(chain[key], dtype=float)
        model_bounds = np.asarray(chain["model_bounds"], dtype=float)
        stack_bounds = np.asarray(chain["stack_bounds"], dtype=float)
    samples = samples.reshape(-1, samples.shape[-1])
    expected_parameters = len(_MODEL_PARAMETER_MACROS[model_name][1])
    if samples.shape[1] != expected_parameters:
        raise ValueError(
            f"parameter count in {chain_path} does not match {model_name}"
        )
    if samples.shape[0] < 2 or not np.isfinite(samples).all():
        raise ValueError(
            f"parameter samples in {chain_path} must contain at least two "
            "finite draws"
        )
    samples = samples[::10]
    if samples.shape[0] > 10000:
        indices = np.linspace(
            0, samples.shape[0] - 1, 10000, dtype=int
        )
        samples = samples[indices]
    integrated = _integrated_occurrence_samples(
        model_name, samples, model_bounds, stack_bounds
    )
    if integrated.size < 2 or not np.isfinite(integrated).all():
        raise ValueError(
            f"integrated occurrence samples in {chain_path} must contain "
            "at least two finite values"
        )
    return integrated


def _model_occurrence_samples(
        result_dir, occurrence_model="piecewise", stack_bin=0):
    """Load integrated-occurrence draws for the requested model."""
    if occurrence_model == "piecewise":
        return _three_parameter_occurrence_samples(Path(result_dir))
    return _parametric_occurrence_samples(
        result_dir, occurrence_model, stack_bin
    )


def _format_occurrence_samples(samples):
    """Format an integrated-occurrence posterior for a LaTeX table."""
    low, median, high = np.percentile(samples, [16, 50, 84])
    return _format_parameter(median, low, high)


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
        three_parameter_cuts=None, occurrence_model="piecewise",
        occurrence_stack_dim="a", occurrence_stack_bin=0):
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
    samples used for the dynamic-range column.  ``occurrence_model`` selects
    which fitted function supplies both the integrated occurrence rates and
    their comparison significances.  Parametric models use the chain selected
    by ``occurrence_stack_bin``; ``occurrence_stack_dim`` determines the
    corresponding LaTeX command suffix.
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
    if (occurrence_model != "piecewise" and
            occurrence_model not in _MODEL_PARAMETER_MACROS):
        raise ValueError(
            "occurrence_model must be 'piecewise' or one of "
            f"{tuple(_MODEL_PARAMETER_MACROS)}"
        )
    if occurrence_stack_dim not in {"a", "m"}:
        raise ValueError("occurrence_stack_dim must be 'a' or 'm'")
    if (not isinstance(occurrence_stack_bin, (int, np.integer)) or
            occurrence_stack_bin < 0):
        raise ValueError("occurrence_stack_bin must be a nonnegative integer")

    table_values = {}
    for tier2_dir in tier2_dirs:
        levels = _three_parameter_levels(tier2_dir)
        if levels in table_values:
            raise ValueError(f"duplicate three-parameter subset: {levels}")
        if use_latex_variables:
            table_values[levels] = {
                statistic: rf"\{_three_parameter_command_name(t1, t3, levels, statistic)}"
                for statistic in ("Nstars", "Neff", "AvgCompl")
            }
            occurrence_name = _three_parameter_occurrence_command_name(
                t1, t3, levels, occurrence_model,
                occurrence_stack_dim, occurrence_stack_bin,
            )
            table_values[levels]["IntOcc"] = rf"\{occurrence_name}"
        else:
            result_dir = results_dir / t1 / tier2_dir / t3
            statistics = _three_parameter_statistics(result_dir)
            table_values[levels] = dict(statistics)
            if occurrence_model == "piecewise":
                formatted_occurrence = statistics["IntOcc"]
            else:
                occurrence_samples = _model_occurrence_samples(
                    result_dir, occurrence_model, occurrence_stack_bin
                )
                formatted_occurrence = _format_occurrence_samples(
                    occurrence_samples
                )
            table_values[levels]["IntOcc"] = f"${formatted_occurrence}$"
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
            posterior_samples[levels] = _model_occurrence_samples(
                results_dir / t1 / tier2_dir / t3,
                occurrence_model, occurrence_stack_bin,
            )
    for varied_parameter, fixed_levels, low_levels, high_levels in (
            _three_parameter_comparisons()):
        key = (varied_parameter, fixed_levels)
        command_name = _three_parameter_significance_command_name(
            t1, t3, varied_parameter, fixed_levels,
            occurrence_model, occurrence_stack_dim, occurrence_stack_bin,
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
        r"\colhead{High/Low}",
        r"}",
        r"\startdata",
    ]

    def add_block(fixed_one, fixed_two, low_heading, high_heading, rows):
        lines.append(
            rf"\textbf{{{fixed_one}}} & \textbf{{{fixed_two}}} & "
            rf"\textbf{{{low_heading}}} & \textbf{{{high_heading}}} & "
            r" & \\"
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
        r"$\text{[Fe/H]}$", "Age", "Low Mass", "High Mass",
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
        "Mass", "Age", r"Low $\text{[Fe/H]}$", r"High $\text{[Fe/H]}$",
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
        "Mass", r"$\text{[Fe/H]}$", "Young", "Old",
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
            results_dir / PAPER_TABLES_DIRNAME /
            f"three_parameter_OR_reordered_{t1}_{t3}.tex"
        )
    else:
        reordered_output_path = Path(reordered_output_file)
    reordered_output_path.parent.mkdir(parents=True, exist_ok=True)
    reordered_output_path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    median_columns = ("Mstar", "feh", "age")
    if use_latex_variables:
        subset_medians = {
            levels: ({
                column: "\\" + _three_parameter_command_name(
                    t1, t3, levels, _MEDIAN_VARIABLES[column])
                for column in median_columns
            }, None)
            for levels in table_values
        }
    else:
        subset_medians = _subset_medians(
            {
                levels: {"Mstar": levels[0], "feh": levels[1],
                         "age": levels[2]}
                for levels in table_values
            },
            median_columns, stellar_catalog_path, three_parameter_cuts,
            mass_range=True,
        )
        _check_subset_counts(subset_medians, table_values, "three-parameter")
    original_lines = [
        r"\begin{deluxetable*}{lccccccccc}",
        rf"\tablecaption{{{original_caption}}}",
        rf"\label{{{original_label}}}",
        *_two_row_tablehead([
            ("", "Mass"), ("", r"$\text{[Fe/H]}$"), ("", "Age"),
            *(_MEDIAN_COLUMN_HEADS[column] for column in median_columns),
            ("", r"$N_{\star}$"), ("", r"$N_{\mathrm{eff}}$"),
            ("Average", "Completeness"), ("", "Occurrence Rate"),
        ]),
        r"\startdata",
    ]
    for tier2_dir in tier2_dirs:
        levels = _three_parameter_levels(tier2_dir)
        mass, metallicity, age = levels
        values = table_values[levels]
        medians = subset_medians[levels][0]
        original_lines.append(
            f"{mass} & {metallicity} & {age} & "
            + "".join(f"${medians[column]}$ & " for column in median_columns)
            + f"{values['Nstars']} & {values['Neff']} & "
            f"{values['AvgCompl']} & {values['IntOcc']} " + r"\\"
        )
    original_lines.extend([
        r"\enddata",
        r"\end{deluxetable*}",
    ])
    if original_output_file is None:
        original_output_path = (
            results_dir / PAPER_TABLES_DIRNAME /
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


def make_two_parameter_tables(
        results_dir, t1="mtrue", t3="stellar2params", tier2_dirs=None,
        reordered_caption=(
            "Occurrence Rates with One Stellar Parameter Held Fixed"
        ),
        reordered_label="tab:two_param_OR_reordered",
        original_caption="Occurrence Rates by Stellar Mass and Metallicity",
        original_label="tab:two_param_OR",
        reordered_output_file=None, original_output_file=None,
        use_latex_variables=True, stellar_catalog_path=None,
        two_parameter_cuts=None):
    """Create reordered and original-form mass-metallicity tables.

    The four subsets span low/high stellar mass and low/high metallicity.
    Unlike the age-activity analysis, dynamic ranges use the complete stellar
    catalog without restricting stellar mass to 0.82--1.21 solar masses.
    Set ``use_latex_variables=False`` to embed values from the saved results;
    otherwise both tables contain the command names that
    :func:`make_variables` writes when given ``two_parameter_t3``.  Embedded significances follow the variables'
    rules: posterior when the draws resolve it, quadrature otherwise.
    """
    results_dir = Path(results_dir)
    if tier2_dirs is None:
        tier2_dirs = _TWO_PARAMETER_TIER2_DIRS
    else:
        tier2_dirs = tuple(tier2_dirs)
    if len(tier2_dirs) != 4:
        raise ValueError("tier2_dirs must contain the four stellar subsets")
    if not isinstance(use_latex_variables, (bool, np.bool_)):
        raise TypeError("use_latex_variables must be a boolean")

    table_values = {}
    for tier2_dir in tier2_dirs:
        levels = _two_parameter_levels(tier2_dir)
        if levels in table_values:
            raise ValueError(f"duplicate two-parameter subset: {levels}")
        if use_latex_variables:
            table_values[levels] = {
                statistic: rf"\{_two_parameter_command_name(t1, t3, levels, statistic)}"
                for statistic in ("Nstars", "Neff", "AvgCompl", "IntOcc")
            }
        else:
            result_dir = results_dir / t1 / tier2_dir / t3
            statistics = _three_parameter_statistics(result_dir)
            table_values[levels] = dict(statistics)
            table_values[levels]["IntOcc"] = f"${statistics['IntOcc']}$"
    expected = {
        (mass, metallicity)
        for mass in ("high", "low")
        for metallicity in ("high", "low")
    }
    if set(table_values) != expected:
        missing = sorted(expected - set(table_values))
        raise ValueError(f"two-parameter subsets are incomplete; missing {missing}")

    significance_values = {}
    posterior_samples = {}
    # High/low ratios of the median stellar mass and metallicity between
    # the two subsets of each comparison: variables.tex commands, or values
    # from the stellar catalog
    if use_latex_variables:
        ratio_values = {
            (varied, fixed): tuple(
                "\\" + _two_parameter_comparison_command_name(
                    t1, t3, varied, fixed, suffix)
                for suffix in ("MedianMstarRatio", "MedianFeHRatio")
            )
            for varied, fixed, _, _ in _two_parameter_comparisons()
        }
    else:
        ratio_values = _median_ratios(
            {
                (varied, fixed): (
                    {"Mstar": low[0], "feh": low[1]},
                    {"Mstar": high[0], "feh": high[1]},
                )
                for varied, fixed, low, high in _two_parameter_comparisons()
            },
            stellar_catalog_path, two_parameter_cuts,
        )
    if not use_latex_variables:
        for tier2_dir in tier2_dirs:
            levels = _two_parameter_levels(tier2_dir)
            posterior_samples[levels] = _three_parameter_occurrence_samples(
                results_dir / t1 / tier2_dir / t3
            )
    for varied_parameter, fixed_level, low_levels, high_levels in (
            _two_parameter_comparisons()):
        key = (varied_parameter, fixed_level)
        significance_name = _two_parameter_comparison_command_name(
            t1, t3, varied_parameter, fixed_level, "Significance"
        )
        if use_latex_variables:
            significance_values[key] = rf"\{significance_name}"
        else:
            # Same rules as the variables: beyond what the draws resolve,
            # fall back to the quadrature significance.
            significance, _ = _comparison_significance(
                posterior_samples[high_levels],
                posterior_samples[low_levels],
            )
            significance_values[key] = f"${significance}$"

    reordered_lines = [
        r"\begin{deluxetable*}{cccccc}",
        rf"\tablecaption{{{reordered_caption}}}",
        rf"\label{{{reordered_label}}}",
        r"\tablehead{",
        r"\colhead{} & \multicolumn{2}{c}{} & \colhead{} & "
        r"\colhead{High/Low} & \colhead{High/Low} \\",
        r"\colhead{Fixed Parameter} & \multicolumn{2}{c}{Occurrence Rate} & "
        r"\colhead{Significance} & \colhead{$M_{\star}$} & "
        r"\colhead{$\text{[Fe/H]}$}",
        r"}",
        r"\startdata",
    ]

    def add_block(fixed_parameter, low_heading, high_heading, rows):
        reordered_lines.append(
            rf"\textbf{{{fixed_parameter}}} & "
            rf"\textbf{{{low_heading}}} & \textbf{{{high_heading}}} & "
            r" &  & \\"
        )
        reordered_lines.append(r"\hline")
        for fixed_level, low_value, high_value, significance, ratios in rows:
            reordered_lines.append(
                rf"{fixed_level} & {low_value} & {high_value} & "
                rf"{significance} & {ratios[0]} & {ratios[1]} \\"
            )
        reordered_lines.append(r"\hline")

    add_block(
        r"$\text{[Fe/H]}$", "Low Mass", "High Mass",
        [
            (metallicity,
             table_values[("low", metallicity)]["IntOcc"],
             table_values[("high", metallicity)]["IntOcc"],
             significance_values[("Mass", metallicity)],
             ratio_values[("Mass", metallicity)])
            for metallicity in ("high", "low")
        ],
    )
    add_block(
        "Mass", r"Low $\text{[Fe/H]}$", r"High $\text{[Fe/H]}$",
        [
            (mass,
             table_values[(mass, "low")]["IntOcc"],
             table_values[(mass, "high")]["IntOcc"],
             significance_values[("FeH", mass)],
             ratio_values[("FeH", mass)])
            for mass in ("high", "low")
        ],
    )
    reordered_lines.extend([
        r"\enddata",
        r"\end{deluxetable*}",
    ])

    if reordered_output_file is None:
        reordered_output_path = (
            results_dir / PAPER_TABLES_DIRNAME /
            f"two_parameter_OR_reordered_{t1}_{t3}.tex"
        )
    else:
        reordered_output_path = Path(reordered_output_file)
    reordered_output_path.parent.mkdir(parents=True, exist_ok=True)
    reordered_output_path.write_text(
        "\n".join(reordered_lines) + "\n", encoding="utf-8"
    )

    median_columns = ("Mstar", "feh")
    if use_latex_variables:
        subset_medians = {
            levels: ({
                column: "\\" + _two_parameter_command_name(
                    t1, t3, levels, _MEDIAN_VARIABLES[column])
                for column in median_columns
            }, None)
            for levels in table_values
        }
    else:
        subset_medians = _subset_medians(
            {levels: {"Mstar": levels[0], "feh": levels[1]}
             for levels in table_values},
            median_columns, stellar_catalog_path, two_parameter_cuts,
        )
        _check_subset_counts(subset_medians, table_values, "two-parameter")
    original_lines = [
        r"\begin{deluxetable*}{lccccccc}",
        rf"\tablecaption{{{original_caption}}}",
        rf"\label{{{original_label}}}",
        *_two_row_tablehead([
            ("", "Mass"), ("", r"$\text{[Fe/H]}$"),
            *(_MEDIAN_COLUMN_HEADS[column] for column in median_columns),
            ("", r"$N_{\star}$"), ("", r"$N_{\mathrm{eff}}$"),
            ("Average", "Completeness"), ("", "Occurrence Rate"),
        ]),
        r"\startdata",
    ]
    for tier2_dir in tier2_dirs:
        mass, metallicity = _two_parameter_levels(tier2_dir)
        values = table_values[(mass, metallicity)]
        medians = subset_medians[(mass, metallicity)][0]
        original_lines.append(
            f"{mass} & {metallicity} & "
            + "".join(f"${medians[column]}$ & " for column in median_columns)
            + f"{values['Nstars']} & {values['Neff']} & "
            f"{values['AvgCompl']} & {values['IntOcc']} " + r"\\"
        )
    original_lines.extend([
        r"\enddata",
        r"\end{deluxetable*}",
    ])
    if original_output_file is None:
        original_output_path = (
            results_dir / PAPER_TABLES_DIRNAME /
            f"two_parameter_OR_{t1}_{t3}.tex"
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


def _table_note_lines(note):
    """Return the lines of an AASTeX table note, or none without a note.

    The note is set after ``\\enddata`` as ``\\tablenotetext{}`` with a bold
    "Note." lead-in, matching the paper's hand-written tables.
    """
    if note is None or not str(note).strip():
        return []
    return ["", r"\tablenotetext{}{%", r"\textbf{Note.} " + str(note).strip(),
            "}"]


# Table labels and stellar-catalog columns for the one-parameter tables,
# keyed by Tier 2 type. [Fe/H] is set in math so a row starting with it is
# not read as the optional argument of the previous row's \\.
_ONE_PARAMETER_TYPES = {
    "Mstar": ("Mass", "Mstar"),
    "FeH": (r"$\text{[Fe/H]}$", "feh"),
}


def make_one_parameter_table(
        results_dir, t1="mtrue", t3="paper_bounds",
        tier2_types=("Mstar", "FeH"), stack_dim="a", stack_bin=0,
        caption="Occurrence Rates by Stellar Mass or Metallicity",
        label="tab:one_param_OR", output_file=None,
        use_latex_variables=True, stellar_catalog_path=None,
        one_parameter_cuts=None, note=None):
    """Create one table comparing the high and low samples of single splits.

    Each Tier 2 type in ``tier2_types`` (``"Mstar"``, ``"FeH"``) contributes
    its ``high`` and ``low`` samples from the one-parameter fits in
    ``t1/<high|low><type>/t3``, so no extra runs are needed.  Each sample's
    row lists its star count, effective companions, average completeness,
    and integrated piecewise occurrence.  The significance of the pair's
    occurrence difference and the high/low ratio of the parameter's median
    are each set once per pair, as is the parameter's name, vertically
    centered across its two rows
    with ``\\multirow`` (the paper must load the ``multirow`` package).
    Unlike :func:`make_two_parameter_tables`, no separate reordered table
    is made, since each parameter has only one comparison.

    By default the cells reference the commands :func:`make_variables`
    already emits for these samples (e.g. ``\\McHighMstarPaperBoundsNstars``
    and ``\\McMstarPaperBoundsPiecewiseIntOccSignificanceBinaZero``), with
    occurrence from stack bin ``stack_bin`` along ``stack_dim``.  Set
    ``use_latex_variables=False`` to embed numbers from the saved results
    instead; significances then follow the same rules as the variables.
    The median ratios always come from the stellar catalog, using
    ``stellar_catalog_path`` and ``one_parameter_cuts``.  ``note`` is LaTeX
    text for a table note below the data; it is omitted when ``None``.
    """
    results_dir = Path(results_dir)
    tier2_types = list(tier2_types)
    unknown = sorted(set(tier2_types) - set(_ONE_PARAMETER_TYPES))
    if not tier2_types or unknown:
        raise ValueError(
            f"tier2_types must list types from {sorted(_ONE_PARAMETER_TYPES)}"
            f"; got {tier2_types}"
        )
    if len(set(tier2_types)) != len(tier2_types):
        raise ValueError("tier2_types contains duplicates")
    if not isinstance(use_latex_variables, (bool, np.bool_)):
        raise TypeError("use_latex_variables must be a boolean")
    if stack_dim not in {"a", "m"}:
        raise ValueError("stack_dim must be 'a' or 'm'")
    if (not isinstance(stack_bin, (int, np.integer)) or stack_bin < 0):
        raise ValueError("stack_bin must be a nonnegative integer")
    bin_suffix = f"Bin{stack_dim}{_number_word(stack_bin)}"

    table_values = {}
    significance_values = {}
    for tier2_type in tier2_types:
        samples = {}
        for level in ("high", "low"):
            tier2_dir = f"{level}{tier2_type}"
            result_dir = results_dir / t1 / tier2_dir / t3
            if use_latex_variables:
                prefix = _experiment_prefix(t1, tier2_dir, t3)
                values = {
                    statistic: rf"\{prefix}{statistic}"
                    for statistic in ("Nstars", "Neff", "AvgCompl")
                }
                values["IntOcc"] = rf"\{prefix}PiecewiseIntOcc{bin_suffix}"
            else:
                values = dict(_three_parameter_statistics(result_dir))
                chain_path = result_dir / "saved_chains" / "chains_piecewise.npz"
                if not chain_path.is_file():
                    raise FileNotFoundError(
                        f"piecewise chain not found: {chain_path}"
                    )
                occurrence = _piecewise_stack_occurrence_samples(
                    chain_path, stack_dim
                )
                if stack_bin >= occurrence.shape[1]:
                    raise ValueError(
                        f"stack_bin {stack_bin} exceeds the "
                        f"{occurrence.shape[1]} bins in {chain_path}"
                    )
                samples[level] = occurrence[:, stack_bin]
                values["IntOcc"] = (
                    f"${_format_occurrence_samples(samples[level])}$"
                )
            table_values[(tier2_type, level)] = values
        if use_latex_variables:
            significance_values[tier2_type] = (
                "\\" + _tier1_prefix(t1) + _latex_token(tier2_type) +
                _latex_token(Path(t3).name) + "PiecewiseIntOccSignificance" +
                bin_suffix
            )
        else:
            significance, _ = _comparison_significance(
                samples["high"], samples["low"]
            )
            significance_values[tier2_type] = f"${significance}$"
    # High/low ratios of the median stellar mass and metallicity of each
    # pair: variables.tex commands, or values from the stellar catalog
    if use_latex_variables:
        ratios = {
            tier2_type: tuple(
                "\\" + _tier1_prefix(t1) + _latex_token(tier2_type) +
                _latex_token(Path(t3).name) + suffix
                for suffix in ("MedianMstarRatio", "MedianFeHRatio")
            )
            for tier2_type in tier2_types
        }
    else:
        ratios = _median_ratios(
            {
                tier2_type: tuple(
                    {_ONE_PARAMETER_TYPES[tier2_type][1]: level}
                    for level in ("low", "high")
                )
                for tier2_type in tier2_types
            },
            stellar_catalog_path, one_parameter_cuts,
        )

    median_columns = ("Mstar", "feh")
    if use_latex_variables:
        subset_medians = {
            (tier2_type, level): ({
                column: "\\" + _experiment_prefix(
                    t1, f"{level}{tier2_type}", t3
                ) + _MEDIAN_VARIABLES[column]
                for column in median_columns
            }, None)
            for tier2_type, level in table_values
        }
    else:
        subset_medians = _subset_medians(
            {key: {_ONE_PARAMETER_TYPES[key[0]][1]: key[1]}
             for key in table_values},
            median_columns, stellar_catalog_path, one_parameter_cuts,
        )
        _check_subset_counts(subset_medians, table_values, "one-parameter")
    lines = [
        r"\begin{deluxetable*}{lcccccccccc}",
        rf"\tablecaption{{{caption}}}",
        rf"\label{{{label}}}",
        *_two_row_tablehead([
            ("", "Parameter"), ("", "Sample"),
            *(_MEDIAN_COLUMN_HEADS[column] for column in median_columns),
            ("", r"$N_{\star}$"), ("", r"$N_{\mathrm{eff}}$"),
            ("Average", "Completeness"), ("", "Occurrence Rate"),
            ("", "Significance"), ("High/Low", r"$M_{\star}$"),
            ("High/Low", r"$\text{[Fe/H]}$"),
        ]),
        r"\startdata",
    ]
    for tier2_type in tier2_types:
        for level in ("high", "low"):
            values = table_values[(tier2_type, level)]
            if level == "high":
                # Set once per pair, centered across its two rows
                parameter_cell = (
                    rf"\multirow{{2}}{{*}}"
                    rf"{{{_ONE_PARAMETER_TYPES[tier2_type][0]}}}"
                )
                pair_cells = " & ".join(
                    rf"\multirow{{2}}{{*}}{{{cell}}}"
                    for cell in (significance_values[tier2_type],
                                 *ratios[tier2_type])
                )
            else:
                parameter_cell = ""
                pair_cells = " &  & "
            medians = subset_medians[(tier2_type, level)][0]
            lines.append(
                f"{parameter_cell} & {level} & "
                + "".join(f"${medians[column]}$ & "
                          for column in median_columns)
                + f"{values['Nstars']} & {values['Neff']} & "
                f"{values['AvgCompl']} & {values['IntOcc']} & "
                f"{pair_cells} " + r"\\"
            )
    lines.append(r"\enddata")
    lines.extend(_table_note_lines(note))
    lines.append(r"\end{deluxetable*}")
    if output_file is None:
        output_path = (
            results_dir / PAPER_TABLES_DIRNAME /
            f"one_parameter_OR_{t1}_{Path(t3).name}.tex"
        )
    else:
        output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def calculate_all_delta_bics(
        results_dir, tier1_dirs, tier2_types, tier3_dirs, stack_dim="a",
        standalone_tier2_dirs=(), tier3_stack_dims=None):
    """Calculate and save delta-BIC values for every requested results folder.

    The arguments and Tier 2 expansion rules match :func:`make_variables`.
    Results are returned as a dictionary keyed by relative results paths and
    are also saved to ``results_dir/paper_tables/delta_bics.json``.  Folders
    containing no recognized parametric chains are represented by an empty
    dictionary; requested folders that do not exist are left out.
    """
    results_dir = Path(results_dir)
    tier1_dirs = list(tier1_dirs)
    tier2_types = list(tier2_types)
    tier3_dirs = list(tier3_dirs)
    if not tier1_dirs or not tier2_types or not tier3_dirs:
        raise ValueError("tier1_dirs, tier2_types, and tier3_dirs cannot be empty")
    if stack_dim not in {"a", "m"}:
        raise ValueError("stack_dim must be 'a' or 'm'")

    tier2_dirs = _requested_tier2_dirs(tier2_types, standalone_tier2_dirs)
    all_delta_bics = {}
    for tier1_dir in tier1_dirs:
        for tier2_dir in tier2_dirs:
                for tier3_dir in tier3_dirs:
                    result_dir = results_dir / tier1_dir / tier2_dir / tier3_dir
                    if not result_dir.is_dir():
                        continue
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
                        _tier3_stack_dim(tier3_dir, stack_dim,
                                         tier3_stack_dims),
                    )
                    all_delta_bics[key] = {
                        model_name: np.asarray(values, dtype=float).tolist()
                        for model_name, values in comparisons.items()
                    }

    paper_tables_dir = results_dir / PAPER_TABLES_DIRNAME
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "stack_dim": stack_dim,
        "delta_bic_convention": "BIC_flat - BIC_model",
        "results": all_delta_bics,
    }
    (paper_tables_dir / DELTA_BIC_FILENAME).write_text(
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


# How each model parameter is compared between samples: rates by their ratio,
# log10 locations by their difference in dex and the corresponding factor,
# and everything else (widths, slopes) by its difference alone.
_RATE_PARAMETERS = {"COne", "CTwo", "A", "C", "CLow", "CHigh"}
_LOCATION_PARAMETERS = {"Center", "Mu", "BPOne", "BPTwo", "LogBreak"}


def _format_two_figures(value):
    """Format a positive ratio or factor to two significant figures."""
    return f"{float(f'{value:.2g}'):g}"


def _comparison_significance(first, second):
    """Return the formatted significance of ``median(first) - median(second)``.

    The posterior probability of the difference's sign is used when the
    draws can resolve it: with ``n`` draws in the smaller posterior, tail
    probabilities below ``1/n`` rest on a handful of chance overlaps, so
    significances beyond ``norm.isf(1/n)`` (about 3.4 sigma for 2500 draws)
    are not trusted.  Those, and posteriors that never overlap, fall back to
    the difference of medians divided by the 16th/84th-percentile errors that
    face each other, added in quadrature.
    """
    _, z_score, lower_bound = _posterior_difference_significance(
        second, first
    )
    resolvable_z = norm.isf(1.0/min(np.size(first), np.size(second)))
    if not lower_bound and z_score <= resolvable_z:
        return _format_posterior_significance(z_score), "posterior"
    first_low, first_median, first_high = np.percentile(first, [16, 50, 84])
    second_low, second_median, second_high = np.percentile(
        second, [16, 50, 84]
    )
    difference = first_median - second_median
    if difference >= 0:
        error = np.hypot(first_median - first_low, second_high - second_median)
    else:
        error = np.hypot(first_high - first_median, second_median - second_low)
    return (
        _format_posterior_significance(abs(difference)/error), "quadrature"
    )


def _comparison_values(name, suffix, first, second, kind):
    """Return ``(command, value, method)`` rows comparing two posteriors.

    Commands are ``name + <statistic> + suffix``.  ``kind`` is ``"rate"``
    (Ratio), ``"location"`` (Diff in dex and Factor ``10**Diff``), or
    ``"difference"`` (Diff alone); every kind also gets a Significance,
    whose ``method`` is ``"posterior"`` or ``"quadrature"`` (``None`` for
    the other rows).
    """
    first = np.asarray(first, dtype=float).reshape(-1)
    second = np.asarray(second, dtype=float).reshape(-1)
    first_median, second_median = np.median(first), np.median(second)
    rows = []
    if kind == "rate":
        if second_median <= 0:
            raise ValueError(f"cannot form the ratio {name}: zero denominator")
        rows.append((name + "Ratio" + suffix,
                     _format_two_figures(first_median/second_median), None))
    else:
        difference = first_median - second_median
        rows.append((name + "Diff" + suffix, f"{difference:.2f}", None))
        if kind == "location":
            rows.append((name + "Factor" + suffix,
                         _format_two_figures(10**difference), None))
    significance, method = _comparison_significance(first, second)
    rows.append((name + "Significance" + suffix, significance, method))
    return rows


def _parameter_samples(chain_path):
    with np.load(chain_path) as chain:
        key = "flat_chains" if "flat_chains" in chain else "chains"
        samples = np.asarray(chain[key], dtype=float)
    return samples.reshape(-1, samples.shape[-1])


def _sample_comparison_blocks(
        results_dir, tier1_dirs, tier2_types, tier3_dirs, stack_dim,
        low_first_types, command_names, median_ratios=None):
    """Build variables comparing every high/low sample pair.

    A pair is the ``high<type>`` and ``low<type>`` folders of one Tier 1
    and Tier 3 experiment; pairs missing either folder are skipped.  Each
    pair compares the piecewise integrated occurrence and every smooth
    model's parameters and integrated occurrence fitted in both samples.
    Comparisons run high relative to low, or low relative to high for the
    types in ``low_first_types``.
    """
    dim = stack_dim.lower()
    blocks = []
    for tier1_dir in tier1_dirs:
        tier1_name = Path(tier1_dir).name
        for tier2_type in tier2_types:
            if str(tier2_type).lower() == "allstars":
                continue
            pair = [f"high{tier2_type}", f"low{tier2_type}"]
            if tier2_type in low_first_types:
                pair.reverse()
            for tier3_dir in tier3_dirs:
                dirs = [results_dir / tier1_dir / tier2_dir / tier3_dir
                        for tier2_dir in pair]
                if not all(directory.is_dir() for directory in dirs):
                    continue
                chains = [directory / "saved_chains" for directory in dirs]
                prefix = (
                    _tier1_prefix(tier1_name) + _latex_token(tier2_type) +
                    _latex_token(Path(tier3_dir).name)
                )
                rows = []
                piecewise = [chain_dir / "chains_piecewise.npz"
                             for chain_dir in chains]
                if all(path.is_file() for path in piecewise):
                    first, second = (
                        _piecewise_stack_occurrence_samples(path, stack_dim)
                        for path in piecewise
                    )
                    for bin_index in range(min(first.shape[1],
                                               second.shape[1])):
                        rows.extend(_comparison_values(
                            prefix + "PiecewiseIntOcc",
                            f"Bin{dim}{_number_word(bin_index)}",
                            first[:, bin_index], second[:, bin_index], "rate",
                        ))
                for model_name, (model_macro, parameter_macros) in (
                        _MODEL_PARAMETER_MACROS.items()):
                    bin_index = 0
                    while all((chain_dir /
                               f"chains_{model_name}_bin{bin_index}.npz"
                               ).is_file() for chain_dir in chains):
                        paths = [chain_dir /
                                 f"chains_{model_name}_bin{bin_index}.npz"
                                 for chain_dir in chains]
                        first, second = (_parameter_samples(path)
                                         for path in paths)
                        suffix = f"Bin{dim}{_number_word(bin_index)}"
                        for index, parameter in enumerate(parameter_macros):
                            kind = (
                                "rate" if parameter in _RATE_PARAMETERS else
                                "location"
                                if parameter in _LOCATION_PARAMETERS else
                                "difference"
                            )
                            rows.extend(_comparison_values(
                                prefix + model_macro + "Param" + parameter,
                                suffix, first[:, index], second[:, index],
                                kind,
                            ))
                        rows.extend(_comparison_values(
                            prefix + model_macro + "IntOcc", suffix,
                            _parametric_occurrence_samples(
                                dirs[0], model_name, bin_index
                            ),
                            _parametric_occurrence_samples(
                                dirs[1], model_name, bin_index
                            ),
                            "rate",
                        ))
                        bin_index += 1
                if rows and median_ratios is not None:
                    ratios = median_ratios(*pair)
                    if ratios is not None:
                        rows.extend([
                            (prefix + "MedianMstarRatio", ratios[0], None),
                            (prefix + "MedianFeHRatio", ratios[1], None),
                        ])
                if not rows:
                    continue
                block = [
                    "%"*72,
                    f"% High/low sample comparison: {tier1_name} / "
                    f"{tier2_type} / {Path(tier3_dir).name}",
                    f"% Ratio = {pair[0]}/{pair[1]}; Diff = {pair[0]} - "
                    f"{pair[1]} (dex); Factor = 10^Diff; MedianFeHRatio "
                    "compares linear abundances",
                ]
                quadrature = []
                for command, value, method in rows:
                    if command in command_names:
                        raise ValueError(
                            f"duplicate LaTeX command name: {command}"
                        )
                    command_names.add(command)
                    block.append(_command(command, value))
                    if method == "quadrature":
                        quadrature.append(command)
                if quadrature:
                    block.append(
                        "% Quadrature significances (beyond what the "
                        "posterior draws resolve): " + ", ".join(quadrature)
                    )
                blocks.append("\n".join(block))
    if blocks:
        blocks.insert(0, "\n".join([
            "%"*72,
            "%"*72,
            "% HIGH/LOW SAMPLE COMPARISONS",
            "% Each block compares one pair of stellar samples (e.g. highMstar",
            "% vs lowMstar) within one Tier 1 / Tier 3 experiment, using",
            "% posterior medians. Rates get a Ratio; log10 locations (sigmoid",
            "% center, logG mu, breakpoints) get a Diff in dex and its Factor",
            "% (10^Diff); widths and slopes get a Diff. Every quantity gets a",
            "% Significance from the posterior probability of the difference's",
            "% sign or, when that exceeds what the draws resolve, from the",
            "% difference divided by the facing 16th/84th-percentile errors",
            "% in quadrature (listed at the end of each block).",
            "%"*72,
            "%"*72,
        ]))
    return blocks


def _two_parameter_variable_blocks(
        results_dir, tier1_dirs, two_parameter_t3s, stellar_catalog_path,
        two_parameter_cuts, command_names, median_values=None,
        median_ratios=None):
    """Build variables for the mass-metallicity subsets of each Tier 3 run.

    Each subset gets the statistics in :func:`make_two_parameter_tables`
    (``Nstars``, ``Neff``, ``AvgCompl``, ``PiecewiseIntOcc``), and each
    comparison in the reordered table gets a ``Significance`` (posterior, or
    quadrature beyond what the draws resolve) and a ``DynamicRange``.
    Returns the blocks and the subsets and comparisons that were missing.
    """
    blocks, missing_results, missing_comparisons = [], [], []

    def add(block, name, value):
        if name in command_names:
            raise ValueError(f"duplicate LaTeX command name: {name}")
        command_names.add(name)
        block.append(_command(name, value))

    for two_parameter_t3 in two_parameter_t3s:
        for tier1_dir in tier1_dirs:
            tier1_name = Path(tier1_dir).name
            samples = {}
            for tier2_dir in _TWO_PARAMETER_TIER2_DIRS:
                result_dir = (
                    results_dir / tier1_dir / tier2_dir / two_parameter_t3
                )
                summary_path = result_dir / "saved_dicts" / SUMMARY_FILENAME
                if not summary_path.is_file():
                    missing_results.append(
                        _result_key(tier1_dir, tier2_dir, two_parameter_t3)
                    )
                    continue
                levels = _two_parameter_levels(tier2_dir)
                block = [
                    "%"*72,
                    f"% {tier1_name} / {tier2_dir} / {two_parameter_t3}",
                ]
                statistics = _three_parameter_statistics(result_dir)
                if median_values is not None:
                    statistics.update(
                        median_values(tier2_dir, statistics["Nstars"])
                    )
                for statistic, value in statistics.items():
                    add(block, _two_parameter_command_name(
                        tier1_name, two_parameter_t3, levels, statistic
                    ), value)
                blocks.append("\n".join(block))
                if (result_dir / "saved_chains" /
                        "chains_piecewise.npz").is_file():
                    samples[levels] = _three_parameter_occurrence_samples(
                        result_dir
                    )

            block = [
                "%"*72,
                f"% {tier1_name} / {two_parameter_t3} comparisons",
            ]
            dynamic_ranges = None
            for varied_parameter, fixed_level, low_levels, high_levels in (
                    _two_parameter_comparisons()):
                significance_name = _two_parameter_comparison_command_name(
                    tier1_name, two_parameter_t3, varied_parameter,
                    fixed_level, "Significance",
                )
                if low_levels not in samples or high_levels not in samples:
                    missing_comparisons.append(significance_name)
                    continue
                significance, _ = _comparison_significance(
                    samples[high_levels], samples[low_levels]
                )
                add(block, significance_name, significance)
                if dynamic_ranges is None:
                    dynamic_ranges = _two_parameter_dynamic_ranges(
                        stellar_catalog_path, two_parameter_cuts
                    )
                add(block, _two_parameter_comparison_command_name(
                    tier1_name, two_parameter_t3, varied_parameter,
                    fixed_level, "DynamicRange",
                ), f"{dynamic_ranges[(varied_parameter, fixed_level)]:.1f}")
                ratios = None if median_ratios is None else median_ratios(
                    "{}Mstar{}FeH".format(*high_levels),
                    "{}Mstar{}FeH".format(*low_levels),
                )
                if ratios is not None:
                    for suffix, value in zip(
                            ("MedianMstarRatio", "MedianFeHRatio"), ratios):
                        add(block, _two_parameter_comparison_command_name(
                            tier1_name, two_parameter_t3, varied_parameter,
                            fixed_level, suffix,
                        ), value)
            if len(block) > 2:
                blocks.append("\n".join(block))
    return blocks, missing_results, missing_comparisons


def make_variables(
        results_dir, tier1_dirs, tier2_types, tier3_dirs, stack_dim="a",
        three_parameter_t3="stellar3params", stellar_catalog_path=None,
        three_parameter_cuts=None, low_first_comparisons=(),
        standalone_tier2_dirs=(), tier3_stack_dims=None,
        percent_tier3_dirs=(), two_parameter_t3=None,
        two_parameter_cuts=None, sample_queries=None, sample_catalog=None):
    """Write non-model fit statistics to a LaTeX variables file.

    ``results_dir`` is the parent of all Tier 1 directories.  The output is
    written to ``results_dir/paper_tables/variables.tex``.  ``tier2_types``
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
    Command names mirror the full directory hierarchy, for example
    ``\\McAllstarsPaperBoundsNeff`` for ``mtrue/allstars/paper_bounds``.
    Every Tier 1/Tier 2/Tier 3 combination whose Tier 3 folder exists is
    included, so Tier 3 experiments covering different subsets can share one
    file; combinations without a folder are skipped and reported.  A folder
    that exists without a piecewise summary is an error, as is a Tier 3
    directory with no folders at all.  Available three-parameter subset results beneath
    ``three_parameter_t3`` are also included.  It may name one Tier 3
    directory, several (a list), or none (``None``).  Each directory's name
    is spelled into its command names, so ``stellar3params`` gives
    ``\\McStellarThreeParamsLowMstarLowFeHYoungNstars`` and several
    three-parameter experiments can share one variables file.  When both
    required piecewise chains are available, the file also includes the posterior significance
    of every comparison in the reordered three-parameter table.  Dynamic
    ranges use ``stellar_catalog_path`` and ``three_parameter_cuts``; their
    defaults are the repository CLS stellar catalog, the inclusive stellar
    mass range 0.82--1.21 solar masses, and cuts at 1 solar mass, zero dex,
    and 5 Gyr.  Missing subsets or comparison chains are reported and omitted
    without interrupting generation of the other commands.

    Every high/low pair of a Tier 2 type (e.g. ``highMstar`` and
    ``lowMstar``) within one Tier 1 and Tier 3 experiment is also compared,
    in a separately labeled section: rates get a Ratio, log10 locations a
    Diff and Factor, other parameters a Diff, and every quantity a
    Significance (see :func:`_sample_comparison_blocks`).  Command names use
    the Tier 2 type in place of the folder, e.g.
    ``\\McMstarPaperBoundsSigmoidParamCenterFactorBinaZero``.  Pairs compare
    high relative to low, except for types listed in
    ``low_first_comparisons`` (e.g. ``["Act"]`` for old relative to young).

    ``standalone_tier2_dirs`` lists Tier 2 directories used as-is, like
    ``"allstars"``, rather than expanded into high/low pairs.
    ``tier3_stack_dims`` maps Tier 3 directory names to the stack dimension
    used for their per-bin statistics, overriding ``stack_dim`` (e.g.
    ``{"cui_run": "m"}`` for rates per mass bin over one separation bin).
    Integrated piecewise occurrence for the Tier 3 directories in
    ``percent_tier3_dirs`` is written in percent.

    ``two_parameter_t3`` names one or more Tier 3 directories (or ``None``)
    holding the mass-metallicity subsets (``highMstarhighFeH`` etc.).  Their
    statistics, comparison significances, and dynamic ranges get the
    command names :func:`make_two_parameter_tables` references, e.g.
    ``\\McStellarTwoParamsHighMstarHighFeHPiecewiseIntOcc`` and
    ``\\McStellarTwoParamsMassHighFeHSignificance``.  Dynamic ranges use the
    full stellar catalog with ``two_parameter_cuts``.

    With ``sample_queries`` (Tier 2 directory -> the pandas query that
    selected its stars, ``None`` for all) and ``sample_catalog`` (the
    stellar DataFrame the fits used), every sample, including the two- and
    three-parameter subsets, also gets ``MedianMstar``, ``MedianFeH``, and,
    when all its stars lie within the age-activity mass range,
    ``MedianAge`` (e.g. ``\\McHighMstarPaperBoundsMedianMstar``), and
    every high/low comparison (1D pairs and the two-parameter comparisons)
    gets ``MedianMstarRatio`` and ``MedianFeHRatio``, the ratios of the two
    samples' medians, with [Fe/H] in linear abundance (e.g.
    ``\\McMstarPaperBoundsMedianFeHRatio``).  Each query must select as many
    stars as the fit used.

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
    for tier3_dir in tier3_dirs:
        _tier3_stack_dim(tier3_dir, stack_dim, tier3_stack_dims)
    unknown_runs = sorted(
        (set(tier3_stack_dims or {}) | set(percent_tier3_dirs))
        - {Path(tier3_dir).name for tier3_dir in tier3_dirs}
    )
    if unknown_runs:
        raise ValueError(
            f"tier3_stack_dims or percent_tier3_dirs name unrequested Tier 3 "
            f"directories {unknown_runs}"
        )
    percent_tier3_dirs = set(percent_tier3_dirs)
    low_first_comparisons = set(low_first_comparisons)
    unknown_types = sorted(low_first_comparisons - set(tier2_types))
    if unknown_types:
        raise ValueError(
            f"low_first_comparisons names unrequested types {unknown_types}"
        )

    if three_parameter_t3 is None:
        three_parameter_t3s = []
    elif isinstance(three_parameter_t3, (str, os.PathLike)):
        three_parameter_t3s = [three_parameter_t3]
    else:
        three_parameter_t3s = list(three_parameter_t3)
    if len(set(map(str, three_parameter_t3s))) != len(three_parameter_t3s):
        raise ValueError("three_parameter_t3 contains duplicate directories")

    if (sample_queries is None) != (sample_catalog is None):
        raise ValueError("sample_queries and sample_catalog go together")

    def median_values(tier2_dir, nstars):
        if sample_queries is None or str(tier2_dir) not in sample_queries:
            return {}
        return _sample_median_values(
            sample_catalog, sample_queries[str(tier2_dir)], nstars, tier2_dir
        )

    def median_ratios(first_dir, second_dir):
        """High/low median mass and metallicity ratios of two samples."""
        if sample_queries is None or not {
                str(first_dir), str(second_dir)} <= set(sample_queries):
            return None
        medians = []
        for tier2_dir in (first_dir, second_dir):
            query = sample_queries[str(tier2_dir)]
            selected = (sample_catalog if query is None
                        else sample_catalog.query(query))
            medians.append({column: float(selected[column].median())
                            for column in ("Mstar", "feh")})
        return _format_median_ratios(*medians)

    all_delta_bics = calculate_all_delta_bics(
        results_dir, tier1_dirs, tier2_types, tier3_dirs, stack_dim,
        standalone_tier2_dirs, tier3_stack_dims,
    )
    blocks = []
    command_names = set()
    skipped_experiments = []
    found_tier3_dirs = set()

    requested_tier2_dirs = _requested_tier2_dirs(
        tier2_types, standalone_tier2_dirs
    )
    for tier1_dir in tier1_dirs:
        tier1_name = Path(tier1_dir).name
        tier1_path = results_dir / tier1_dir
        for tier2_dir in requested_tier2_dirs:
                for tier3_dir in tier3_dirs:
                    experiment_dim = _tier3_stack_dim(
                        tier3_dir, stack_dim, tier3_stack_dims
                    )
                    if not (tier1_path / tier2_dir / tier3_dir).is_dir():
                        skipped_experiments.append(
                            _result_key(tier1_dir, tier2_dir, tier3_dir)
                        )
                        continue
                    found_tier3_dirs.add(str(tier3_dir))
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
                            summary, experiment_dim, summary_path
                        )
                        n_stack_bins = len(bin_neff)

                    prefix = _experiment_prefix(
                        tier1_name, tier2_dir, tier3_dir
                    )
                    values = {
                        "Nstars": str(int(nstars)),
                        "Neff": f"{neff:.1f}",
                        "AvgCompl": f"{avg_compl:.2f}",
                    }
                    values.update(median_values(tier2_dir, nstars))
                    dim = experiment_dim.upper()
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
                        chain_dir, prefix, experiment_dim, n_stack_bins,
                        percent=Path(tier3_dir).name in percent_tier3_dirs,
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
                            chain_dir, prefix, experiment_dim, n_stack_bins)
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
                                    f"Bin{experiment_dim}{_number_word(bin_index)}"
                                )
                                if name in command_names:
                                    raise ValueError(
                                        f"duplicate LaTeX command name: {name}"
                                    )
                                command_names.add(name)
                                block.append(_command(name, f"{delta:.1f}"))
                    blocks.append("\n".join(block))

    empty_tier3_dirs = [
        str(tier3_dir) for tier3_dir in tier3_dirs
        if str(tier3_dir) not in found_tier3_dirs
    ]
    if empty_tier3_dirs:
        raise FileNotFoundError(
            "no requested Tier 1/Tier 2 folders contain results for "
            f"{empty_tier3_dirs}"
        )
    if skipped_experiments:
        print(
            "post_fit_analysis.make_variables: skipped experiments without "
            "result folders: " + ", ".join(skipped_experiments)
        )

    blocks.extend(_sample_comparison_blocks(
        results_dir, tier1_dirs, tier2_types, tier3_dirs, stack_dim,
        low_first_comparisons, command_names, median_ratios,
    ))

    missing_three_parameter_results = []
    missing_three_parameter_comparisons = []
    for three_parameter_t3 in three_parameter_t3s:
        for tier1_dir in tier1_dirs:
            tier1_name = Path(tier1_dir).name
            occurrence_samples_by_model = {("piecewise", None): {}}
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
                    occurrence_samples_by_model[("piecewise", None)][levels] = (
                        _three_parameter_occurrence_samples(result_dir)
                    )
                statistics = _three_parameter_statistics(result_dir)
                statistics.update(
                    median_values(tier2_dir, statistics["Nstars"])
                )
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

                chain_dir = result_dir / "saved_chains"
                has_parametric_chains = any(
                    chain_dir.glob("chains_*_bin*.npz")
                )
                parametric_groups = []
                if has_parametric_chains:
                    with np.load(summary_path) as summary:
                        bin_neff, _ = _stack_statistics(
                            summary, stack_dim, summary_path
                        )
                    prefix = _three_parameter_command_name(
                        tier1_name, three_parameter_t3, levels, statistic=""
                    )
                    parametric_groups = _parametric_values(
                        chain_dir, prefix, stack_dim, len(bin_neff)
                    )
                for model_name, model_values in parametric_groups:
                    integrated_values = [
                        (name, value) for name, value in model_values
                        if "IntOcc" in name
                    ]
                    if not integrated_values:
                        continue
                    block.extend([
                        "", "%"*36,
                        f"% Parametric integrated occurrence: {model_name}",
                    ])
                    for name, value in integrated_values:
                        if name in command_names:
                            raise ValueError(
                                f"duplicate LaTeX command name: {name}"
                            )
                        command_names.add(name)
                        block.append(_command(name, value))
                    for bin_index in range(len(bin_neff)):
                        occurrence_samples_by_model.setdefault(
                            (model_name, bin_index), {}
                        )[levels] = _parametric_occurrence_samples(
                            result_dir, model_name, bin_index
                        )
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
                piecewise_samples = occurrence_samples_by_model[
                    ("piecewise", None)
                ]
                if (low_levels not in piecewise_samples or
                        high_levels not in piecewise_samples):
                    missing_three_parameter_comparisons.append(
                        significance_name
                    )
                else:
                    probability, z_score, lower_bound = (
                        _posterior_difference_significance(
                            piecewise_samples[low_levels],
                            piecewise_samples[high_levels],
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

                for (model_name, bin_index), model_samples in (
                        occurrence_samples_by_model.items()):
                    if model_name == "piecewise":
                        continue
                    if (low_levels not in model_samples or
                            high_levels not in model_samples):
                        continue
                    model_significance_name = (
                        _three_parameter_significance_command_name(
                            tier1_name, three_parameter_t3,
                            varied_parameter, fixed_levels,
                            model_name, stack_dim, bin_index,
                        )
                    )
                    probability, z_score, lower_bound = (
                        _posterior_difference_significance(
                            model_samples[low_levels],
                            model_samples[high_levels],
                        )
                    )
                    if model_significance_name in command_names:
                        raise ValueError(
                            "duplicate LaTeX command name: "
                            f"{model_significance_name}"
                        )
                    command_names.add(model_significance_name)
                    comparison_block.append(_command(
                        model_significance_name,
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
                        dynamic_name, f"{dynamic_ranges[(varied_parameter, fixed_levels)]:.1f}"
                    ))
            if len(comparison_block) > 2:
                blocks.append("\n".join(comparison_block))
    if two_parameter_t3 is None:
        two_parameter_t3s = []
    elif isinstance(two_parameter_t3, (str, os.PathLike)):
        two_parameter_t3s = [two_parameter_t3]
    else:
        two_parameter_t3s = list(two_parameter_t3)
    two_blocks, missing_two_results, missing_two_comparisons = (
        _two_parameter_variable_blocks(
            results_dir, tier1_dirs, two_parameter_t3s,
            stellar_catalog_path, two_parameter_cuts, command_names,
            median_values, median_ratios,
        )
    )
    blocks.extend(two_blocks)
    if missing_two_results:
        print(
            "post_fit_analysis.make_variables: two-parameter occurrence "
            "results have not been calculated for: "
            + ", ".join(missing_two_results)
        )
    if missing_two_comparisons:
        print(
            "post_fit_analysis.make_variables: two-parameter significance "
            "could not be calculated without both piecewise chains for: "
            + ", ".join(missing_two_comparisons)
        )
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

    paper_tables_dir = results_dir / PAPER_TABLES_DIRNAME
    output_path = paper_tables_dir / "variables.tex"
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    contents = "% Auto-generated by post_fit_analysis.make_variables\n\n"
    contents += "\n\n".join(blocks) + "\n"
    output_path.write_text(contents, encoding="utf-8")
    return output_path
