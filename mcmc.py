"""MCMC sampling for histogram-free occurrence models."""

import os
import multiprocessing as mp
from pathlib import Path

import emcee
import numpy as np

from occurrence import fit_utils as dfu
from occurrence import likelihood as dl
from occurrence import mcmc_powerlaw


def log_probability_piecewise(theta, cache):
    """Apply physical cell-rate bounds to the direct piecewise likelihood."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != cache.cell_areas.shape:
        raise ValueError(f"piecewise model requires {cache.cell_areas.size} parameters")
    cell_rates = theta*cache.cell_areas
    if not np.isfinite(theta).all() or np.any(cell_rates <= 0) or np.any(cell_rates > 1):
        return -np.inf
    return dl.cached_piecewise_log_likelihood(theta, cache)


def mcmc_piecewise(
        companions,
        exposure,
        x_edges,
        y_edges,
        nwalkers=50,
        nsteps=5000,
        burnin=1000,
        parallel=False,
        save_path="chains_piecewise.npz",
        random_seed=None):
    """Fit cell-wise densities directly to companion posterior samples."""
    x_edges = dl._validate_edges(x_edges, "x_edges")
    y_edges = dl._validate_edges(y_edges, "y_edges")
    if not np.allclose(x_edges[[0, -1]], exposure.x_bounds):
        raise ValueError("outer x_edges must match the exposure x_bounds")
    if not np.allclose(y_edges[[0, -1]], exposure.y_bounds):
        raise ValueError("outer y_edges must match the exposure y_bounds")
    cache = dl.build_piecewise_cache(
        companions, exposure, x_edges, y_edges
    )
    cell_areas = cache.cell_areas
    ndim = cell_areas.size
    if nwalkers < 2*ndim:
        raise ValueError("nwalkers must be at least twice the number of cells")

    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    center = rng.uniform(0.01, 0.05, size=ndim)/cell_areas
    pos = center*(1 + 1e-2*rng.randn(nwalkers, ndim))
    pos = np.clip(pos, 1e-8, 0.99/cell_areas)
    arguments = (cache,)
    initial_log_probabilities = np.array([
        log_probability_piecewise(position, *arguments) for position in pos
    ])
    if not np.isfinite(initial_log_probabilities).all():
        unsupported = [
            name for name, companion in companions.items()
            if not np.any(companion.roi_mask)
        ]
        detail = f"; companions without ROI support: {unsupported}" if unsupported else ""
        raise ValueError(f"MCMC initialization has nonfinite likelihoods{detail}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if parallel:
        with mp.Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability_piecewise, args=arguments, pool=pool
            )
            _run_sampler(sampler, pos, burnin, nsteps, rng, random_seed)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_piecewise, args=arguments
        )
        _run_sampler(sampler, pos, burnin, nsteps, rng, random_seed)

    log_probabilities = sampler.get_log_prob()
    if not np.isfinite(log_probabilities).all():
        raise RuntimeError("MCMC produced nonfinite log probabilities")
    np.savez_compressed(
        save_path,
        chains=sampler.get_chain(),
        log_probs=log_probabilities,
        flat_chains=sampler.get_chain(flat=True),
        flat_log_probs=sampler.get_log_prob(flat=True),
        x_edges=x_edges,
        y_edges=y_edges,
        cell_areas=cache.cell_areas,
        cell_exposure=cache.cell_exposure,
        effective_counts=cache.effective_counts,
        log_weight_constant=cache.log_weight_constant,
    )
    return sampler


def fit_piecewise_file(fit_path, x_edges, y_edges, **mcmc_options):
    """Load Stage 2 materials and run a direct piecewise-constant fit."""
    companions, exposure = dfu.load_fit_data(fit_path)
    return mcmc_piecewise(
        companions, exposure, x_edges, y_edges, **mcmc_options
    )


def physical_smooth_parameters(model_name, transformed_theta):
    """Convert a smooth model's sampled parameters to physical coordinates."""
    parameters = np.asarray(transformed_theta, dtype=float).copy()
    if model_name == "logG":
        parameters[..., 0] = np.exp(parameters[..., 0])
        parameters[..., 2] = np.exp(parameters[..., 2])
    elif model_name == "escarpment":
        parameters[..., :2] = np.exp(parameters[..., :2])
    elif model_name == "sigmoid":
        parameters[..., :2] = np.exp(parameters[..., :2])
        parameters[..., 3] = np.exp(parameters[..., 3])
    elif model_name == "bpl":
        parameters[..., 0] = np.exp(parameters[..., 0])
    else:
        raise ValueError(f"unsupported smooth model {model_name!r}")
    return parameters


def integrated_smooth_occurrence(model_name, transformed_theta, cache):
    """Numerically integrate a registered smooth ORD over the fitted region."""
    physical = physical_smooth_parameters(model_name, transformed_theta)
    density = mcmc_powerlaw.get_model_spec(model_name).function(
        physical, 10**cache.log_x_grid
    )
    model_weights = dl._trapezoid_weights(cache.log_x_grid)
    stack_width = np.log10(cache.stack_bounds[1]/cache.stack_bounds[0])
    return np.dot(density, model_weights)*stack_width


def log_prior_smooth(
        transformed_theta, cache, model_name,
        amplitude_bounds=(1e-6, 10.0), slope_bounds=(-4.0, 4.0),
        width_bounds=None,
        max_integrated_occurrence=1.0):
    """Evaluate a smooth model's physical priors in sampled coordinates."""
    physical = physical_smooth_parameters(model_name, transformed_theta)
    log_min, log_max = np.log10(cache.model_bounds)
    if model_name == "logG":
        amplitude, mu, sigma = physical
        if width_bounds is None:
            width_bounds = _default_width_bounds(cache)
        valid = (
            amplitude_bounds[0] <= amplitude <= amplitude_bounds[1] and
            log_min <= mu <= log_max and
            width_bounds[0] <= sigma <= width_bounds[1]
        )
        jacobian = transformed_theta[0] + transformed_theta[2]
    elif model_name == "escarpment":
        low, high, break1, break2 = physical
        valid = (
            amplitude_bounds[0] <= low <= amplitude_bounds[1] and
            amplitude_bounds[0] <= high <= amplitude_bounds[1] and
            log_min <= break1 < break2 <= log_max
        )
        jacobian = transformed_theta[0] + transformed_theta[1]
    elif model_name == "sigmoid":
        c1, c2, center, width = physical
        if width_bounds is None:
            width_bounds = _default_width_bounds(cache)
        valid = (
            amplitude_bounds[0] <= c1 <= amplitude_bounds[1] and
            amplitude_bounds[0] <= c2 <= amplitude_bounds[1] and
            log_min <= center <= log_max and
            width_bounds[0] <= width <= width_bounds[1]
        )
        jacobian = (
            transformed_theta[0] + transformed_theta[1] +
            transformed_theta[3]
        )
    elif model_name == "bpl":
        amplitude, log_break, beta, gamma = physical
        valid = (
            amplitude_bounds[0] <= amplitude <= amplitude_bounds[1] and
            log_min <= log_break <= log_max and
            slope_bounds[0] <= beta <= slope_bounds[1] and
            slope_bounds[0] <= gamma <= slope_bounds[1]
        )
        jacobian = transformed_theta[0]
    else:
        raise ValueError(f"unsupported smooth model {model_name!r}")
    if not valid:
        return -np.inf
    if (max_integrated_occurrence is not None and
            integrated_smooth_occurrence(model_name, transformed_theta, cache) >
            max_integrated_occurrence):
        return -np.inf
    return jacobian


def log_probability_smooth(
        transformed_theta, cache, model_name,
        amplitude_bounds=(1e-6, 10.0), slope_bounds=(-4.0, 4.0),
        width_bounds=None,
        max_integrated_occurrence=1.0):
    """Return prior plus cached likelihood for a registered direct model."""
    prior = log_prior_smooth(
        transformed_theta, cache, model_name, amplitude_bounds, slope_bounds,
        width_bounds, max_integrated_occurrence,
    )
    if not np.isfinite(prior):
        return -np.inf
    physical = physical_smooth_parameters(model_name, transformed_theta)
    likelihood = dl.cached_smooth_log_likelihood(
        physical, cache, mcmc_powerlaw.get_model_spec(model_name).function
    )
    return prior + likelihood if np.isfinite(likelihood) else -np.inf


def initialize_smooth(
        cache, model_name, amplitude_bounds=(1e-6, 10.0),
        width_bounds=None):
    """Initialize a smooth model from direct catalog and exposure information."""
    log_min, log_max = np.log10(cache.model_bounds)
    span = log_max - log_min
    constant_exposure = np.sum(cache.exposure_weights)
    amplitude = np.sum(cache.companion_weights)/constant_exposure
    amplitude = np.clip(amplitude, amplitude_bounds[0]*10, amplitude_bounds[1]/10)
    if model_name == "logG":
        if width_bounds is None:
            width_bounds = _default_width_bounds(cache)
        mu = (np.median(cache.log_x_samples) if cache.log_x_samples.size
              else (log_min + log_max)/2)
        sigma = np.clip(
            np.std(cache.log_x_samples) if cache.log_x_samples.size else span/4,
            *width_bounds
        )
        if sigma == width_bounds[0]:
            sigma = min(max(span/4, width_bounds[0]), width_bounds[1])
        shape = np.exp(-0.5*((cache.log_x_grid - mu)/sigma)**2)
        amplitude = np.sum(cache.companion_weights)/np.dot(
            shape, cache.exposure_weights
        )
        amplitude = np.clip(
            amplitude, amplitude_bounds[0]*10, amplitude_bounds[1]/10
        )
        return np.array([np.log(amplitude), mu, np.log(sigma)])
    if model_name == "escarpment":
        return np.array([
            np.log(amplitude), np.log(amplitude),
            log_min + span/3, log_min + 2*span/3,
        ])
    if model_name == "sigmoid":
        if width_bounds is None:
            width_bounds = _default_width_bounds(cache)
        width = np.clip(span/10, *width_bounds)
        return np.array([
            np.log(amplitude), np.log(amplitude),
            (log_min + log_max)/2, np.log(width),
        ])
    if model_name == "bpl":
        center = (log_min + log_max)/2
        shape = dl.broken_powerlaw_density([1.0, center, 0.0, 1.0],
                                           10**cache.log_x_grid)
        normalized_amplitude = np.sum(cache.companion_weights)/np.dot(
            shape, cache.exposure_weights
        )
        normalized_amplitude = np.clip(
            normalized_amplitude, amplitude_bounds[0]*10, amplitude_bounds[1]/10
        )
        return np.array([np.log(normalized_amplitude), center, 0.0, 1.0])
    raise ValueError(f"unsupported smooth model {model_name!r}")


def mcmc_smooth(
        cache, model_name, nwalkers=50, nsteps=5000, burnin=1000,
        parallel=False, save_path=None, random_seed=None,
        amplitude_bounds=(1e-6, 10.0), slope_bounds=(-4.0, 4.0),
        width_bounds=None,
        max_integrated_occurrence=1.0):
    """Fit one registered direct smooth model in one stack interval."""
    if model_name not in {"logG", "escarpment", "sigmoid", "bpl"}:
        raise ValueError(
            "model_name must be 'logG', 'escarpment', 'sigmoid', or 'bpl'"
        )
    ndim = mcmc_powerlaw.MODEL_REGISTRY[model_name].ndim
    if nwalkers < 2*ndim:
        raise ValueError(f"nwalkers must be at least {2*ndim} for {model_name}")
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    center = initialize_smooth(
        cache, model_name, amplitude_bounds, width_bounds
    )
    if max_integrated_occurrence is not None:
        integrated = integrated_smooth_occurrence(model_name, center, cache)
        if integrated > 0.8*max_integrated_occurrence:
            adjustment = np.log(0.8*max_integrated_occurrence/integrated)
            center[0] += adjustment
            if model_name in {"escarpment", "sigmoid"}:
                center[1] += adjustment
    if model_name == "logG":
        scales = np.array([0.05, 0.02, 0.05])
    elif model_name == "escarpment":
        scales = np.array([0.05, 0.05, 0.02, 0.02])
    elif model_name == "sigmoid":
        scales = np.array([0.05, 0.05, 0.02, 0.05])
    else:
        scales = np.array([0.05, 0.02, 0.05, 0.05])
    arguments = (
        cache, model_name, amplitude_bounds, slope_bounds, width_bounds,
        max_integrated_occurrence,
    )
    positions = center + rng.randn(nwalkers, ndim)*scales
    for index in range(nwalkers):
        attempts = 0
        while not np.isfinite(log_probability_smooth(positions[index], *arguments)):
            positions[index] = center + rng.randn(ndim)*scales
            attempts += 1
            if attempts >= 1000:
                raise ValueError(f"could not initialize finite {model_name} walkers")
    if save_path is None:
        save_path = f"chains_{model_name}.npz"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if parallel:
        with mp.Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability_smooth, args=arguments, pool=pool
            )
            _run_sampler(sampler, positions, burnin, nsteps, rng, random_seed)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_smooth, args=arguments
        )
        _run_sampler(sampler, positions, burnin, nsteps, rng, random_seed)
    transformed = sampler.get_chain()
    transformed_flat = sampler.get_chain(flat=True)
    physical = physical_smooth_parameters(model_name, transformed)
    physical_flat = physical_smooth_parameters(model_name, transformed_flat)
    log_probs = sampler.get_log_prob()
    try:
        autocorrelation = sampler.get_autocorr_time(quiet=True)
    except Exception:
        autocorrelation = np.full(ndim, np.nan)
    np.savez_compressed(
        save_path, chains=physical, flat_chains=physical_flat,
        transformed_chains=transformed,
        transformed_flat_chains=transformed_flat,
        log_probs=log_probs, flat_log_probs=sampler.get_log_prob(flat=True),
        acceptance_fraction=sampler.acceptance_fraction,
        autocorrelation_time=autocorrelation,
        amplitude_bounds=np.asarray(amplitude_bounds),
        slope_bounds=np.asarray(slope_bounds),
        width_bounds=np.asarray(
            width_bounds if width_bounds is not None
            else _default_width_bounds(cache)
        ),
        max_integrated_occurrence=(np.nan if max_integrated_occurrence is None
                                   else max_integrated_occurrence),
        model_bounds=np.asarray(cache.model_bounds),
        stack_bounds=np.asarray(cache.stack_bounds),
        model_coordinate=cache.model_coordinate,
        stack_coordinate=cache.stack_coordinate,
        model_name=model_name,
    )
    return sampler


def fit_smooth_file(
        fit_path, a_edges, m_edges, stack_dim, model_name,
        **mcmc_options):
    """Fit one registered smooth model in every stack interval."""
    companions, exposure = dfu.load_fit_data(fit_path)
    a_edges = dl._validate_edges(a_edges, "a_edges")
    m_edges = dl._validate_edges(m_edges, "m_edges")
    stack_edges = a_edges if stack_dim == "a" else m_edges
    base_path = Path(mcmc_options.pop(
        "save_dir", Path(fit_path).parents[1] / "saved_chains"
    ))
    base_path.mkdir(parents=True, exist_ok=True)
    samplers, paths = [], []
    for index, bounds in enumerate(zip(stack_edges[:-1], stack_edges[1:])):
        cache = dl.build_smooth_cache(companions, exposure, stack_dim, bounds)
        path = base_path / f"chains_{model_name}_bin{index}.npz"
        options = dict(mcmc_options)
        if options.get("random_seed") is not None:
            options["random_seed"] += index
        samplers.append(mcmc_smooth(
            cache, model_name, save_path=path, **options
        ))
        paths.append(str(path))
    return samplers, paths


def _default_width_bounds(cache):
    """Return default positive transition/dispersion bounds in dex."""
    span = np.log10(cache.model_bounds[1]/cache.model_bounds[0])
    return max(span/100, 1e-3), span


def sample_smooth_prior(
        cache, model_name, size,
        amplitude_bounds=(1e-6, 10.0),
        slope_bounds=(-4.0, 4.0),
        width_bounds=None,
        max_integrated_occurrence=1.0,
        random_seed=None):
    """Draw physical parameter vectors from a smooth model's prior."""
    if size < 1:
        raise ValueError("size must be positive")
    width_bounds = width_bounds or _default_width_bounds(cache)
    rng = np.random.RandomState(random_seed)
    log_min, log_max = np.log10(cache.model_bounds)
    accepted = []
    attempts = 0
    while len(accepted) < size:
        if model_name == "logG":
            physical = np.array([
                rng.uniform(*amplitude_bounds),
                rng.uniform(log_min, log_max),
                rng.uniform(*width_bounds),
            ])
            transformed = np.array([
                np.log(physical[0]), physical[1], np.log(physical[2])
            ])
        elif model_name == "escarpment":
            breaks = np.sort(rng.uniform(log_min, log_max, 2))
            physical = np.r_[rng.uniform(*amplitude_bounds, 2), breaks]
            transformed = physical.copy()
            transformed[:2] = np.log(transformed[:2])
        elif model_name == "sigmoid":
            physical = np.array([
                rng.uniform(*amplitude_bounds),
                rng.uniform(*amplitude_bounds),
                rng.uniform(log_min, log_max),
                rng.uniform(*width_bounds),
            ])
            transformed = physical.copy()
            transformed[:2] = np.log(transformed[:2])
            transformed[3] = np.log(transformed[3])
        elif model_name == "bpl":
            physical = np.array([
                rng.uniform(*amplitude_bounds),
                rng.uniform(log_min, log_max),
                rng.uniform(*slope_bounds),
                rng.uniform(*slope_bounds),
            ])
            transformed = physical.copy()
            transformed[0] = np.log(transformed[0])
        else:
            raise ValueError(f"unsupported smooth model {model_name!r}")
        if np.isfinite(log_prior_smooth(
                transformed, cache, model_name, amplitude_bounds,
                slope_bounds, width_bounds, max_integrated_occurrence)):
            accepted.append(physical)
        attempts += 1
        if attempts > max(10000, 1000*size):
            raise ValueError(
                f"configured {model_name} prior has negligible valid volume"
            )
    return np.asarray(accepted)


def add_smooth_model_to_figures(
        chain_paths,
        model_name,
        figures,
        output_dir,
        stack_dim,
        m_unit="earth",
        title="Smooth-model fit",
        plot_occurrence=True,
        plot_cumulative=False,
        plot_density=True,
        plot_corner=True,
        max_curve_samples=10000,
        model_edges=None,
        model_plot_style="credible",
        n_posterior_draws=100,
        plot_random_seed=None):
    """Add one smooth model to shared figures without saving those figures."""
    from occurrence import plotting_utils as pu

    if stack_dim not in {"a", "m"}:
        raise ValueError("stack_dim must be 'a' or 'm'")
    if isinstance(chain_paths, (str, Path)):
        chain_paths = [chain_paths]
    if not chain_paths:
        raise ValueError("at least one smooth-model chain path is required")
    if model_plot_style not in {"credible", "draws"}:
        raise ValueError("model_plot_style must be 'credible' or 'draws'")
    if n_posterior_draws < 1:
        raise ValueError("n_posterior_draws must be positive")
    rng = np.random.RandomState(plot_random_seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = []
    for path in chain_paths:
        with np.load(path) as data:
            all_samples = np.asarray(data["flat_chains"])
            if "flat_log_probs" in data:
                log_likelihoods = (
                    np.asarray(data["flat_log_probs"]) -
                    _physical_log_jacobian(model_name, all_samples)
                )
                maximum_likelihood_sample = all_samples[
                    np.argmax(log_likelihoods)
                ]
            else:
                maximum_likelihood_sample = None
            if model_plot_style == "draws" and maximum_likelihood_sample is None:
                raise ValueError(
                    "draw plotting requires flat_log_probs in the chain file"
                )
            samples = all_samples[::10]
            if model_plot_style == "credible" and samples.shape[0] > max_curve_samples:
                indices = np.linspace(
                    0, samples.shape[0] - 1, max_curve_samples, dtype=int
                )
                samples = samples[indices]
            loaded.append({
                "path": str(path),
                "samples": samples,
                "maximum_likelihood_sample": maximum_likelihood_sample,
                "model_bounds": tuple(data["model_bounds"]),
                "stack_bounds": tuple(data["stack_bounds"]),
                "model_coordinate": str(data["model_coordinate"]),
                "stack_coordinate": str(data["stack_coordinate"]),
            })

    coordinate = loaded[0]["model_coordinate"]
    if any(item["model_coordinate"] != coordinate for item in loaded):
        raise ValueError("all model chains must use the same model coordinate")
    model_color = mcmc_powerlaw.MODEL_REGISTRY[model_name].color
    density_function = mcmc_powerlaw.get_model_spec(model_name).function
    paths = {}
    for stack_index, item in enumerate(loaded):
        grid = np.logspace(*np.log10(item["model_bounds"]), 300)
        samples = item["samples"]
        curves = np.asarray([
            density_function(sample, grid) for sample in samples
        ])
        label = model_name
        if len(loaded) > 1:
            label += " (" + _stack_interval_label(
                item["stack_coordinate"], item["stack_bounds"], m_unit
            ) + ")"
        if plot_density:
            axis = _axis_for_stack(figures["density"][1], stack_index)
            if model_plot_style == "credible":
                _plot_credible_curves(
                    axis, grid, curves, label, color=model_color
                )
            else:
                _plot_posterior_draws(
                    axis, grid, curves, item["maximum_likelihood_sample"],
                    n_posterior_draws, rng, label, density_function,
                    color=model_color,
                )
        if plot_occurrence:
            axis = _axis_for_stack(figures["occurrence"][1], stack_index)
            if model_edges is None:
                log_grid = np.log10(grid)
                increments = 0.5*(curves[:, 1:] + curves[:, :-1])*np.diff(log_grid)
                occurrence = np.column_stack([
                    np.zeros(curves.shape[0]), np.cumsum(increments, axis=1)
                ])
                occurrence *= np.log10(
                    item["stack_bounds"][1]/item["stack_bounds"][0]
                )
                _plot_credible_curves(
                    axis, grid, occurrence, label, color=model_color
                )
            else:
                _plot_smooth_binned_occurrence(
                    axis, np.asarray(model_edges), samples,
                    item["stack_bounds"], label, density_function,
                    color=model_color
                )
        if plot_cumulative:
            log_grid = np.log10(grid)
            increments = 0.5*(curves[:, 1:] + curves[:, :-1])*np.diff(log_grid)
            cumulative = np.column_stack([
                np.zeros(curves.shape[0]), np.cumsum(increments, axis=1)
            ])
            cumulative *= np.log10(
                item["stack_bounds"][1]/item["stack_bounds"][0]
            )
            _plot_credible_curves(
                _axis_for_stack(figures["cumulative"][1], stack_index),
                grid, cumulative, label, color=model_color,
            )

    if plot_corner:
        corner_paths = []
        for index, item in enumerate(loaded):
            path = output_dir / f"corner_{model_name}_bin{index}.png"
            pu.plot_corner_from_file(
                path_to_chains=item["path"],
                model_name=model_name,
                outpath=str(path),
                thin=10,
                max_samples=50000,
                reference_values=item["maximum_likelihood_sample"],
            )
            corner_paths.append(str(path))
        paths["corner"] = corner_paths
    return paths


def save_model_figures(
        figures, output_dir, stack_dim, model_edges, title,
        m_unit="earth"):
    """Format and save completed direct-model figures exactly once."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinate = "mass" if stack_dim == "a" else "sma"
    x_label = "Semimajor axis (AU)" if coordinate == "sma" else (
        "Companion mass ($M_{Jup}$)" if m_unit == "jupiter"
        else "Companion mass ($M_{Earth}$)"
    )
    plot_specs = {
        "density": (
            "Occurrence rate density\n"
            r"[Planets/star/$\Delta \log_{10}(\omega)$]",
            "occurrence_ORD.png",
        ),
        "occurrence": (
            "Occurrence rate (OR)\n[Planets/star]",
            "occurrence_OR_models.png",
        ),
        "cumulative": (
            "Cumulative OR\n[Planets/star]",
            "occurrence_CDF.png",
        ),
    }
    model_edges = np.asarray(model_edges, dtype=float)
    log_edges = np.log10(model_edges)
    padding = 0.025*(log_edges[-1] - log_edges[0])
    paths = {}
    for name, (figure, axes) in figures.items():
        ylabel, filename = plot_specs[name]
        for axis in np.atleast_1d(axes):
            axis.set_xscale("log")
            axis.set_xlim(
                10**(log_edges[0] - padding),
                10**(log_edges[-1] + padding),
            )
            axis.xaxis.set_major_locator(FixedLocator(model_edges))
            axis.xaxis.set_major_formatter(
                FuncFormatter(lambda value, position: f"{value:g}")
            )
            axis.xaxis.set_minor_locator(NullLocator())
            axis.legend()
        if np.ndim(axes) == 0:
            axes.set_xlabel(x_label)
            axes.set_ylabel(ylabel)
            axes.set_title(title)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=300)
        plt.close(figure)
        paths[name] = str(path)
    return paths


def _plot_credible_curves(axis, x_values, curves, label, color=None):
    """Add a posterior median and central 68 percent band to an axis."""
    low68, median, high68 = np.percentile(curves, [16, 50, 84], axis=0)
    line, = axis.plot(x_values, median, label=label, color=color)
    axis.fill_between(x_values, low68, high68, color=line.get_color(), alpha=0.28)


def _plot_posterior_draws(
        axis, x_values, curves, maximum_likelihood_sample,
        n_draws, rng, label, density_function, color=None):
    """Plot sampled posterior curves and emphasize the maximum-likelihood one."""
    replace = n_draws > curves.shape[0]
    indices = rng.choice(curves.shape[0], size=n_draws, replace=replace)
    if color is None:
        raise ValueError("a registered model color is required")
    for curve in curves[indices]:
        axis.plot(x_values, curve, color=color, alpha=0.08, linewidth=0.8)
    ml_curve = density_function(maximum_likelihood_sample, x_values)
    axis.plot(x_values, ml_curve, color=color, linewidth=2.5,
              label=f"{label}")


def _physical_log_jacobian(model_name, samples):
    """Return the sampling-transform Jacobian for saved physical samples."""
    if model_name == "logG":
        return np.log(samples[:, 0]) + np.log(samples[:, 2])
    if model_name == "escarpment":
        return np.log(samples[:, 0]) + np.log(samples[:, 1])
    if model_name == "sigmoid":
        return (
            np.log(samples[:, 0]) + np.log(samples[:, 1]) +
            np.log(samples[:, 3])
        )
    if model_name == "bpl":
        return np.log(samples[:, 0])
    raise ValueError(f"unsupported smooth model {model_name!r}")


def _plot_smooth_binned_occurrence(
        axis, edges, samples, stack_bounds, label, density_function,
        color=None):
    """Overlay any smooth model integrated over the histogram bins."""
    occurrence = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        grid = np.logspace(np.log10(lower), np.log10(upper), 100)
        weights = dl._trapezoid_weights(np.log10(grid))
        occurrence.append([
            np.dot(density_function(sample, grid), weights)
            for sample in samples
        ])
    occurrence = np.asarray(occurrence).T
    occurrence *= np.log10(stack_bounds[1]/stack_bounds[0])
    low, median, high = np.percentile(occurrence, [16, 50, 84], axis=0)
    line, = axis.step(
        edges, np.r_[median, median[-1]], where="post",
        label=label, color=color,
    )
    axis.fill_between(
        edges, np.r_[low, low[-1]], np.r_[high, high[-1]],
        step="post", color=line.get_color(), alpha=0.28,
    )


def _axis_for_stack(axes, index):
    """Select the plotting axis associated with a saved stack-bin chain."""
    axes = np.asarray(axes, dtype=object)
    return axes.item() if axes.ndim == 0 else axes[index]


def _stack_interval_label(coordinate, bounds, m_unit):
    """Return a concise legend label for one marginalized stack interval."""
    if coordinate == "sma":
        unit = "AU"
    else:
        unit = "$M_{Jup}$" if m_unit == "jupiter" else "$M_{Earth}$"
    return f"{bounds[0]:g}--{bounds[1]:g} {unit}"


def piecewise_cumulative_figure(
        chain_path, stack_dim, title="", m_unit="earth"):
    """Plot the cumulative piecewise posterior for every stack interval."""
    import matplotlib.pyplot as plt

    with np.load(chain_path) as data:
        samples = np.asarray(data["flat_chains"])[::10]
        a_edges = np.asarray(data["x_edges"])
        y_key = "y_edges" if "y_edges" in data else "stack_edges"
        m_edges = np.asarray(data[y_key])
    n_a, n_m = len(a_edges) - 1, len(m_edges) - 1
    densities = samples.reshape(-1, n_m, n_a)
    areas = np.outer(np.diff(np.log10(m_edges)), np.diff(np.log10(a_edges)))
    occurrence = densities*areas[None, :, :]
    if stack_dim == "a":
        model_edges, n_stack = m_edges, n_a
        coordinate_label = (
            "Companion mass ($M_{Jup}$)" if m_unit == "jupiter"
            else "Companion mass ($M_{Earth}$)"
        )
        cumulative_samples = [
            np.column_stack([
                np.zeros(samples.shape[0]),
                np.cumsum(occurrence[:, :, index], axis=1),
            ])
            for index in range(n_stack)
        ]
    elif stack_dim == "m":
        model_edges, n_stack = a_edges, n_m
        coordinate_label = "Semimajor axis (AU)"
        cumulative_samples = [
            np.column_stack([
                np.zeros(samples.shape[0]),
                np.cumsum(occurrence[:, index, :], axis=1),
            ])
            for index in range(n_stack)
        ]
    else:
        raise ValueError("stack_dim must be 'a' or 'm'")

    figure, axes = plt.subplots(
        n_stack, 1, figsize=(6, max(4, 2.4*n_stack)), sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0][::-1]
    for index, axis in enumerate(axes):
        _plot_credible_curves(
            axis, model_edges, cumulative_samples[index],
            "piecewise", color="black",
        )
        axis.set_xscale("log")
        axis.set_ylabel("Cumulative OR\n[Planets/star]")
    axes[0].set_xlabel(coordinate_label)
    axes[-1].set_title(title)
    return figure, axes


def summarize_piecewise_file(
        fit_path, chain_path, nstars, save_path=None):
    """Create plotting summaries directly from a piecewise-fit chain."""
    from occurrence import occurrence_utils as ou

    companions, exposure = dfu.load_fit_data(fit_path)
    with np.load(chain_path) as data:
        samples = data["flat_chains"]
        a_edges = data["x_edges"]
        y_key = "y_edges" if "y_edges" in data else "stack_edges"
        m_edges = data[y_key]
        if "cell_exposure" in data and "effective_counts" in data:
            cache = dl.PiecewiseLikelihoodCache(
                x_edges=a_edges,
                y_edges=m_edges,
                cell_areas=data["cell_areas"],
                cell_exposure=data["cell_exposure"],
                effective_counts=data["effective_counts"],
                log_weight_constant=float(data["log_weight_constant"]),
            )
        else:
            cache = dl.build_piecewise_cache(
                companions, exposure, a_edges, m_edges
            )
    samples = samples[::10]
    a_widths = np.diff(np.log10(a_edges))
    m_widths = np.diff(np.log10(m_edges))
    cell_areas = np.outer(m_widths, a_widths).ravel()
    if samples.shape[1] != cell_areas.size:
        raise ValueError("chain dimension does not match its saved bin edges")

    occurrence_samples = samples*cell_areas
    summary = {
        **ou.summarize_chains(samples, rate_type="ORD", hdi_frac=0.68, grid_size=1000),
        **ou.summarize_chains(
            occurrence_samples, rate_type="OR", hdi_frac=0.68, grid_size=1000
        ),
        **ou.summarize_chains(
            occurrence_samples.sum(axis=1, keepdims=True),
            rate_type="OR_single", hdi_frac=0.68, grid_size=1000,
        ),
    }
    summary.update(_piecewise_metadata(cache, a_edges, m_edges, nstars))
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, **summary)
    return summary


def plot_catalog_roi_completeness(
        tier1_dir, tier2_dir, output_dir, a_edges, m_edges,
        m_unit="earth", title="Occurrence fit"):
    """Plot the catalog and fitted ROI without requiring a fitted model."""
    from occurrence import plotting_utils as pu

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "catalog_inROI_and_completeness.png"
    pu.plot_catalog(
        tier1_dir=tier1_dir,
        tier2_dir=tier2_dir,
        catalog_path=os.path.join(
            tier1_dir, tier2_dir, "sampled_post_prior_compl.npz"
        ),
        a_edges=np.asarray(a_edges),
        m_edges=np.asarray(m_edges),
        zoom=False,
        m_unit=m_unit,
        fig_title=title,
        fig_savepath=str(output_path),
    )
    return str(output_path)


def plot_roi_occurrence_completeness(
        tier1_dir, tier2_dir, output_dir, summary,
        mtype="mtrue", m_unit="earth", title="Occurrence fit"):
    """Plot a piecewise occurrence summary over the average completeness map."""
    from occurrence import plotting_utils as pu

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ROI_with_occurrence.png"
    average_map_dir = os.path.join(tier1_dir, tier2_dir, "avg_map")
    xgrid = np.load(os.path.join(average_map_dir, "parent_xgrid.npy"))
    ygrid = np.load(os.path.join(average_map_dir, "parent_ygrid.npy"))
    zgrid = np.load(os.path.join(average_map_dir, "parent_zgrid.npy"))
    pu.completeness_plotter(
        xgrid=xgrid,
        ygrid=ygrid,
        zgrid=zgrid,
        save_path=str(output_path),
        title=title,
        save_plot=True,
        a_m_lims_pairs=summary["a_m_lims_pairs"],
        summary_dict=summary,
        zoom=True,
        ycol=f"inj_{mtype}",
        m_unit=m_unit,
    )
    return str(output_path)


def plot_piecewise_results(
        fit_path,
        chain_path,
        output_dir,
        nstars,
        stack_dim,
        m_unit="earth",
        mtype="mtrue",
        title="Piecewise-constant fit",
        plot_occurrence=True,
        plot_density=True,
        plot_corner=True,
        plot_catalog_roi=False,
        plot_roi_occurrence=False,
        tier1_dir=None,
        tier2_dir=None,
        summary=None):
    """Load a direct fit and save the requested result plots."""
    from occurrence import plotting_utils as pu

    if stack_dim not in {"a", "m"}:
        raise ValueError("stack_dim must be 'a' or 'm'")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir.parent / "saved_dicts" / "summary_dict_piecewise.npz"
    if summary is None:
        summary = summarize_piecewise_file(
            fit_path, chain_path, nstars, save_path=summary_path
        )
    or_path = output_dir / "occurrence_OR_models.png"
    ord_path = output_dir / "occurrence_ORD.png"
    corner_path = output_dir / "corner_piecewise.png"
    common = dict(
        summary_dict=summary, stack_dim=stack_dim, m_unit=m_unit,
        mtype=mtype, title=title,
    )
    paths = {"summary": str(summary_path)}
    if plot_occurrence:
        pu.plot_occurrence_hist(
            **common, rate_type="OR", savepath=str(or_path)
        )
        paths["occurrence"] = str(or_path)
    if plot_density:
        pu.plot_occurrence_hist(
            **common, rate_type="ORD", savepath=str(ord_path)
        )
        paths["density"] = str(ord_path)
    if plot_corner:
        maximum_likelihood = None
        if Path(chain_path).is_file():
            with np.load(chain_path) as chain_data:
                flat_chains = np.asarray(chain_data["flat_chains"])
                maximum_likelihood = flat_chains[
                    np.argmax(np.asarray(chain_data["flat_log_probs"]))
                ]
        pu.plot_corner_from_file(
            path_to_chains=chain_path,
            param_names=[
                f"$\\lambda_{{{index}}}$"
                for index in range(len(summary["mode_ORD"]))
            ],
            outpath=str(corner_path),
            thin=10,
            max_samples=50000,
            reference_values=maximum_likelihood,
        )
        paths["corner"] = str(corner_path)
    if plot_catalog_roi or plot_roi_occurrence:
        if tier1_dir is None or tier2_dir is None:
            raise ValueError(
                "tier1_dir and tier2_dir are required for completeness-map plots"
            )
    if plot_catalog_roi:
        cell_pairs = np.asarray(summary["a_m_lims_pairs"])
        a_edges = np.unique(cell_pairs[:, 0, :])
        m_edges = np.unique(cell_pairs[:, 1, :])
        paths["catalog_roi"] = plot_catalog_roi_completeness(
            tier1_dir=tier1_dir, tier2_dir=tier2_dir,
            output_dir=output_dir,
            a_edges=a_edges,
            m_edges=m_edges,
            m_unit=m_unit,
            title=title,
        )
    if plot_roi_occurrence:
        paths["roi_occurrence"] = plot_roi_occurrence_completeness(
            tier1_dir=tier1_dir, tier2_dir=tier2_dir,
            output_dir=output_dir,
            summary=summary,
            mtype=mtype,
            m_unit=m_unit,
            title=title,
        )
    return paths


def _piecewise_metadata(cache, a_edges, m_edges, nstars):
    """Calculate metadata required by the occurrence plotting functions."""
    cell_completeness = cache.cell_exposure/(nstars*cache.cell_areas)
    pairs = np.array([
        ((a_edges[a], a_edges[a + 1]), (m_edges[m], m_edges[m + 1]))
        for m in range(len(m_edges) - 1)
        for a in range(len(a_edges) - 1)
    ])
    total_area = np.sum(cache.cell_areas)
    return {
        "nstars": nstars,
        "cell_weights": cache.effective_counts,
        "cell_compls": cell_completeness,
        "cell_compl_single": np.sum(cache.cell_exposure)/(nstars*total_area),
        "a_m_lims_pairs": pairs,
        "n_abins": len(a_edges) - 1,
        "n_mbins": len(m_edges) - 1,
    }


def _run_sampler(sampler, pos, burnin, nsteps, rng, random_seed):
    """Run burn-in and production steps with optional reproducibility."""
    if random_seed is not None:
        sampler.random_state = rng.get_state()
    pos, _, _ = sampler.run_mcmc(pos, burnin, progress=True)
    sampler.reset()
    sampler.run_mcmc(pos, nsteps, progress=True)
