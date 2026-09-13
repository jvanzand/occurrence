"""Poisson point-process likelihoods for unbinned occurrence fitting."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PiecewiseLikelihoodCache:
    """Parameter-independent sufficient statistics for a piecewise fit."""

    x_edges: np.ndarray
    stack_edges: np.ndarray
    cell_areas: np.ndarray
    cell_exposure: np.ndarray
    effective_counts: np.ndarray
    log_weight_constant: float


@dataclass(frozen=True)
class SmoothLikelihoodCache:
    """Parameter-independent arrays for one smooth-model stack interval."""

    model_coordinate: str
    stack_coordinate: str
    model_bounds: tuple
    stack_bounds: tuple
    log_x_samples: np.ndarray
    log_weight_samples: np.ndarray
    companion_starts: np.ndarray
    companion_counts: np.ndarray
    companion_weights: np.ndarray
    companion_names: tuple
    log_x_grid: np.ndarray
    exposure_weights: np.ndarray


def log_gaussian_density(theta, x):
    """Evaluate ``A exp[-(log10(x)-mu)^2/(2 sigma^2)]``."""
    amplitude, mu, sigma = np.asarray(theta, dtype=float)
    x = np.asarray(x, dtype=float)
    return amplitude*np.exp(-0.5*((np.log10(x) - mu)/sigma)**2)


def escarpment_density(theta, x):
    """Evaluate two plateaus joined linearly in ``log10(x)``."""
    low, high, log_break1, log_break2 = np.asarray(theta, dtype=float)
    log_x = np.log10(np.asarray(x, dtype=float))
    fraction = np.clip(
        (log_x - log_break1)/(log_break2 - log_break1), 0.0, 1.0
    )
    return low + (high - low)*fraction


def sigmoid_density(theta, x):
    """Evaluate a logistic transition between two ORD plateaus.

    Parameters
    ----------
    theta : sequence of float
        ``(A, B, C, D)``, where ``A`` is the low-x occurrence-rate-density
        plateau, ``B`` is the high-x plateau, ``C`` is the transition midpoint
        in ``log10(x)``, and positive ``D`` is the transition width in dex.
        The density equals ``(A + B)/2`` at ``x = 10**C``.
    x : array-like
        Positive physical coordinate values at which to evaluate the model.

    Returns
    -------
    numpy.ndarray
        Occurrence-rate density at each supplied coordinate.
    """
    low, high, midpoint, width = np.asarray(theta, dtype=float)
    scaled = (np.log10(np.asarray(x, dtype=float)) - midpoint)/width
    transition = 1.0/(1.0 + np.exp(-np.clip(scaled, -700.0, 700.0)))
    return low + (high - low)*transition


def broken_powerlaw_density(theta, x):
    """Evaluate the registered broken power law with guarded exponentiation."""
    amplitude, log_break, beta, gamma = np.asarray(theta, dtype=float)
    x = np.asarray(x, dtype=float)
    log_ratio = np.log(x) - log_break*np.log(10.0)
    turnover_power = np.exp(np.clip(gamma*log_ratio, -700, 700))
    turnover = -np.expm1(-turnover_power)
    powerlaw = np.exp(np.clip(beta*np.log(x), -700, 700))
    return amplitude*powerlaw*turnover


def cached_smooth_log_likelihood(theta, cache, density_function):
    """Evaluate a smooth model using parameter-independent cached arrays."""
    grid_density = np.asarray(
        density_function(theta, 10**cache.log_x_grid), dtype=float
    )
    if (grid_density.shape != cache.log_x_grid.shape or
            not np.isfinite(grid_density).all() or np.any(grid_density < 0)):
        return -np.inf
    expected = np.dot(grid_density, cache.exposure_weights)
    if cache.companion_starts.size:
        sample_density = np.asarray(
            density_function(theta, 10**cache.log_x_samples), dtype=float
        )
        if (not np.isfinite(sample_density).all() or
                np.any(sample_density <= 0)):
            return -np.inf
        log_terms = np.log(sample_density) + cache.log_weight_samples
        grouped_sums = np.logaddexp.reduceat(log_terms, cache.companion_starts)
        grouped_means = grouped_sums - np.log(cache.companion_counts)
    else:
        grouped_means = np.array([])
    return -expected + np.dot(cache.companion_weights, grouped_means)


def build_smooth_cache(
        companions, exposure, stack_dim, stack_bounds):
    """Precompute samples and collapsed exposure for one smooth-model fit.

    Stage 2 stores physical coordinates as ``(SMA, mass)``. ``stack_dim``
    selects which coordinate is integrated over; the other is the model's
    independent variable.
    """
    if stack_dim not in {"a", "m"}:
        raise ValueError("stack_dim must be 'a' or 'm'")
    stack_bounds = _validate_bounds_pair(stack_bounds, "stack_bounds")
    if stack_dim == "a":
        model_coordinate, stack_coordinate = "mass", "sma"
        model_bounds = exposure.stack_bounds
        grid_model = exposure.stack_values[0, :]
        grid_stack = exposure.x_values[:, 0]
    else:
        model_coordinate, stack_coordinate = "sma", "mass"
        model_bounds = exposure.x_bounds
        grid_model = exposure.x_values[:, 0]
        grid_stack = exposure.stack_values[0, :]

    full_stack_bounds = exposure.x_bounds if stack_dim == "a" else exposure.stack_bounds
    if stack_bounds[0] < full_stack_bounds[0] or stack_bounds[1] > full_stack_bounds[1]:
        raise ValueError("stack_bounds must lie inside the exposure domain")
    exposure_weights = _collapse_exposure_over_stack(
        exposure, stack_dim, stack_bounds
    )

    sample_logs = []
    weight_logs = []
    starts = []
    counts = []
    overlap_weights = []
    names = []
    offset = 0
    for name, companion in companions.items():
        if stack_dim == "a":
            model_samples = companion.y_samples
            stacked_samples = companion.x_samples
        else:
            model_samples = companion.x_samples
            stacked_samples = companion.y_samples
        mask = companion.roi_mask.copy()
        mask &= (
            (stacked_samples > stack_bounds[0]) &
            (stacked_samples <= stack_bounds[1])
        )
        count = np.count_nonzero(mask)
        if count == 0:
            continue
        ratios = companion.completeness_over_prior[mask]
        if not np.isfinite(ratios).all() or np.any(ratios <= 0):
            raise ValueError(f"{name} has invalid completeness/prior in the fit interval")
        starts.append(offset)
        counts.append(count)
        overlap_weights.append(count/companion.original_sample_count)
        names.append(name)
        sample_logs.append(np.log10(model_samples[mask]))
        weight_logs.append(np.log(ratios))
        offset += count
    return SmoothLikelihoodCache(
        model_coordinate=model_coordinate,
        stack_coordinate=stack_coordinate,
        model_bounds=tuple(model_bounds),
        stack_bounds=stack_bounds,
        log_x_samples=(np.concatenate(sample_logs) if sample_logs else np.array([])),
        log_weight_samples=(np.concatenate(weight_logs) if weight_logs else np.array([])),
        companion_starts=np.asarray(starts, dtype=int),
        companion_counts=np.asarray(counts, dtype=int),
        companion_weights=np.asarray(overlap_weights, dtype=float),
        companion_names=tuple(names),
        log_x_grid=np.log10(grid_model),
        exposure_weights=np.asarray(exposure_weights, dtype=float),
    )


def _collapse_exposure_over_stack(exposure, stack_dim, stack_bounds):
    """Integrate completeness over a stack interval with explicit boundaries.

    Interpolated values at the requested boundaries prevent a stack bin from
    inheriting the full trapezoidal weight of a grid node outside that bin.
    The remaining model-coordinate trapezoidal weights are included in the
    returned one-dimensional exposure kernel.
    """
    log_a = np.log10(exposure.x_values[:, 0])
    log_m = np.log10(exposure.stack_values[0, :])
    log_bounds = np.log10(stack_bounds)
    completeness = exposure.completeness_sum
    if stack_dim == "a":
        collapsed = np.array([
            _bounded_trapezoid(log_a, completeness[:, index], log_bounds)
            for index in range(completeness.shape[1])
        ])
        return collapsed*_trapezoid_weights(log_m)
    collapsed = np.array([
        _bounded_trapezoid(log_m, completeness[index, :], log_bounds)
        for index in range(completeness.shape[0])
    ])
    return collapsed*_trapezoid_weights(log_a)


def _bounded_trapezoid(coordinates, values, bounds):
    """Integrate one sampled curve between arbitrary in-domain boundaries."""
    interior = (coordinates > bounds[0]) & (coordinates < bounds[1])
    selected_x = np.concatenate(([bounds[0]], coordinates[interior], [bounds[1]]))
    selected_y = np.interp(selected_x, coordinates, values)
    return np.trapz(selected_y, selected_x)


def _trapezoid_weights(coordinates):
    """Return one-dimensional trapezoidal quadrature weights."""
    coordinates = np.asarray(coordinates, dtype=float)
    weights = np.empty_like(coordinates)
    weights[0] = (coordinates[1] - coordinates[0])/2
    weights[-1] = (coordinates[-1] - coordinates[-2])/2
    weights[1:-1] = (coordinates[2:] - coordinates[:-2])/2
    return weights


def build_piecewise_cache(companions, exposure, x_edges, stack_edges):
    """Precompute all fixed terms in the piecewise-constant likelihood."""
    x_edges = _validate_edges(x_edges, "x_edges")
    stack_edges = _validate_edges(stack_edges, "stack_edges")
    x_widths = np.diff(np.log10(x_edges))
    stack_widths = np.diff(np.log10(stack_edges))
    cell_areas = np.outer(stack_widths, x_widths).ravel()
    cell_count = cell_areas.size

    exposure_indices = piecewise_cell_indices(
        exposure.x_values,
        exposure.stack_values,
        x_edges,
        stack_edges,
        include_lower_boundaries=True,
    )
    exposure_terms = exposure.completeness_sum*exposure.integration_weights
    inside_exposure = exposure_indices >= 0
    cell_exposure = np.bincount(
        exposure_indices[inside_exposure].ravel(),
        weights=exposure_terms[inside_exposure].ravel(),
        minlength=cell_count,
    )

    effective_counts = np.zeros(cell_count)
    log_weight_constant = 0.0
    for companion in companions.values():
        mask = companion.roi_mask
        if not np.any(mask):
            raise ValueError(
                f"{companion.name} has no posterior support inside the ROI"
            )
        indices = piecewise_cell_indices(
            companion.x_samples[mask],
            companion.y_samples[mask],
            x_edges,
            stack_edges,
        )
        ratios = companion.completeness_over_prior[mask]
        for cell_index in np.unique(indices[indices >= 0]):
            in_cell = indices == cell_index
            cell_weight = np.count_nonzero(in_cell)/companion.original_sample_count
            mean_ratio = np.mean(ratios[in_cell])
            if not np.isfinite(mean_ratio) or mean_ratio <= 0:
                raise ValueError(
                    f"{companion.name} has invalid completeness/prior in cell {cell_index}"
                )
            effective_counts[cell_index] += cell_weight
            log_weight_constant += cell_weight*np.log(mean_ratio)

    return PiecewiseLikelihoodCache(
        x_edges=x_edges,
        stack_edges=stack_edges,
        cell_areas=cell_areas,
        cell_exposure=cell_exposure,
        effective_counts=effective_counts,
        log_weight_constant=float(log_weight_constant),
    )


def cached_piecewise_log_likelihood(theta, cache):
    """Evaluate a piecewise likelihood from precomputed sufficient statistics."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != cache.cell_areas.shape:
        raise ValueError(f"piecewise model requires {cache.cell_areas.size} parameters")
    if not np.isfinite(theta).all() or np.any(theta < 0):
        return -np.inf
    occupied = cache.effective_counts > 0
    if np.any(theta[occupied] <= 0):
        return -np.inf
    return (
        -np.dot(theta, cache.cell_exposure)
        + np.dot(cache.effective_counts[occupied], np.log(theta[occupied]))
        + cache.log_weight_constant
    )


def piecewise_cell_indices(
        x, stack, x_edges, stack_edges, include_lower_boundaries=False):
    """Assign coordinates to flattened piecewise cells, or -1 outside."""
    x_edges = _validate_edges(x_edges, "x_edges")
    stack_edges = _validate_edges(stack_edges, "stack_edges")
    x, stack = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(stack, dtype=float)
    )
    x_index = np.searchsorted(x_edges, x, side="left") - 1
    stack_index = np.searchsorted(stack_edges, stack, side="left") - 1
    if include_lower_boundaries:
        x_index = np.where(x == x_edges[0], 0, x_index)
        stack_index = np.where(stack == stack_edges[0], 0, stack_index)
    inside = (
        (x_index >= 0) & (x_index < x_edges.size - 1) &
        (stack_index >= 0) & (stack_index < stack_edges.size - 1)
    )
    indices = np.full(x.shape, -1, dtype=int)
    indices[inside] = stack_index[inside]*(x_edges.size - 1) + x_index[inside]
    return indices


def _validate_edges(edges, label):
    """Return finite, positive, strictly increasing bin edges."""
    edges = np.asarray(edges, dtype=float)
    if (
            edges.ndim != 1 or edges.size < 2 or
            not np.isfinite(edges).all() or np.any(edges <= 0) or
            np.any(np.diff(edges) <= 0)
    ):
        raise ValueError(f"{label} must be finite, positive, and increasing")
    return edges


def _validate_bounds_pair(bounds, label):
    """Return two finite, positive, increasing bounds."""
    bounds = tuple(float(value) for value in bounds)
    if (
            len(bounds) != 2 or not np.isfinite(bounds).all() or
            bounds[0] <= 0 or bounds[1] <= bounds[0]
    ):
        raise ValueError(f"{label} must be finite, positive, and increasing")
    return bounds
