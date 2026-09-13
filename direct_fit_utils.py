"""Prepare unbinned inputs for direct parametric occurrence fitting.

This module does not evaluate a population likelihood. It converts the saved
posterior-sample product into named records and precomputes survey exposure
on a logarithmic integration grid.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple
import warnings

import numpy as np


@dataclass(frozen=True)
class CompanionSamples:
    """Unbinned samples and weights for one detected companion.

    All sample arrays have shape ``(n_samples,)``. ``roi_mask`` identifies
    draws in the fitted rectangle; samples are retained outside the ROI so the
    original Monte Carlo normalization remains available.
    """

    name: str
    x_samples: np.ndarray
    y_samples: np.ndarray
    completeness: np.ndarray
    interim_prior: np.ndarray
    completeness_over_prior: np.ndarray
    roi_mask: np.ndarray

    @property
    def original_sample_count(self):
        """Number of posterior draws before applying the ROI mask."""
        return self.x_samples.size


@dataclass(frozen=True)
class ExposureGrid:
    """Precomputed survey completeness and quadrature weights.

    Arrays have shape ``resolution``. ``completeness_sum`` is the sum of
    detection probabilities over surveyed stars. ``integration_weights`` are
    trapezoidal weights in ``dlog10(x) dlog10(stack)``.
    """

    x_values: np.ndarray
    stack_values: np.ndarray
    completeness_sum: np.ndarray
    integration_weights: np.ndarray
    x_bounds: Tuple[float, float]
    stack_bounds: Tuple[float, float]


def _validate_bounds(bounds, label):
    """Return two positive, increasing bounds as floats."""
    if len(bounds) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    lower, upper = (float(value) for value in bounds)
    if not np.isfinite([lower, upper]).all() or lower <= 0 or upper <= lower:
        raise ValueError(f"{label} must be finite, positive, and increasing")
    return lower, upper


def prepare_companion_samples(
        name,
        x_samples,
        y_samples,
        completeness,
        interim_prior,
        x_bounds,
        stack_bounds):
    """Validate and package the posterior draws for one companion.

    ROI membership uses lower-open and upper-closed bounds so adjacent regions
    upper bounds are closed. Invalid values outside the ROI are retained but
    excluded by ``roi_mask``; invalid values inside the ROI raise an error.
    """
    x_bounds = _validate_bounds(x_bounds, "x_bounds")
    stack_bounds = _validate_bounds(stack_bounds, "stack_bounds")
    arrays = [
        np.asarray(values, dtype=float)
        for values in (x_samples, y_samples, completeness, interim_prior)
    ]
    lengths = {array.size for array in arrays}
    if len(lengths) != 1 or any(array.ndim != 1 for array in arrays):
        raise ValueError("companion sample fields must be one-dimensional and equal length")
    if not arrays[0].size:
        raise ValueError("companion sample fields cannot be empty")

    x, y, completeness, interim_prior = arrays
    roi_mask = (
        (x > x_bounds[0]) & (x <= x_bounds[1]) &
        (y > stack_bounds[0]) & (y <= stack_bounds[1])
    )
    finite = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(completeness) &
        np.isfinite(interim_prior)
    )
    if np.any(roi_mask & ~finite):
        raise ValueError(f"{name} has nonfinite values inside the ROI")
    if np.any(roi_mask & (interim_prior <= 0)):
        raise ValueError(f"{name} has a nonpositive interim prior inside the ROI")
    probability_tolerance = 1e-12
    if np.any(
            roi_mask & (
                (completeness < -probability_tolerance) |
                (completeness > 1 + probability_tolerance)
            )):
        raise ValueError(f"{name} has completeness outside [0, 1] inside the ROI")
    # Interpolation can overshoot probability bounds at machine precision.
    completeness = np.clip(completeness, 0.0, 1.0)

    completeness_over_prior = np.full(x.size, np.nan)
    valid_weight = finite & (interim_prior > 0)
    completeness_over_prior[valid_weight] = (
        completeness[valid_weight] / interim_prior[valid_weight]
    )
    roi_mask &= finite

    return CompanionSamples(
        name=str(name),
        x_samples=x,
        y_samples=y,
        completeness=completeness,
        interim_prior=interim_prior,
        completeness_over_prior=completeness_over_prior,
        roi_mask=roi_mask,
    )


def prepare_catalog(
        catalog,
        x_bounds,
        stack_bounds,
        completeness_type="single",
        interim_prior_fn: Optional[Callable] = None):
    """Convert the current seven-row catalog format into named sample records.

    Rows 0 and 1 are the fitted and stack coordinates. Rows 2/3 contain the
    average and host-specific completeness, and row 6 contains the interim
    prior. Priors entering the likelihood are interpreted as densities with
    respect to ``dlog10(x) dlog10(stack)``. Existing catalogs are not converted;
    they must already follow this convention.
    """
    if completeness_type not in {"average", "single"}:
        raise ValueError("completeness_type must be 'average' or 'single'")
    completeness_row = 2 if completeness_type == "average" else 3
    use_stored_prior = interim_prior_fn is None
    if interim_prior_fn is None:
        interim_prior_fn = lambda x, stack: np.ones_like(x, dtype=float)

    prepared = {}
    outside_roi = []
    for name, values in catalog.items():
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[0] < 6:
            raise ValueError(
                f"{name} must have at least six rows and one column per sample"
            )
        if values.shape[0] < 7 and use_stored_prior:
            raise ValueError(
                f"{name} does not store an explicit log-measure interim prior; "
                "regenerate the catalog or supply interim_prior_fn"
            )
        if values.shape[1] == 0:
            warnings.warn(
                f"{name} has no retained posterior draws and was omitted",
                RuntimeWarning,
            )
            continue
        x_samples, y_samples = values[:2]
        if values.shape[0] >= 7 and use_stored_prior:
            interim_prior = values[6]
        else:
            interim_prior = np.asarray(
                interim_prior_fn(x_samples, y_samples), dtype=float
            )
        if interim_prior.shape != x_samples.shape:
            raise ValueError("interim_prior_fn must return one value per posterior draw")
        record = prepare_companion_samples(
            name=name,
            x_samples=x_samples,
            y_samples=y_samples,
            completeness=values[completeness_row],
            interim_prior=interim_prior,
            x_bounds=x_bounds,
            stack_bounds=stack_bounds,
        )
        if not np.any(record.roi_mask):
            outside_roi.append(name)
            continue
        prepared[name] = record
    if outside_roi:
        warnings.warn(
            f"omitted {len(outside_roi)} companions with no posterior support "
            f"inside the ROI: {', '.join(outside_roi)}",
            RuntimeWarning,
        )
    return prepared


def load_catalog(
        catalog_path,
        x_bounds,
        stack_bounds,
        completeness_type="single",
        interim_prior_fn=None):
    """Load and prepare an existing ``sampled_post_prior_compl.npz`` file."""
    with np.load(catalog_path) as data:
        catalog = {name: data[name] for name in data.files}
    return prepare_catalog(
        catalog=catalog,
        x_bounds=x_bounds,
        stack_bounds=stack_bounds,
        completeness_type=completeness_type,
        interim_prior_fn=interim_prior_fn,
    )


def _trapezoid_weights(coordinates):
    """Return one-dimensional trapezoidal integration weights."""
    coordinates = np.asarray(coordinates, dtype=float)
    weights = np.empty_like(coordinates)
    weights[0] = (coordinates[1] - coordinates[0])/2
    weights[-1] = (coordinates[-1] - coordinates[-2])/2
    weights[1:-1] = (coordinates[2:] - coordinates[:-2])/2
    return weights


def build_exposure_grid(
        x_bounds,
        stack_bounds,
        resolution=(100, 100),
        completeness_interpolators: Optional[Iterable[Callable]] = None,
        average_completeness: Optional[Callable] = None,
        nstars: Optional[int] = None):
    """Precompute summed completeness and weights on a log-spaced grid.

    Supply either individual stellar interpolators or an average interpolator
    together with ``nstars``. The former retains star-by-star map coverage;
    the latter reproduces the current average-map approximation.
    """
    x_bounds = _validate_bounds(x_bounds, "x_bounds")
    stack_bounds = _validate_bounds(stack_bounds, "stack_bounds")
    if len(resolution) != 2 or min(resolution) < 2:
        raise ValueError("resolution must contain two integers of at least 2")
    resolution = tuple(int(value) for value in resolution)

    using_individual = completeness_interpolators is not None
    using_average = average_completeness is not None or nstars is not None
    if using_individual == using_average:
        raise ValueError(
            "supply either completeness_interpolators or average_completeness and nstars"
        )
    if using_average and (average_completeness is None or nstars is None or nstars <= 0):
        raise ValueError("average_completeness requires a positive nstars")

    log_x = np.linspace(np.log10(x_bounds[0]), np.log10(x_bounds[1]), resolution[0])
    log_stack = np.linspace(
        np.log10(stack_bounds[0]), np.log10(stack_bounds[1]), resolution[1]
    )
    x_values, stack_values = np.meshgrid(
        10**log_x, 10**log_stack, indexing="ij"
    )

    if using_individual:
        interpolators = list(completeness_interpolators)
        if not interpolators:
            raise ValueError("completeness_interpolators cannot be empty")
        completeness_sum = np.zeros(resolution, dtype=float)
        for interpolator in interpolators:
            completeness_sum += np.asarray(
                interpolator((x_values, stack_values)), dtype=float
            )
    else:
        completeness_sum = nstars*np.asarray(
            average_completeness((x_values, stack_values)), dtype=float
        )

    if completeness_sum.shape != resolution:
        raise ValueError("a completeness interpolator returned the wrong shape")
    if not np.isfinite(completeness_sum).all() or np.any(completeness_sum < 0):
        raise ValueError("summed completeness must be finite and nonnegative")

    integration_weights = np.outer(
        _trapezoid_weights(log_x), _trapezoid_weights(log_stack)
    )
    return ExposureGrid(
        x_values=x_values,
        stack_values=stack_values,
        completeness_sum=completeness_sum,
        integration_weights=integration_weights,
        x_bounds=x_bounds,
        stack_bounds=stack_bounds,
    )


def save_direct_fit_data(path, companions, exposure):
    """Save prepared direct-fit inputs to a compressed, pickle-free NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(companions)
    arrays = {
        "companion_names": np.asarray(names),
        "x_values": exposure.x_values,
        "stack_values": exposure.stack_values,
        "completeness_sum": exposure.completeness_sum,
        "integration_weights": exposure.integration_weights,
        "x_bounds": np.asarray(exposure.x_bounds),
        "stack_bounds": np.asarray(exposure.stack_bounds),
    }
    for index, name in enumerate(names):
        record = companions[name]
        prefix = f"companion_{index}"
        arrays[f"{prefix}_x_samples"] = record.x_samples
        arrays[f"{prefix}_y_samples"] = record.y_samples
        arrays[f"{prefix}_completeness"] = record.completeness
        arrays[f"{prefix}_interim_prior"] = record.interim_prior
        arrays[f"{prefix}_completeness_over_prior"] = record.completeness_over_prior
        arrays[f"{prefix}_roi_mask"] = record.roi_mask
    np.savez_compressed(path, **arrays)


def load_direct_fit_data(path):
    """Load a file written by :func:`save_direct_fit_data`."""
    with np.load(path, allow_pickle=False) as data:
        names = [str(name) for name in data["companion_names"]]
        companions: Dict[str, CompanionSamples] = {}
        for index, name in enumerate(names):
            prefix = f"companion_{index}"
            y_key = f"{prefix}_y_samples"
            legacy_stack_key = f"{prefix}_stack_samples"
            if y_key not in data and legacy_stack_key not in data:
                raise KeyError(f"missing y samples for {name}")
            companions[name] = CompanionSamples(
                name=name,
                x_samples=data[f"{prefix}_x_samples"],
                y_samples=data[y_key if y_key in data else legacy_stack_key],
                completeness=data[f"{prefix}_completeness"],
                interim_prior=data[f"{prefix}_interim_prior"],
                completeness_over_prior=data[f"{prefix}_completeness_over_prior"],
                roi_mask=data[f"{prefix}_roi_mask"],
            )
        exposure = ExposureGrid(
            x_values=data["x_values"],
            stack_values=data["stack_values"],
            completeness_sum=data["completeness_sum"],
            integration_weights=data["integration_weights"],
            x_bounds=tuple(data["x_bounds"]),
            stack_bounds=tuple(data["stack_bounds"]),
        )
    return companions, exposure
