"""Focused tests for the Stage 3 unbinned direct likelihood."""

import numpy as np
import pytest

from occurrence import direct_fit_utils as dfu
from occurrence import direct_likelihood as dl


def _exposure(completeness_sum=2.0):
    """Return a grid whose total logarithmic area is one dex squared."""
    x_values, stack_values = np.meshgrid([1.0, 10.0], [1.0, 10.0], indexing="ij")
    return dfu.ExposureGrid(
        x_values=x_values,
        stack_values=stack_values,
        completeness_sum=np.full((2, 2), completeness_sum),
        integration_weights=np.full((2, 2), 0.25),
        x_bounds=(1.0, 10.0),
        stack_bounds=(1.0, 10.0),
    )


def _companion(name, completeness, prior=None, x_samples=None):
    """Create a small companion record with all draws inside the fit region."""
    completeness = np.asarray(completeness, dtype=float)
    sample_count = completeness.size
    if prior is None:
        prior = np.ones(sample_count)
    if x_samples is None:
        x_samples = np.full(sample_count, 2.0)
    return dfu.prepare_companion_samples(
        name=name,
        x_samples=x_samples,
        stack_samples=np.full(sample_count, 2.0),
        completeness=completeness,
        interim_prior=prior,
        x_bounds=(1.0, 10.0),
        stack_bounds=(1.0, 10.0),
    )


def test_cached_piecewise_likelihood_uses_precomputed_statistics():
    """Piecewise likelihood should use finite precomputed cell statistics."""
    exposure = _exposure(completeness_sum=2.0)
    companion = _companion(
        "b",
        completeness=[0.5, 1.0, 0.8, 0.2],
        prior=[0.5, 0.5, 0.4, 0.2],
        x_samples=[2.0, 2.0, 7.0, 20.0],
    )
    theta = np.array([0.2, 0.8])
    a_edges = np.array([1.0, 5.0, 10.0])
    m_edges = np.array([1.0, 10.0])
    cache = dl.build_piecewise_cache(
        {"b": companion}, exposure, a_edges, m_edges
    )
    cached = dl.cached_piecewise_log_likelihood(theta, cache)
    assert np.isfinite(cached)
    np.testing.assert_allclose(cache.effective_counts, [0.5, 0.25])


def test_log_gaussian_density_peaks_at_ten_to_mu():
    """The physical location parameter should identify the peak coordinate."""
    values = dl.log_gaussian_density([0.7, 1.2, 0.3], [1.0, 10**1.2, 100.0])
    assert values[1] == pytest.approx(0.7)
    assert values[1] > values[0]
    assert values[1] > values[2]


def test_smooth_cache_selects_mass_when_stacking_sma():
    """stack_dim='a' should collapse SMA and retain mass as model coordinate."""
    exposure = _exposure()
    companion = dfu.prepare_companion_samples(
        name="b", x_samples=[2.0, 8.0], stack_samples=[3.0, 7.0],
        completeness=[1.0, 1.0], interim_prior=[1.0, 1.0],
        x_bounds=(1.0, 10.0), stack_bounds=(1.0, 10.0),
    )
    cache = dl.build_smooth_cache(
        {"b": companion}, exposure, stack_dim="a", stack_bounds=(1.0, 10.0)
    )
    assert cache.model_coordinate == "mass"
    assert cache.stack_coordinate == "sma"
    np.testing.assert_allclose(cache.log_x_samples, np.log10([3.0, 7.0]))


def test_smooth_likelihood_is_invariant_to_duplicated_draws():
    """Duplicating posterior draws must not alter their Monte Carlo average."""
    exposure = _exposure()
    one = _companion("b", completeness=[0.5], x_samples=[2.0])
    two = _companion("b", completeness=[0.5, 0.5], x_samples=[2.0, 2.0])
    theta = np.array([0.4, np.log10(2.0), 0.2])
    cache_one = dl.build_smooth_cache(
        {"b": one}, exposure, "m", (1.0, 10.0)
    )
    cache_two = dl.build_smooth_cache(
        {"b": two}, exposure, "m", (1.0, 10.0)
    )
    assert dl.cached_smooth_log_likelihood(
        theta, cache_one, dl.log_gaussian_density
    ) == pytest.approx(
        dl.cached_smooth_log_likelihood(
            theta, cache_two, dl.log_gaussian_density
        )
    )


@pytest.mark.parametrize("sigma", [0.03, 0.8])
def test_log_gaussian_likelihood_handles_narrow_and_broad_widths(sigma):
    """Both narrow and broad permitted models should have finite likelihoods."""
    companions, exposure = {"b": _companion("b", [1.0])}, _exposure()
    cache = dl.build_smooth_cache(companions, exposure, "m", (1.0, 10.0))
    theta = [0.2, np.log10(2.0), sigma]
    assert np.isfinite(dl.cached_smooth_log_likelihood(
        theta, cache, dl.log_gaussian_density
    ))


def test_smooth_exposure_kernel_converges_with_grid_resolution():
    """The pre-collapsed exposure integral should converge on finer grids."""
    companion = _companion("b", [1.0])

    def expected(resolution):
        values = np.logspace(0, 1, resolution)
        x_values, stack_values = np.meshgrid(values, values, indexing="ij")
        exposure = dfu.ExposureGrid(
            x_values=x_values, stack_values=stack_values,
            completeness_sum=1 + 0.2*np.log10(x_values),
            integration_weights=np.ones_like(x_values),
            x_bounds=(1.0, 10.0), stack_bounds=(1.0, 10.0),
        )
        cache = dl.build_smooth_cache(
            {"b": companion}, exposure, "m", (1.17, 8.43)
        )
        shape = np.exp(-0.5*((cache.log_x_grid - 0.45)/0.18)**2)
        return np.dot(shape, cache.exposure_weights)

    coarse, fine = expected(31), expected(301)
    assert coarse == pytest.approx(fine, rel=3e-3)


def test_smooth_likelihood_allows_an_empty_stack_interval():
    """A bin without detections should still constrain models through exposure."""
    companion = _companion("b", [1.0])
    cache = dl.build_smooth_cache(
        {"b": companion}, _exposure(), "m", (5.0, 10.0)
    )
    assert cache.companion_names == ()
    value = dl.cached_smooth_log_likelihood(
        [0.2, 0.5, 0.2], cache, dl.log_gaussian_density
    )
    assert np.isfinite(value)
    assert value < 0


@pytest.mark.parametrize(
    "model_name, theta",
    [
        ("escarpment", [0.1, 0.5, 0.25, 0.75]),
        ("sigmoid", [0.1, 0.5, 0.5, 0.1]),
        ("bpl", [0.2, 0.5, -0.4, 1.2]),
    ],
)
def test_direct_smooth_densities_match_registered_models(model_name, theta):
    """Direct evaluators should preserve the established model definitions."""
    from occurrence import mcmc_powerlaw
    x = np.logspace(0, 1, 20)
    direct_function = {
        "escarpment": dl.escarpment_density,
        "sigmoid": dl.sigmoid_density,
        "bpl": dl.broken_powerlaw_density,
    }[model_name]
    registered = mcmc_powerlaw.MODEL_REGISTRY[model_name].function
    np.testing.assert_allclose(direct_function(theta, x), registered(theta, x))


def test_cached_smooth_likelihood_is_finite_for_new_models():
    """New smooth models should use the common precomputed likelihood inputs."""
    cache = dl.build_smooth_cache(
        {"b": _companion("b", [1.0])}, _exposure(), "m", (1.0, 10.0)
    )
    escarpment = dl.cached_smooth_log_likelihood(
        [0.1, 0.3, 0.2, 0.8], cache, dl.escarpment_density
    )
    bpl = dl.cached_smooth_log_likelihood(
        [0.2, 0.5, -0.5, 1.0], cache, dl.broken_powerlaw_density
    )
    sigmoid = dl.cached_smooth_log_likelihood(
        [0.1, 0.3, 0.5, 0.1], cache, dl.sigmoid_density
    )
    assert np.isfinite(escarpment)
    assert np.isfinite(bpl)
    assert np.isfinite(sigmoid)


def test_sigmoid_plateaus_and_midpoint():
    """Sigmoid parameters should retain their stated physical meanings."""
    values = dl.sigmoid_density([0.1, 0.5, 1.0, 0.1], [1e-6, 10.0, 1e6])
    assert values[0] == pytest.approx(0.1)
    assert values[1] == pytest.approx(0.3)
    assert values[2] == pytest.approx(0.5)
