"""Tests for direct piecewise-constant MCMC fitting."""

import numpy as np
import pytest
from scipy.optimize import minimize

from occurrence import fit_utils as dfu
from occurrence import mcmc


def _materials():
    x_values, y_values = np.meshgrid([1.0, 10.0], [1.0, 10.0], indexing="ij")
    exposure = dfu.ExposureGrid(
        x_values=x_values,
        y_values=y_values,
        completeness_sum=np.ones((2, 2)),
        integration_weights=np.full((2, 2), 0.25),
        x_bounds=(1.0, 10.0),
        y_bounds=(1.0, 10.0),
    )
    companion = dfu.prepare_companion_samples(
        name="b",
        x_samples=[2.0],
        y_samples=[2.0],
        completeness=[0.5],
        interim_prior=[1.0],
        x_bounds=(1.0, 10.0),
        y_bounds=(1.0, 10.0),
    )
    return {"b": companion}, exposure


def test_piecewise_sampler_is_reproducible(tmp_path):
    """A fixed random seed should reproduce direct-fit chains exactly."""
    companions, exposure = _materials()

    def run_once(filename):
        sampler = mcmc.mcmc_piecewise(
            companions=companions,
            exposure=exposure,
            x_edges=[1.0, 10.0],
            y_edges=[1.0, 10.0],
            nwalkers=4,
            nsteps=8,
            burnin=5,
            save_path=tmp_path / filename,
            random_seed=1234,
        )
        return sampler.get_chain(), sampler.get_log_prob()

    chain_a, probability_a = run_once("a.npz")
    chain_b, probability_b = run_once("b.npz")
    np.testing.assert_array_equal(chain_a, chain_b)
    np.testing.assert_array_equal(probability_a, probability_b)


def test_piecewise_sampler_requires_matching_fit_domain():
    """Piecewise edges and Stage 2 exposure must describe the same region."""
    companions, exposure = _materials()
    with pytest.raises(ValueError, match="x_edges"):
        mcmc.mcmc_piecewise(
            companions=companions,
            exposure=exposure,
            x_edges=[2.0, 10.0],
            y_edges=[1.0, 10.0],
            nwalkers=4,
            nsteps=2,
            burnin=2,
        )


def test_piecewise_sampler_rejects_unsupported_companion(tmp_path):
    """MCMC should fail before sampling if a companion has no ROI support."""
    companions, exposure = _materials()
    outside = dfu.prepare_companion_samples(
        name="outside",
        x_samples=[20.0], y_samples=[2.0], completeness=[0.5],
        interim_prior=[1.0], x_bounds=(1.0, 10.0), y_bounds=(1.0, 10.0),
    )
    companions["outside"] = outside
    with pytest.raises(ValueError, match="no posterior support"):
        mcmc.mcmc_piecewise(
            companions, exposure, [1.0, 10.0], [1.0, 10.0],
            nwalkers=4, nsteps=2, burnin=2,
            save_path=tmp_path / "invalid.npz",
        )


def test_piecewise_plotting_produces_all_requested_outputs(tmp_path, monkeypatch):
    """The plotting workflow should request OR, ORD, and corner figures."""
    calls = []
    summary = {
        "mode_ORD": np.array([0.2]),
        "hdi_low_ORD": np.array([0.1]),
        "hdi_high_ORD": np.array([0.3]),
    }
    monkeypatch.setattr(
        mcmc, "summarize_piecewise_file",
        lambda *args, **kwargs: summary,
    )
    from occurrence import plotting_utils
    monkeypatch.setattr(
        plotting_utils, "plot_occurrence_hist",
        lambda **kwargs: calls.append(kwargs["rate_type"]),
    )
    monkeypatch.setattr(
        plotting_utils, "plot_corner_from_file",
        lambda **kwargs: calls.append("corner"),
    )
    paths = mcmc.plot_piecewise_results(
        fit_path="materials.npz",
        chain_path="chains.npz",
        output_dir=tmp_path / "plots",
        nstars=10,
        stack_dim="a",
        m_unit="jupiter",
    )
    assert calls == ["OR", "ORD", "corner"]
    assert set(paths) == {"summary", "occurrence", "density", "corner"}


def test_piecewise_plotting_loads_catalog_and_completeness_inputs(
        tmp_path, monkeypatch):
    """Optional ingredient plots should load their Tier 2 products internally."""
    summary = {
        "mode_ORD": np.array([0.2]),
        "a_m_lims_pairs": np.array([[[1.0, 10.0], [1.0, 10.0]]]),
    }
    monkeypatch.setattr(
        mcmc, "summarize_piecewise_file",
        lambda *args, **kwargs: summary,
    )
    from occurrence import plotting_utils
    calls = []
    monkeypatch.setattr(
        plotting_utils, "plot_catalog",
        lambda **kwargs: calls.append(("catalog", kwargs)),
    )
    monkeypatch.setattr(
        plotting_utils, "completeness_plotter",
        lambda **kwargs: calls.append(("roi", kwargs)),
    )
    tier1 = tmp_path / "mtrue"
    tier2 = "allstars"
    average_map = tier1 / tier2 / "avg_map"
    average_map.mkdir(parents=True)
    for name in ("parent_xgrid.npy", "parent_ygrid.npy", "parent_zgrid.npy"):
        np.save(average_map / name, np.ones(2))

    paths = mcmc.plot_piecewise_results(
        fit_path="materials.npz",
        chain_path="chains.npz",
        output_dir=tmp_path / "output" / "plots",
        nstars=10,
        stack_dim="a",
        plot_occurrence=False,
        plot_density=False,
        plot_corner=False,
        plot_catalog_roi=True,
        plot_roi_occurrence=True,
        tier1_dir=str(tier1),
        tier2_dir=tier2,
    )

    assert [name for name, _ in calls] == ["catalog", "roi"]
    assert set(paths) == {"summary", "catalog_roi", "roi_occurrence"}
    assert calls[1][1]["summary_dict"] is summary


def test_logg_prior_transform_includes_physical_uniform_jacobians():
    """Log-coordinate sampling should retain priors uniform in A and sigma."""
    companions, exposure = _materials()
    cache = mcmc.dl.build_smooth_cache(
        companions, exposure, "m", (1.0, 10.0)
    )
    first = np.array([np.log(0.1), 0.5, np.log(0.1)])
    second = np.array([np.log(0.2), 0.5, np.log(0.2)])
    difference = mcmc.log_prior_smooth(
        second, cache, "logG", max_integrated_occurrence=None
    ) - mcmc.log_prior_smooth(
        first, cache, "logG", max_integrated_occurrence=None
    )
    assert difference == pytest.approx(2*np.log(2.0))


def test_logg_prior_predictive_draws_respect_all_bounds():
    """Prior draws should obey physical bounds and the occurrence constraint."""
    companions, exposure = _materials()
    cache = mcmc.dl.build_smooth_cache(
        companions, exposure, "m", (1.0, 10.0)
    )
    draws = mcmc.sample_smooth_prior(
        cache, "logG", 50, amplitude_bounds=(0.01, 2.0),
        width_bounds=(0.05, 0.8), max_integrated_occurrence=0.5,
        random_seed=9,
    )
    assert np.all((draws[:, 0] >= 0.01) & (draws[:, 0] <= 2.0))
    assert np.all((draws[:, 2] >= 0.05) & (draws[:, 2] <= 0.8))
    for draw in draws:
        transformed = [np.log(draw[0]), draw[1], np.log(draw[2])]
        assert mcmc.integrated_smooth_occurrence(
            "logG", transformed, cache
        ) <= 0.5


def test_logg_sampler_saves_physical_reproducible_chains(tmp_path):
    """Seeded logG fits should save positive physical A and sigma chains."""
    companions, exposure = _materials()
    cache = mcmc.dl.build_smooth_cache(
        companions, exposure, "m", (1.0, 10.0)
    )
    paths = [tmp_path / "first.npz", tmp_path / "second.npz"]
    for path in paths:
        mcmc.mcmc_smooth(
            cache, "logG", nwalkers=8, nsteps=8, burnin=5,
            save_path=path, random_seed=123,
            max_integrated_occurrence=None,
        )
    with np.load(paths[0]) as first, np.load(paths[1]) as second:
        np.testing.assert_array_equal(first["chains"], second["chains"])
        assert np.all(first["chains"][..., [0, 2]] > 0)
        assert first["chains"].shape[-1] == 3


def test_logg_plotting_requests_credible_curves_and_each_corner(tmp_path, monkeypatch):
    """Plot loading should create smooth curves and one corner per stack bin."""
    chain_paths = []
    for index, bounds in enumerate(((1.0, 3.0), (3.0, 10.0))):
        path = tmp_path / f"chain{index}.npz"
        samples = np.tile([0.2, 0.5, 0.2], (100, 1))
        np.savez(
            path, flat_chains=samples, model_bounds=[1.0, 10.0],
            stack_bounds=bounds, model_coordinate="sma",
            stack_coordinate="mass", flat_log_probs=np.zeros(100),
        )
        chain_paths.append(path)
    from occurrence import plotting_utils
    corner_calls = []
    monkeypatch.setattr(
        plotting_utils, "plot_corner_from_file",
        lambda **kwargs: corner_calls.append(kwargs),
    )
    import matplotlib.pyplot as plt
    figures = {
        "density": plt.subplots(), "occurrence": plt.subplots(),
    }
    paths = mcmc.add_smooth_model_to_figures(
        chain_paths, "logG", figures, tmp_path / "plots", stack_dim="m"
    )
    paths.update(mcmc.save_model_figures(
        figures, tmp_path / "plots", "m", [1.0, 10.0], "test",
    ))
    assert set(paths) == {"density", "occurrence", "corner"}
    assert len(paths["corner"]) == 2
    assert all(call["model_name"] == "logG" for call in corner_calls)
    assert (tmp_path / "plots" / "occurrence_ORD.png").exists()


def test_logg_plotting_option_adds_cumulative_distribution(tmp_path):
    """The cumulative logG plot should be independently selectable."""
    path = tmp_path / "chain.npz"
    samples = np.tile([0.2, 0.5, 0.2], (20, 1))
    np.savez(
        path, flat_chains=samples, flat_log_probs=np.zeros(20),
        model_bounds=[1.0, 10.0], stack_bounds=[1.0, 10.0],
        model_coordinate="sma", stack_coordinate="mass",
    )
    import matplotlib.pyplot as plt
    figures = {"cumulative": plt.subplots()}
    paths = mcmc.add_smooth_model_to_figures(
        [path], "logG", figures, tmp_path / "plots", stack_dim="m",
        plot_occurrence=False, plot_density=False, plot_corner=False,
        plot_cumulative=True,
    )
    paths.update(mcmc.save_model_figures(
        figures, tmp_path / "plots", "m", [1.0, 10.0], "test",
    ))
    assert set(paths) == {"cumulative"}
    assert (tmp_path / "plots" / "occurrence_CDF.png").exists()


def test_piecewise_cumulative_figure_integrates_successive_bins(tmp_path):
    """The piecewise CDF should cumulatively sum occurrence across model bins."""
    import matplotlib.pyplot as plt
    path = tmp_path / "piecewise.npz"
    samples = np.tile([1.0, 2.0], (20, 1))
    np.savez(
        path, flat_chains=samples, x_edges=[1.0, 10.0],
        y_edges=[1.0, np.sqrt(10.0), 10.0],
    )
    figure, axes = mcmc.piecewise_cumulative_figure(
        path, stack_dim="a"
    )
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [0.0, 0.5, 1.5])
    assert axes[0].lines[0].get_label() == "piecewise"
    plt.close(figure)


def test_logg_credible_curve_has_only_one_sigma_band():
    """Smooth plots should contain a median line and only the 68-percent band."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots()
    curves = np.arange(50, dtype=float)[:, None]*np.ones((1, 3))
    color = mcmc.mcmc_powerlaw.MODEL_REGISTRY["logG"].color
    mcmc._plot_credible_curves(
        axis, [1.0, 2.0, 3.0], curves, "logG", color=color
    )
    assert len(axis.lines) == 1
    assert len(axis.collections) == 1
    assert axis.lines[0].get_color() == color
    plt.close(figure)


def test_logg_draw_plot_highlights_maximum_likelihood_curve():
    """Draw mode should add faint samples plus one emphasized ML curve."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots()
    samples = np.array([[0.1, 0.5, 0.2], [0.4, 0.5, 0.2]])
    grid = np.logspace(0, 1, 5)
    curves = np.array([
        mcmc.dl.log_gaussian_density(sample, grid)
        for sample in samples
    ])
    mcmc._plot_posterior_draws(
        axis, grid, curves, samples[1], 2, np.random.RandomState(1), "logG",
        mcmc.dl.log_gaussian_density,
        color=mcmc.mcmc_powerlaw.MODEL_REGISTRY["logG"].color,
    )
    assert len(axis.lines) == 3
    assert axis.lines[-1].get_linewidth() == pytest.approx(2.5)
    assert axis.lines[-1].get_label() == "logG"
    plt.close(figure)


def test_synthetic_logg_likelihood_recovers_location_and_width():
    """A well-sampled synthetic population should recover its input shape."""
    rng = np.random.RandomState(42)
    true_mu, true_sigma = 0.55, 0.16
    samples = []
    while len(samples) < 250:
        candidates = rng.normal(true_mu, true_sigma, 500)
        samples.extend(candidates[(candidates >= 0) & (candidates <= 1)])
    log_samples = np.asarray(samples[:250])
    log_grid = np.linspace(0, 1, 501)
    grid_weights = mcmc.dl._trapezoid_weights(log_grid)*1000.0
    cache = mcmc.dl.SmoothLikelihoodCache(
        model_coordinate="sma", stack_coordinate="mass",
        model_bounds=(1.0, 10.0), stack_bounds=(1.0, 10.0),
        log_x_samples=log_samples, log_weight_samples=np.zeros(250),
        companion_starts=np.arange(250), companion_counts=np.ones(250, dtype=int),
        companion_weights=np.ones(250),
        companion_names=tuple(str(index) for index in range(250)),
        log_x_grid=log_grid, exposure_weights=grid_weights,
    )
    result = minimize(
        lambda theta: -mcmc.dl.cached_smooth_log_likelihood(
            mcmc.physical_smooth_parameters("logG", theta),
            cache, mcmc.dl.log_gaussian_density,
        ),
        x0=[np.log(0.6), 0.5, np.log(0.2)], method="Nelder-Mead",
    )
    assert result.success
    assert result.x[1] == pytest.approx(true_mu, abs=0.04)
    assert np.exp(result.x[2]) == pytest.approx(true_sigma, abs=0.04)


def test_escarpment_prior_requires_ordered_breakpoints():
    """The two escarpment transitions must remain ordered inside the domain."""
    companions, exposure = _materials()
    cache = mcmc.dl.build_smooth_cache(
        companions, exposure, "m", (1.0, 10.0)
    )
    ordered = [np.log(0.2), np.log(0.4), 0.2, 0.8]
    reversed_breaks = [np.log(0.2), np.log(0.4), 0.8, 0.2]
    assert np.isfinite(mcmc.log_prior_smooth(
        ordered, cache, "escarpment", max_integrated_occurrence=None
    ))
    assert mcmc.log_prior_smooth(
        reversed_breaks, cache, "escarpment",
        max_integrated_occurrence=None,
    ) == -np.inf


@pytest.mark.parametrize("model_name", ["escarpment", "sigmoid", "bpl"])
def test_new_smooth_samplers_save_physical_chains(tmp_path, model_name):
    """Each new direct sampler should save positive physical amplitudes."""
    companions, exposure = _materials()
    cache = mcmc.dl.build_smooth_cache(
        companions, exposure, "m", (1.0, 10.0)
    )
    path = tmp_path / f"{model_name}.npz"
    mcmc.mcmc_smooth(
        cache, model_name, nwalkers=10, nsteps=6, burnin=4,
        save_path=path, random_seed=21, max_integrated_occurrence=None,
    )
    with np.load(path) as data:
        assert data["flat_chains"].shape[1] == 4
        assert np.all(data["flat_chains"][:, 0] > 0)
        if model_name in {"escarpment", "sigmoid"}:
            assert np.all(data["flat_chains"][:, 1] > 0)
        if model_name == "sigmoid":
            assert np.all(data["flat_chains"][:, 3] > 0)
        if model_name == "escarpment":
            assert np.all(data["flat_chains"][:, 2] < data["flat_chains"][:, 3])


@pytest.mark.parametrize(
    "model_name, sample",
    [
        ("escarpment", [0.1, 0.4, 0.25, 0.75]),
        ("sigmoid", [0.1, 0.4, 0.5, 0.1]),
        ("bpl", [0.2, 0.5, -0.4, 1.2]),
    ],
)
def test_new_smooth_models_use_shared_plotting(tmp_path, model_name, sample):
    """New model chains should produce density, CDF, and registered corners."""
    path = tmp_path / f"{model_name}.npz"
    rng = np.random.RandomState(12)
    samples = np.tile(sample, (200, 1))
    samples += rng.normal(0.0, 0.002, size=samples.shape)
    np.savez(
        path, flat_chains=samples, flat_log_probs=np.linspace(-1.0, 0.0, 200),
        model_bounds=[1.0, 10.0], stack_bounds=[1.0, 10.0],
        model_coordinate="sma", stack_coordinate="mass",
    )
    import matplotlib.pyplot as plt
    figures = {"density": plt.subplots(), "cumulative": plt.subplots()}
    paths = mcmc.add_smooth_model_to_figures(
        [path], model_name, figures, tmp_path / "plots", stack_dim="m",
        plot_occurrence=False, plot_cumulative=True, plot_corner=True,
    )
    paths.update(mcmc.save_model_figures(
        figures, tmp_path / "plots", "m", [1.0, 10.0], "test",
    ))
    assert set(paths) == {"density", "cumulative", "corner"}
    assert (tmp_path / "plots" /
            f"corner_{model_name}_bin0.png").exists()


def test_sigmoid_prior_jacobian_uses_both_plateaus_and_width():
    """The sigmoid transform must preserve physical-uniform priors."""
    companions, exposure = _materials()
    cache = mcmc.dl.build_smooth_cache(
        companions, exposure, "m", (1.0, 10.0)
    )
    first = np.array([np.log(0.1), np.log(0.2), 0.5, np.log(0.1)])
    second = np.array([np.log(0.2), np.log(0.4), 0.5, np.log(0.2)])
    difference = mcmc.log_prior_smooth(
        second, cache, "sigmoid", max_integrated_occurrence=None
    ) - mcmc.log_prior_smooth(
        first, cache, "sigmoid", max_integrated_occurrence=None
    )
    assert difference == pytest.approx(3*np.log(2.0))


def test_combined_figures_are_finalized_once_with_user_ticks(tmp_path, monkeypatch):
    """Final layout and saving should occur once after all model overlays."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots()
    axis.plot([0.4, 50.0], [0.0, 1.0], label="model")
    calls = []
    original_tight_layout = figure.tight_layout

    def record_layout(*args, **kwargs):
        calls.append(True)
        return original_tight_layout(*args, **kwargs)

    monkeypatch.setattr(figure, "tight_layout", record_layout)
    edges = [0.4, 0.8, 1.6, 3.2, 6.4, 13.0, 26.0, 50.0]
    paths = mcmc.save_model_figures(
        {"cumulative": (figure, axis)}, tmp_path, "a", edges, "test",
    )
    assert len(calls) == 1
    np.testing.assert_allclose(axis.get_xticks(), edges)
    assert (tmp_path / "occurrence_CDF.png").exists()
