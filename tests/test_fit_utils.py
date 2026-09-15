"""Unit tests for Stage 2 unbinned direct-fit data preparation."""

import pickle

import numpy as np
import pandas as pd
import pytest

from occurrence import fit_utils as dfu
from occurrence import sampling_utils
from occurrence import completeness_utils


class ConstantInterpolator:
    """Pickleable constant completeness surface for integration tests."""

    def __init__(self, value):
        self.value = value

    def __call__(self, points):
        return np.full(np.broadcast(*points).shape, self.value, dtype=float)


def test_single_completeness_map_plot_is_opt_in(tmp_path, monkeypatch):
    """Tier 1 map creation should save grids without plotting by default."""
    calls = []

    class FakePlots:
        def __init__(self, *args, **kwargs):
            pass

        def save_comp_grids(self, save_dir):
            calls.append("grids")

        def completeness_plot(self, **kwargs):
            calls.append("plot")
            raise AssertionError("default map creation should not plot")

    monkeypatch.setattr(
        completeness_utils.rvsb.Completeness, "from_csv",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(completeness_utils.rvsb, "CompletenessPlots", FakePlots)
    completeness_utils.single_map_maker(
        "star", "recoveries.csv", tmp_path / "map", 1.0
    )
    assert calls == ["grids"]


def test_loguniform_interim_prior_is_constant_in_log_coordinates():
    """Catalog generation should store log-uniform priors in the log measure."""
    catalog = {"b": np.array([[1.0, 10.0], [2.0, 20.0]])}
    result = sampling_utils.interim_prior(catalog)["b"]
    np.testing.assert_array_equal(result[2], [1.0, 1.0])


class PartlyUndefinedInterpolator:
    """Completeness surface with a NaN at a selected posterior draw."""

    def __call__(self, points):
        x_values = np.asarray(points[0])
        return np.where(x_values == 2.0, np.nan, 0.5)


def _constant_interpolator(value):
    def interpolate(points):
        return np.full(np.broadcast(*points).shape, value, dtype=float)
    return interpolate


def test_prepare_companion_samples_preserves_draws_and_marks_roi():
    """Preparation should retain every draw and mark only ROI members."""
    record = dfu.prepare_companion_samples(
        name="planet_b",
        x_samples=[0.5, 1.0, 5.0, 12.0],
        y_samples=[1.0, 2.0, 4.0, 4.0],
        completeness=[0.2, 0.4, 0.8, 0.9],
        interim_prior=[0.5, 0.25, 0.05, 0.02],
        x_bounds=(0.5, 10.0),
        y_bounds=(1.0, 5.0),
    )

    assert record.original_sample_count == 4
    np.testing.assert_array_equal(record.roi_mask, [False, True, True, False])
    np.testing.assert_allclose(record.completeness_over_prior, [0.4, 1.6, 16.0, 45.0])


def test_prepare_companion_samples_clips_machine_precision_overshoot():
    """Interpolation noise at machine precision should remain a valid probability."""
    record = dfu.prepare_companion_samples(
        name="planet_b",
        x_samples=[1.0],
        y_samples=[2.0],
        completeness=[1.0 + np.finfo(float).eps],
        interim_prior=[0.5],
        x_bounds=(0.5, 2.0),
        y_bounds=(1.0, 3.0),
    )
    np.testing.assert_array_equal(record.completeness, [1.0])


@pytest.mark.parametrize(
    "overrides",
    [
        {"interim_prior": [0.5, 0.0]},
        {"completeness": [0.5, 1.2]},
        {"y_samples": [1.0]},
    ],
)
def test_prepare_companion_samples_rejects_invalid_roi_data(overrides):
    """Invalid weights, completeness, or array lengths should fail clearly."""
    arguments = {
        "name": "planet_b",
        "x_samples": [1.0, 2.0],
        "y_samples": [1.0, 2.0],
        "completeness": [0.5, 0.6],
        "interim_prior": [0.5, 0.4],
        "x_bounds": (0.5, 3.0),
        "y_bounds": (0.5, 3.0),
    }
    arguments.update(overrides)
    with pytest.raises(ValueError):
        dfu.prepare_companion_samples(**arguments)


def test_six_row_catalog_requires_explicit_log_measure_prior():
    """Old catalogs must not be silently assigned a coordinate convention."""
    values = np.array([
        [1.0, 2.0],       # x
        [2.0, 4.0],       # stack
        [0.3, 0.4],       # average completeness
        [0.5, 0.8],       # host completeness
        [0.6, 3.2],       # average completeness/prior
        [1.0, 6.4],       # host completeness/prior
    ])
    with pytest.raises(ValueError, match="explicit log-measure interim prior"):
        dfu.prepare_catalog(
            {"planet_b": values},
            x_bounds=(0.5, 3.0),
        y_bounds=(1.0, 5.0),
            completeness_type="single",
        )


def test_catalog_adapter_interprets_stored_prior_in_log_measure():
    """Seven-row products should use their stored log-measure prior literally."""
    values = np.array([
        [1.0, 2.0], [2.0, 4.0], [0.3, 0.4], [0.5, 0.8],
        [0.6, 3.2], [1.0, 6.4], [0.5, 0.125],
    ])
    records = dfu.prepare_catalog(
        {"planet_b": values},
        x_bounds=(0.5, 3.0),
        y_bounds=(1.0, 5.0),
    )
    np.testing.assert_allclose(records["planet_b"].interim_prior, [0.5, 0.125])


def test_catalog_adapter_preserves_stored_log_measure_prior():
    """New constant log-coordinate prior values should require no conversion."""
    values = np.array([
        [1.0, 2.0], [2.0, 4.0], [0.3, 0.4], [0.5, 0.8],
        [0.3, 0.4], [0.5, 0.8], [1.0, 1.0],
    ])
    records = dfu.prepare_catalog(
        {"planet_b": values}, x_bounds=(0.5, 3.0), y_bounds=(1.0, 5.0)
    )
    np.testing.assert_allclose(records["planet_b"].interim_prior, 1.0)

def test_catalog_adapter_omits_empty_companions():
    """Legacy entries with no retained draws should be reported and omitted."""
    with pytest.warns(RuntimeWarning, match="no retained posterior draws"):
        records = dfu.prepare_catalog(
            {"empty_planet": np.empty((7, 0))},
            x_bounds=(0.5, 3.0),
        y_bounds=(1.0, 5.0),
        )
    assert records == {}


def test_catalog_adapter_omits_companions_outside_roi():
    """Catalog preparation should omit wholly unsupported companions."""
    values = np.array([
        [20.0], [2.0], [0.5], [0.5], [10.0], [10.0], [0.05],
    ])
    with pytest.warns(RuntimeWarning, match="no posterior support"):
        records = dfu.prepare_catalog(
            {"outside": values},
            x_bounds=(1.0, 10.0),
            y_bounds=(1.0, 10.0),
        )
    assert records == {}


def test_constant_exposure_integral_matches_analytic_value():
    """Log-grid quadrature should integrate constant exposure exactly."""
    exposure = dfu.build_exposure_grid(
        x_bounds=(1.0, 100.0),
        y_bounds=(0.1, 10.0),
        resolution=(17, 19),
        average_completeness=_constant_interpolator(0.5),
        nstars=4,
    )
    integral = np.sum(exposure.completeness_sum*exposure.integration_weights)
    assert integral == pytest.approx(8.0, abs=1e-12)


def test_individual_and_average_exposure_agree_when_maps_do():
    """Equivalent individual and average maps should produce equal exposure."""
    arguments = {
        "x_bounds": (0.1, 10.0),
        "y_bounds": (1.0, 100.0),
        "resolution": (8, 9),
    }
    individual = dfu.build_exposure_grid(
        **arguments,
        completeness_interpolators=[
            _constant_interpolator(0.2),
            _constant_interpolator(0.3),
        ],
    )
    average = dfu.build_exposure_grid(
        **arguments,
        average_completeness=_constant_interpolator(0.25),
        nstars=2,
    )
    np.testing.assert_allclose(individual.completeness_sum, average.completeness_sum)


def test_fit_data_round_trip(tmp_path):
    """Prepared inputs should survive pickle-free NPZ serialization."""
    companions = {
        "planet_b": dfu.prepare_companion_samples(
            name="planet_b",
            x_samples=[1.0, 2.0],
            y_samples=[2.0, 3.0],
            completeness=[0.5, 0.75],
            interim_prior=[0.5, 0.25],
            x_bounds=(0.5, 3.0),
            y_bounds=(1.0, 4.0),
        )
    }
    exposure = dfu.build_exposure_grid(
        x_bounds=(0.5, 3.0),
        y_bounds=(1.0, 4.0),
        resolution=(4, 5),
        average_completeness=_constant_interpolator(0.5),
        nstars=2,
    )
    path = tmp_path / "fit_data.npz"
    dfu.save_fit_data(path, companions, exposure)
    loaded_companions, loaded_exposure = dfu.load_fit_data(path)

    loaded = loaded_companions["planet_b"]
    np.testing.assert_array_equal(loaded.x_samples, companions["planet_b"].x_samples)
    np.testing.assert_array_equal(loaded.roi_mask, companions["planet_b"].roi_mask)
    np.testing.assert_allclose(loaded_exposure.completeness_sum, exposure.completeness_sum)
    np.testing.assert_allclose(loaded_exposure.integration_weights, exposure.integration_weights)


def test_main_preparation_entry_point_does_not_require_histograms(tmp_path):
    """The high-level adapter should write inputs without histogram products."""
    from occurrence.main import prep_fit_materials

    tier1_dir = tmp_path / "mtrue"
    tier2_dir = "allstars"
    average_dir = tier1_dir / tier2_dir / "avg_map"
    average_dir.mkdir(parents=True)
    with (average_dir / "interp_fn.pkl").open("wb") as stream:
        pickle.dump(ConstantInterpolator(0.5), stream)

    sample_array = np.array([
        [1.0, 2.0], [2.0, 4.0], [0.5, 0.5],
        [0.6, 0.6], [1.0, 4.0], [1.2, 4.8],
        [1.0, 1.0],
    ])
    np.savez(average_dir.parent / "sampled_post_prior_compl.npz", planet_b=sample_array)
    stars = pd.DataFrame({"star_name": ["star_1", "star_2"]})

    output_path = prep_fit_materials(
        tier1_dir=str(tier1_dir),
        tier2_dir=tier2_dir,
        tier3_dir="fit_test",
        x_bounds=(0.5, 3.0),
        y_bounds=(1.0, 5.0),
        star_df=stars,
        integration_resolution=(5, 6),
    )
    companions, exposure = dfu.load_fit_data(output_path)

    assert set(companions) == {"planet_b"}
    assert exposure.completeness_sum.shape == (5, 6)
    np.testing.assert_allclose(exposure.completeness_sum, 1.0)


def test_completeness_attachment_preserves_all_posterior_draws(tmp_path, monkeypatch):
    """NaN completeness must not shorten the original posterior sample arrays."""
    from occurrence.sampling_utils import include_post_completeness

    monkeypatch.chdir(tmp_path)
    average_dir = tmp_path / "mtrue" / "allstars" / "avg_map"
    single_dir = tmp_path / "mtrue" / "saved_maps_mtrue" / "star1"
    average_dir.mkdir(parents=True)
    single_dir.mkdir(parents=True)
    for path in (average_dir / "interp_fn.pkl", single_dir / "interp_fn.pkl"):
        with path.open("wb") as stream:
            pickle.dump(PartlyUndefinedInterpolator(), stream)

    sample_count = 500
    x_samples = np.linspace(1.0, 3.0, sample_count)
    x_samples[100] = 2.0
    y_samples = np.full(sample_count, 4.0)
    prior = 1.0/(x_samples*y_samples)
    sampled_post_dict = {
        "star1_0": np.vstack([x_samples, y_samples, prior])
    }
    stars = pd.DataFrame({
        "star_name": ["star1"],
        "comp_list": [["star1_0"]],
    })

    result = include_post_completeness(
        sampled_post_dict=sampled_post_dict,
        star_df=stars,
        tier1_dir="mtrue",
        tier2_dir="allstars",
    )["star1_0"]

    assert result.shape == (7, sample_count)
    assert np.isnan(result[2, 100])
    assert np.isnan(result[3, 100])
    assert np.isnan(result[4, 100])
    assert np.isnan(result[5, 100])
    np.testing.assert_array_equal(result[6], prior)


def test_undefined_single_completeness_falls_back_to_average(tmp_path, monkeypatch):
    """Finite average completeness should replace only undefined single values."""
    from occurrence.sampling_utils import include_post_completeness
    monkeypatch.chdir(tmp_path)
    average_dir = tmp_path / "mtrue" / "allstars" / "avg_map"
    single_dir = tmp_path / "mtrue" / "saved_maps_mtrue" / "star1"
    average_dir.mkdir(parents=True)
    single_dir.mkdir(parents=True)
    with (average_dir / "interp_fn.pkl").open("wb") as stream:
        pickle.dump(ConstantInterpolator(0.8), stream)
    with (single_dir / "interp_fn.pkl").open("wb") as stream:
        pickle.dump(PartlyUndefinedInterpolator(), stream)

    x_samples = np.array([1.0, 2.0, 3.0])
    y_samples = np.full(3, 4.0)
    prior = np.array([0.5, 0.25, 0.5])
    sampled_post_dict = {
        "star1_0": np.vstack([x_samples, y_samples, prior])
    }
    stars = pd.DataFrame({
        "star_name": ["star1"],
        "comp_list": [["star1_0"]],
    })

    with pytest.warns(RuntimeWarning, match="replaced host-specific completeness"):
        result = include_post_completeness(
            sampled_post_dict=sampled_post_dict,
            star_df=stars,
            tier1_dir="mtrue",
            tier2_dir="allstars",
        )["star1_0"]

    np.testing.assert_allclose(result[2], [0.8, 0.8, 0.8])
    np.testing.assert_allclose(result[3], [0.5, 0.8, 0.5])
    np.testing.assert_allclose(result[5], [1.0, 3.2, 1.0])


def test_single_completeness_fallback_can_be_disabled(tmp_path, monkeypatch):
    """Strict catalog generation should retain undefined single completeness."""
    from occurrence.sampling_utils import include_post_completeness

    monkeypatch.chdir(tmp_path)
    average_dir = tmp_path / "mtrue" / "allstars" / "avg_map"
    single_dir = tmp_path / "mtrue" / "saved_maps_mtrue" / "star1"
    average_dir.mkdir(parents=True)
    single_dir.mkdir(parents=True)
    with (average_dir / "interp_fn.pkl").open("wb") as stream:
        pickle.dump(ConstantInterpolator(0.8), stream)
    with (single_dir / "interp_fn.pkl").open("wb") as stream:
        pickle.dump(PartlyUndefinedInterpolator(), stream)

    samples = {"star1_0": np.array([[2.0], [4.0], [1.0]])}
    stars = pd.DataFrame({
        "star_name": ["star1"],
        "comp_list": [["star1_0"]],
    })
    result = include_post_completeness(
        samples, stars, "mtrue", "allstars",
        fill_single_nan_with_average=False,
    )["star1_0"]

    assert np.isnan(result[3, 0])
    assert np.isnan(result[5, 0])
