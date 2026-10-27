"""Tests for eccentricity support: dimension roles, marginal distributions,
eccentricity-sliced completeness, eccentricity draws, and eccentricity models.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from occurrence import completeness_utils as cu
from occurrence import dimensions as dm
from occurrence import fit_utils
from occurrence import likelihood as dl
from occurrence import marginal_distributions as md
from occurrence import mcmc
from occurrence import plotting_utils as pu
from occurrence import rvsearch_borrowed as rvsb
from occurrence import sampling_utils as su

ECC_FIT = {
    "e": {"role": "fit", "edges": [0.0, 0.25, 0.5, 1.0]},
    "a": {"role": "marginalize", "bounds": (0.1, 30.0)},
    "m": {"role": "marginalize", "bounds": (0.4, 50.0)},
}


# -- marginal distributions -------------------------------------------------

@pytest.mark.parametrize("name", ["a", "m", "e"])
def test_default_marginals_are_normalized(name):
    distribution = md.DEFAULT_MARGINALS[name]
    low, high = distribution.bounds
    edges = (np.geomspace(low, high, 200) if distribution.scale == "log"
             else np.linspace(low, high, 200))
    assert distribution.cell_probabilities(edges).sum() == pytest.approx(1)


def test_default_marginals_are_flat_in_a_and_m():
    mass = md.default_marginal("m", (1.0, 10.0))
    assert mass.form == "flat"
    # One dex of mass: density 1 per dex everywhere inside
    assert mass.density([1.0, 3.0, 10.0]).tolist() == [1.0, 1.0, 1.0]
    separation = md.DEFAULT_MARGINALS["a"]
    assert separation.density(1.0) == pytest.approx(1/np.log10(300))


def test_default_eccentricity_marginal_is_the_injection_beta():
    # Runs without dimensions average e over the injections' Beta inside
    # each star's map; a marginalized e matches them by default
    assert md.DEFAULT_MARGINALS["e"] == md.INJECTION_ECCENTRICITY
    rebounded = md.default_marginal("e", (0.0, 0.5))
    assert rebounded.form == "beta"
    expected = (stats.beta.cdf(0.1, 0.867, 3.03) /
                stats.beta.cdf(0.5, 0.867, 3.03))
    assert rebounded.cell_probabilities([0.0, 0.1, 0.5])[0] == \
        pytest.approx(expected)


def test_population_marginals_are_opt_in():
    default = dm.parse_dimensions(ECC_FIT)
    assert {default.dimensions[name].distribution.form
            for name in ("a", "m")} == {"flat"}
    lognormal = md.population_marginal("m", (0.4, 50.0))
    opted = dm.parse_dimensions(
        dict(ECC_FIT, m=dict(ECC_FIT["m"], distribution=lognormal.record()))
    )
    assert opted.dimensions["m"].distribution == lognormal
    # Tier 2 products depend on the distribution, so the token differs
    assert opted.token != default.token


def test_beta_marginal_matches_the_injections():
    # The CLS injections put about a third of their draws below e = 0.1
    beta = md.INJECTION_ECCENTRICITY
    assert beta.cell_probabilities([0, 0.1])[0] == pytest.approx(0.333, abs=0.002)
    # The density diverges at 0 (alpha < 1) but cell probabilities are exact
    assert np.isfinite(beta.cell_probabilities([0, 1e-6])).all()


def test_rebounded_marginal_is_renormalized_and_zero_outside():
    mass = md.default_marginal("m", (1.0, 10.0))
    assert mass.cell_probabilities(np.geomspace(1, 10, 50)).sum() == \
        pytest.approx(1)
    assert mass.density([0.5, 20.0]).tolist() == [0.0, 0.0]


def test_marginal_rejects_wrong_parameter_count():
    with pytest.raises(ValueError, match="takes 2 parameters"):
        md.MarginalDistribution("beta", (1.0,), (0, 1), "linear")


# -- dimension specs --------------------------------------------------------

def test_one_dimensional_fit_puts_a_marginalized_dimension_in_y():
    spec = dm.parse_dimensions(ECC_FIT)
    assert (spec.x, spec.y, spec.stack_dim) == ("e", "m", "m")
    assert spec.y_is_marginalized
    # Engine coordinates for e are u = 10**e
    assert spec.engine_edges("x") == pytest.approx(10**np.array([0, .25, .5, 1]))
    assert spec.engine_edges("y") == [0.4, 50.0]
    assert spec.token.startswith("fit-e_marg-a-m_")


def test_spec_round_trips_and_token_ignores_fit_edges():
    spec = dm.parse_dimensions(ECC_FIT)
    assert dm.spec_from_json(spec.to_json()) == spec
    rebinned = dict(ECC_FIT, e={"role": "fit", "edges": [0.0, 0.5, 1.0]})
    assert dm.parse_dimensions(rebinned).token == spec.token
    rebounded = dict(ECC_FIT, a={"role": "marginalize", "bounds": (0.1, 10.0)})
    assert dm.parse_dimensions(rebounded).token != spec.token


@pytest.mark.parametrize("settings, message", [
    ({"a": {"role": "fit", "edges": [0.1, 10]}}, "exactly a, m, and e"),
    (dict(ECC_FIT, a={"role": "fit", "edges": [0.1, 10]},
          m={"role": "fit", "edges": [0.4, 50]}), "one or two"),
    (dict(ECC_FIT, e={"role": "fit", "edges": [0.0, 1.5]}), "outside"),
    (dict(ECC_FIT, m={"role": "marginalize"}), "needs bounds"),
])
def test_invalid_dimension_settings_are_rejected(settings, message):
    with pytest.raises(ValueError, match=message):
        dm.parse_dimensions(settings)


def test_mismatched_and_unrecorded_products_are_refused():
    spec = dm.parse_dimensions(ECC_FIT)
    other = dm.parse_dimensions(dict(
        ECC_FIT, a={"role": "fit", "edges": [0.1, 1, 10]},
        e={"role": "marginalize", "bounds": (0, 1)},
    ))
    dm.check_matching(spec, spec, "same")
    with pytest.raises(ValueError, match="different dimension roles"):
        dm.check_matching(spec, other, "saved")
    with pytest.raises(ValueError, match="no dimension record"):
        dm.check_matching(spec, None, "old product")
    # Legacy runs accept legacy products
    dm.check_matching(dm.legacy_spec([0.1, 10], [0.4, 50]), None, "old")


# -- sliced completeness ----------------------------------------------------

def _synthetic_recoveries(n=3000, seed=0):
    rng = np.random.RandomState(seed)
    a = 10**rng.uniform(-1.5, 2.2, n)
    m = 10**rng.uniform(-1, 4, n)
    e = rng.beta(0.867, 3.03, n)
    detection = 1/(1 + (a/(m/30))**2)*(1 - 0.5*e)
    return pd.DataFrame({"inj_au": a, "inj_mtrue": m, "inj_e": e,
                         "recovered": rng.uniform(size=n) < detection})


def _original_completeness_grid(recoveries, xlim, ylim, resolution=30,
                                xlogwin=0.5, ylogwin=0.5):
    """The original rvsearch moving-average estimator, loop for loop.

    ``rvsb.Completeness.completeness_grid`` now calls the vectorized
    ``cu.moving_average_grid``; this copy is the reference it must match.
    """
    xgrid = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), resolution)
    ygrid = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), resolution)
    xinj, yinj = recoveries["inj_au"], recoveries["inj_mtrue"]
    good = recoveries["recovered"]
    z = np.zeros((len(ygrid), len(xgrid)))
    for i, x in enumerate(xgrid):
        for j, y in enumerate(ygrid):
            xlow = 10**(np.log10(x) - xlogwin/2)
            xhigh = 10**(np.log10(x) + xlogwin/2)
            ylow = 10**(np.log10(y) - ylogwin/2)
            yhigh = 10**(np.log10(y) + ylogwin/2)
            xbox = yinj[np.where((xinj <= xhigh) & (xinj >= xlow))[0]]
            if len(xbox) == 0 or y > max(xbox) or y < min(xbox):
                z[j, i] = np.nan
                continue
            boxall = np.where((xinj <= xhigh) & (xinj >= xlow) &
                              (yinj <= yhigh) & (yinj >= ylow))[0]
            boxgood = np.where((xinj[good] <= xhigh) & (xinj[good] >= xlow) &
                               (yinj[good] <= yhigh) & (yinj[good] >= ylow))[0]
            z[j, i] = (float(len(boxgood))/len(boxall) if len(boxall) > 5
                       else np.nan)
    z = cu.fill_completeness_nans(z, direction="high-mass")
    z = cu.fill_completeness_nans(z, direction="low-mass")
    return xgrid, ygrid, z


@pytest.mark.parametrize("subset", ["all", "sparse"])
def test_moving_average_grid_matches_the_original_estimator(subset):
    recoveries = _synthetic_recoveries()
    if subset == "sparse":
        recoveries = recoveries[recoveries.inj_e > 0.5].reset_index(drop=True)
    lims = ((recoveries.inj_au.min(), recoveries.inj_au.max()),
            (recoveries.inj_mtrue.min(), recoveries.inj_mtrue.max()))
    expected = _original_completeness_grid(recoveries, *lims)
    fast = cu.moving_average_grid(recoveries.inj_au, recoveries.inj_mtrue,
                                  recoveries.recovered, *lims)
    comp = rvsb.Completeness(recoveries.copy(), xcol="inj_au",
                             ycol="inj_mtrue", y_unit="earth")
    via_class = comp.completeness_grid(*lims, fill_nans=True)
    for result in (fast, via_class):
        for expected_part, part in zip(expected, result):
            np.testing.assert_array_equal(expected_part, part)


def test_single_slice_reproduces_the_original_map(tmp_path):
    recoveries = _synthetic_recoveries()
    path = tmp_path / "star_recoveries.csv"
    recoveries.to_csv(path, index=False)
    cu.single_map_maker("star", path, tmp_path / "legacy", 1.0,
                        ycol="inj_mtrue")
    cu.sliced_map_maker(path, tmp_path / "sliced", ycol="inj_mtrue",
                        e_edges=(0.0, 1.0))
    sliced = cu.load_sliced(tmp_path / "sliced")
    np.testing.assert_array_equal(
        sliced.zgrids[0], np.load(tmp_path / "legacy" / "zgrid.npy")
    )


def test_sliced_lookup_uses_each_draws_slice():
    x, y = np.logspace(-1, 1, 5), np.logspace(-1, 1, 6)
    edges = (0.0, 0.3, 1.0)
    zgrids = np.stack([np.full((6, 5), 0.9), np.full((6, 5), 0.4)])
    sliced = cu.SlicedCompleteness(x, y, zgrids, edges)
    values = sliced((np.ones(3), np.ones(3), np.array([0.1, 0.5, 1.0])))
    assert values.tolist() == pytest.approx([0.9, 0.4, 0.4])


@pytest.mark.parametrize("a_distribution", [
    None,  # the flat default
    md.population_marginal("a", (0.1, 30.0)),
])
def test_engine_completeness_integrates_with_population_weights(
        a_distribution):
    x, y = np.logspace(-2, 2.5, 100), np.logspace(-1.5, 2.5, 100)
    edges = np.asarray(cu.ECC_SLICE_EDGES)
    centers = 0.5*(edges[1:] + edges[:-1])
    X, Y = np.meshgrid(x, y)
    shape = 1/(1 + (X/5)**1.5)
    sliced = cu.SlicedCompleteness(
        x, y, np.stack([(1 - e)*shape for e in centers]), edges
    )
    # Fit e; marginalize a and m: Q(e) = (1 - e) * E_a[g(a)]
    settings = dict(ECC_FIT)
    if a_distribution is not None:
        settings["a"] = dict(ECC_FIT["a"], distribution=a_distribution)
    spec = dm.parse_dimensions(settings)
    engine = cu.engine_completeness(spec, sliced)
    a_dist = spec.dimensions["a"].distribution
    log_a = np.linspace(-1, np.log10(30), 20001)
    expected = np.trapz(a_dist.density(10**log_a)/(1 + (10**log_a/5)**1.5),
                        log_a)
    for e in (0.05, 0.45, 0.9):
        slice_center = centers[cu.eccentricity_slice_index(e, edges)]
        assert float(engine((10**e, 2.0))) == pytest.approx(
            # The map's 100-point grid limits the agreement, not the weights
            (1 - slice_center)*expected, rel=1e-3
        )


def test_engine_completeness_refuses_undefined_marginal_ranges():
    x, y = np.logspace(-2, 2.5, 30), np.logspace(-1.5, 2.5, 30)
    edges = np.asarray(cu.ECC_SLICE_EDGES)
    zgrids = np.full((len(edges) - 1, 30, 30), 0.5)
    zgrids[2][:, (x > 1) & (x < 3)] = np.nan
    sliced = cu.SlicedCompleteness(x, y, zgrids, edges)
    with pytest.raises(ValueError, match="undefined"):
        cu.engine_completeness(dm.parse_dimensions(ECC_FIT), sliced)


# -- eccentricity draws -----------------------------------------------------

def test_split_normal_draws_stay_in_range_and_follow_the_widths():
    rng = np.random.RandomState(0)
    draws = su.split_normal_draws(0.1, 0.05, 0.2, 20000, rng=rng)
    assert draws.min() >= 0 and draws.max() < 1
    assert np.median(draws) == pytest.approx(0.11, abs=0.02)


def _write_orvara_file(path, names, values):
    chains = np.asarray(values, dtype=float).T.reshape(-1, 1, len(names))
    with h5py.File(path, "w") as stream:
        dataset = stream.create_dataset("chains", data=chains)
        dataset.attrs["param_names"] = names


def test_post_sampler_adds_eccentricity_without_changing_a_and_m(tmp_path):
    rng = np.random.RandomState(1)
    n = 400
    esino, ecoso = rng.uniform(-0.5, 0.5, n), rng.uniform(-0.5, 0.5, n)
    _write_orvara_file(tmp_path / "HD1.h5",
                       ["msec0", "sau0", "esino0", "ecoso0"],
                       [rng.uniform(1e-3, 2e-3, n), rng.uniform(1, 2, n),
                        esino, ecoso])
    star_df = pd.DataFrame({"star_name": ["1"]})
    np.random.seed(3)
    without = su.post_sampler2(tmp_path, star_df, num_samples=50)
    np.random.seed(3)
    with_e = su.post_sampler2(tmp_path, star_df, num_samples=50,
                              include_ecc=True)
    np.testing.assert_array_equal(without["1_0"][0], with_e["1_0"][0])
    np.testing.assert_array_equal(without["1_0"][1], with_e["1_0"][1])
    a, m, e = with_e["1_0"]
    # e comes from the same draws: check it is one of the chain's e values
    chain_e = esino**2 + ecoso**2
    assert np.isin(np.round(e, 12), np.round(chain_e, 12)).all()


def test_missing_eccentricity_raises_unless_listed(tmp_path):
    _write_orvara_file(tmp_path / "HD2.h5", ["msec0", "sau0"],
                       [[1e-3]*10, [1.0]*10])
    with pytest.raises(KeyError, match="LITERATURE_ECCENTRICITIES"):
        su.post_sampler2(tmp_path, pd.DataFrame({"star_name": ["2"]}),
                         num_samples=5, include_ecc=True)



def _write_hd219134(path, semimajor_axes):
    names, values = [], []
    for index, a in enumerate(semimajor_axes):
        names += [f"msec{index}", f"sau{index}"]
        values += [[1e-4]*10, [a]*10]
    _write_orvara_file(path / "HD219134.h5", names, values)


def test_literature_eccentricities_follow_the_files_semimajor_axes(tmp_path):
    # The file orders companions inner first, like the table
    _write_hd219134(tmp_path, [0.0384, 0.0647, 0.145, 0.235, 2.97])
    samples = su.post_sampler2(tmp_path, pd.DataFrame({"star_name": ["219134"]}),
                               num_samples=4000, include_ecc=True)
    medians = [np.median(samples[f"219134_{i}"][2]) for i in range(5)]
    expected = [su.LITERATURE_ECCENTRICITIES[("HD219134", i)][0]
                for i in range(5)]
    assert np.allclose(medians, expected, rtol=0.15)


def test_literature_eccentricity_for_the_wrong_planet_raises(tmp_path):
    _write_hd219134(tmp_path, [2.97, 0.235, 0.145, 0.0647, 0.0384])
    with pytest.raises(ValueError, match="HD 219134_0"):
        su.post_sampler2(tmp_path, pd.DataFrame({"star_name": ["219134"]}),
                         num_samples=5, include_ecc=True)

# -- fit inputs -------------------------------------------------------------

def test_marginal_factor_weights_draws_and_drops_zero_weight_draws():
    record = fit_utils.prepare_companion_samples(
        "c", x_samples=[2.0, 3.0, 4.0], y_samples=[1.0, 1.0, 1.0],
        completeness=[0.5, 0.5, 0.5], interim_prior=[1.0, 1.0, 1.0],
        x_bounds=(1.0, 10.0), y_bounds=(0.4, 50.0),
        marginal_factor=[2.0, 0.0, 1.0],
    )
    assert record.roi_mask.tolist() == [True, False, True]
    assert record.completeness_over_prior.tolist() == [1.0, 0.0, 0.5]


def test_catalog_ignores_auxiliary_arrays_and_applies_the_factor():
    main_array = np.vstack([[2.0, 3.0], [1.0, 1.0], [0.5, 0.5], [0.5, 0.5],
                            [0.5, 0.5], [0.5, 0.5], [1.0, 1.0]])
    catalog = {
        "c": main_array,
        "c::marginal_factor": np.array([1.0, 0.0]),
        "c::e": np.array([0.1, 0.2]),
        fit_utils.DIMENSIONS_KEY: np.asarray("{}"),
    }
    prepared = fit_utils.prepare_catalog(catalog, (1.0, 10.0), (0.4, 50.0))
    assert list(prepared) == ["c"]
    assert prepared["c"].roi_mask.tolist() == [True, False]


def test_fit_data_records_dimensions(tmp_path):
    spec = dm.parse_dimensions(ECC_FIT)
    record = fit_utils.prepare_companion_samples(
        "c", [2.0], [1.0], [0.5], [1.0], (1.0, 10.0), (0.4, 50.0)
    )
    exposure = fit_utils.build_exposure_grid(
        (1.0, 10.0), (0.4, 50.0), resolution=(3, 3),
        average_completeness=lambda points: np.ones_like(points[0]),
        nstars=2,
    )
    path = tmp_path / "fit_data.npz"
    fit_utils.save_fit_data(path, {"c": record}, exposure, spec.to_json())
    assert fit_utils.load_fit_dimensions(path) == spec
    fit_utils.save_fit_data(path, {"c": record}, exposure)
    assert fit_utils.load_fit_dimensions(path) is None


# -- eccentricity models ----------------------------------------------------

@pytest.mark.parametrize("function, theta", [
    (dl.beta_ecc_density, (0.3, 0.867, 3.03)),
    (dl.rayleigh_ecc_density, (0.3, 0.25)),
    (dl.ecc_mixture_density, (0.3, 0.6, 0.05, 0.3)),
])
@pytest.mark.parametrize("e_range", [(0.0, 1.0), (0.1, 0.6)])
def test_eccentricity_models_integrate_to_their_amplitude(function, theta,
                                                          e_range):
    e = np.linspace(0, 1, 200001)
    bounds = tuple(10**np.asarray(e_range))
    density = function(theta, 10**e, model_bounds=bounds)
    assert np.trapz(density, e) == pytest.approx(0.3, rel=2e-3)
    assert function(theta, [10**0.05], model_bounds=(10**0.1, 10**0.6))[0] \
        in (0.0,) or e_range == (0.0, 1.0)


@pytest.mark.parametrize("model", sorted(mcmc.ECC_MODEL_SETTINGS))
def test_eccentricity_model_priors_are_consistent(model):
    settings = mcmc.ECC_MODEL_SETTINGS[model]
    physical = np.array((0.2,) + settings["start"][1:])
    transformed = physical.copy()
    indices = list(settings["log_indices"])
    transformed[indices] = np.log(physical[indices])
    np.testing.assert_allclose(
        mcmc.physical_smooth_parameters(model, transformed), physical
    )
    # The sampler's Jacobian equals the one recomputed from physical samples
    jacobian = mcmc._physical_log_jacobian(model, physical[None, :])[0]
    assert jacobian == pytest.approx(transformed[indices].sum())


def test_mixture_prior_keeps_the_narrow_component_first():
    cache = type("Cache", (), {"model_bounds": (1.0, 10.0)})()
    ordered = np.array([np.log(0.2), 0.5, np.log(0.05), np.log(0.3)])
    swapped = np.array([np.log(0.2), 0.5, np.log(0.3), np.log(0.05)])
    assert np.isfinite(mcmc.log_prior_smooth(
        ordered, cache, "ecc_mixture", max_integrated_occurrence=None))
    assert mcmc.log_prior_smooth(
        swapped, cache, "ecc_mixture", max_integrated_occurrence=None
    ) == -np.inf


# -- plotting ---------------------------------------------------------------

def test_eccentricity_ticks_show_e_and_styles_are_scoped():
    formatter = pu.eccentricity_tick_formatter(10**np.array([0, 0.25, 1.0]))
    assert [formatter(10**e) for e in (0, 0.25, 1.0)] == ["0", "0.25", "1"]
    assert pu.styled("x_label", "default") == "default"
    with pu.axis_style(x_label="Eccentricity"):
        assert pu.styled("x_label", "default") == "Eccentricity"
    assert pu.styled("x_label", "default") == "default"



def test_cumulative_figure_uses_the_styled_label(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    path = tmp_path / "chains_piecewise.npz"
    np.savez(path, flat_chains=np.full((20, 2), 0.1),
             x_edges=10**np.array([0.0, 0.5, 1.0]), y_edges=[0.4, 50.0])
    with pu.axis_style(x_label="Eccentricity"):
        figure, axes = mcmc.piecewise_cumulative_figure(path, "m")
    assert axes[0].get_xlabel() == "Eccentricity"
    plt.close(figure)

def test_weigh_dimension_samples_builds_engine_catalog(tmp_path):
    from occurrence import main
    x, y = np.logspace(-2, 2.5, 20), np.logspace(-1.5, 2.5, 20)
    edges = np.asarray(cu.ECC_SLICE_EDGES)
    host_grids = np.full((len(edges) - 1, 20, 20), 0.6)
    host_grids[0] = 0.0      # a zero estimate in the lowest slice...
    host_grids[1] = np.nan   # ...and an undefined one in the next
    cu.SlicedCompleteness(x, y, host_grids, edges).save(tmp_path / "star")
    average = cu.SlicedCompleteness(
        x, y, np.full((len(edges) - 1, 20, 20), 0.3), edges
    )
    spec = dm.parse_dimensions(ECC_FIT)
    a = np.array([1.0, 1.0, 1.0, 50.0])     # the last is outside 0.1-30 AU
    m = np.array([2.0, 2.0, 2.0, 2.0])
    e = np.array([0.05, 0.15, 0.5, 0.5])
    star_df = pd.DataFrame({"star_name": ["star"], "comp_list": [["c"]]})
    catalog, fallback = main.weigh_dimension_samples(
        spec, {"c": [a, m, e]}, star_df, tmp_path, average
    )
    rows = catalog["c"]
    np.testing.assert_allclose(rows[0], 10**e)          # x slot: u = 10**e
    np.testing.assert_allclose(rows[1], m)              # y slot: mass
    # Zero and NaN host values fall back to the average
    np.testing.assert_allclose(rows[3], [0.3, 0.3, 0.6, 0.6])
    assert fallback == 2
    factor = catalog["c::marginal_factor"]
    expected = (spec.dimensions["a"].distribution.density(a) *
                spec.dimensions["m"].distribution.density(m))
    np.testing.assert_allclose(factor, expected)
    assert factor[-1] == 0
    assert dm.spec_from_json(str(catalog[fit_utils.DIMENSIONS_KEY])) == spec


@pytest.mark.parametrize("settings", [
    ECC_FIT,  # one fitted dimension: a curve
    {"a": {"role": "fit", "edges": [0.1, 1.0, 30.0]},
     "m": {"role": "fit", "edges": [0.4, 5.0, 50.0]},
     "e": {"role": "marginalize", "bounds": (0.0, 1.0)}},  # two: a map
])
def test_dimension_average_map_images_are_written_once(tmp_path, settings):
    from occurrence import main
    x, y = np.logspace(-2, 2.5, 20), np.logspace(-1.5, 2.5, 20)
    edges = np.asarray(cu.ECC_SLICE_EDGES)
    maps_dir = Path(main.sliced_maps_dir(str(tmp_path), "mtrue"))
    counts = np.arange(len(edges) - 1)*10
    for star in ("s1", "s2"):
        cu.SlicedCompleteness(
            x, y, np.full((len(edges) - 1, 20, 20), 0.5), edges
        ).save(maps_dir / star, extra={"counts": counts})
    spec = dm.parse_dimensions(settings)
    stars = pd.DataFrame({"star_name": ["s1", "s2"]})
    main.make_dimension_average_maps(str(tmp_path), "allstars", spec, stars,
                                     "mtrue")
    written = main.plot_dimension_average_maps(
        str(tmp_path), "allstars", spec, ["s1", "s2"], "mtrue"
    )
    assert [Path(path).name for path in written] == [
        "sliced_average_maps.png", "average_completeness.png"
    ]
    assert all(Path(path).stat().st_size > 0 for path in written)
    # Existing images are kept unless asked to overwrite
    assert main.plot_dimension_average_maps(
        str(tmp_path), "allstars", spec, ["s1", "s2"], "mtrue"
    ) == []


def test_run_multiple_needs_edges_or_dimensions(tmp_path):
    from occurrence import run
    stars = pd.DataFrame({"star_name": ["a"]})
    with pytest.raises(ValueError, match="a_edges and m_edges, or dimensions"):
        run.run_multiple(["mtrue"], ["allstars"], ["t3"], star_df=stars,
                         tier2_df_cuts_dict={}, output_dir=tmp_path)


@pytest.mark.parametrize("plot_option",
                         ["plot_catalog_roi", "plot_roi_occurrence"])
def test_run_multiple_refuses_roi_plots_with_dimensions(tmp_path,
                                                        plot_option):
    from occurrence import run
    stars = pd.DataFrame({"star_name": ["a"]})
    with pytest.raises(ValueError, match=f"{plot_option} need the a-m"):
        run.run_multiple(["mtrue"], ["allstars"], ["t3"], star_df=stars,
                         tier2_df_cuts_dict={}, output_dir=tmp_path,
                         m_unit="jupiter", dimensions=ECC_FIT,
                         **{plot_option: True})


def test_companion_plot_background_weights_slices_by_e(tmp_path):
    from occurrence import post_fit_analysis as pfa
    edges = np.asarray(cu.ECC_SLICE_EDGES)
    x, y = np.logspace(-1, 1.5, 5), np.logspace(-0.5, 1.7, 4)
    values = np.linspace(0.9, 0.2, len(edges) - 1)
    grids = values[:, None, None]*np.ones((len(edges) - 1, 4, 5))
    cu.SlicedCompleteness(x, y, grids, edges).save(tmp_path / "avg_map_eslices")
    spec = dm.parse_dimensions(ECC_FIT)
    xgrid, ygrid, zgrid = pfa._companion_plot_completeness(tmp_path, spec)
    weights = md.default_marginal("e").cell_probabilities(edges)
    assert np.allclose(zgrid, np.sum(weights*values)/weights.sum())
    assert np.array_equal(xgrid, x) and np.array_equal(ygrid, y)
