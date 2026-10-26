"""Orbital period as an alternative x coordinate to semi-major axis."""

import numpy as np
import pandas as pd
import pytest

from occurrence import completeness_utils as cu
from occurrence import main
from occurrence import radvel_borrowed as rvb
from occurrence import run
from occurrence import sampling_utils as su


def test_orbital_period_inverts_the_injections_semimajor_axis():
    periods = np.array([3.0, 365.25, 4332.6, 60000.0])
    for mstar in (0.5, 1.0, 1.4):
        a = rvb.semi_major_axis(periods, mstar)
        assert np.allclose(su.orbital_period(a, 0.0, mstar, "jupiter"),
                           periods, rtol=1e-12)


def test_orbital_period_includes_the_companion_mass():
    one_sun = su.orbital_period(1.0, 0.0, 1.0, "solar")
    # Equal masses: total mass 2 M_sun shortens P by sqrt(2)
    assert np.isclose(su.orbital_period(1.0, 1.0, 1.0, "solar"),
                      one_sun/np.sqrt(2))
    assert np.isclose(su.orbital_period(1.0, su.Ms2Mj, 1.0, "jupiter"),
                      one_sun/np.sqrt(2))
    with pytest.raises(ValueError, match="m_unit"):
        su.orbital_period(1.0, 1.0, 1.0, "grams")


def test_periods_from_samples_uses_each_host_and_keeps_masses():
    stars = pd.DataFrame({"star_name": ["s1", "s2"], "Mstar": [1.0, 0.5],
                          "comp_list": [["s1_0"], ["s2_0"]]})
    a, m = np.array([1.0, 2.0]), np.array([1.0, 3.0])
    samples = {"s1_0": np.array([a, m]), "s2_0": np.array([a, m])}
    periods = main.periods_from_samples(samples, stars, "jupiter")
    for name, mstar in (("s1_0", 1.0), ("s2_0", 0.5)):
        np.testing.assert_allclose(
            periods[name][0], su.orbital_period(a, m, mstar, "jupiter"))
        np.testing.assert_array_equal(periods[name][1], m)
    # Mass-ratio Tier 1s sample masses in M_sun (m_unit None)
    solar = main.periods_from_samples(samples, stars, None)
    np.testing.assert_allclose(
        solar["s1_0"][0], su.orbital_period(a, m, 1.0, "solar"))


def test_period_tier1_definitions_and_old_definitions_default_to_a():
    for label in ("mtrue_P", "msini_P", "qtrue_P", "qsini_P"):
        definition = run._tier1_definition(label,
                                           run.DEFAULT_TIER1_DEFINITIONS)
        assert definition["a_or_p"] == "p"
        assert label.startswith(definition["m_or_q"])
    old = {"custom": {"m_or_q": "m", "true_or_sini": "true"}}
    assert run._tier1_definition("custom", old)["a_or_p"] == "a"
    with pytest.raises(ValueError, match="a_or_p"):
        run._tier1_definition("bad", {"bad": {
            "m_or_q": "m", "true_or_sini": "true", "a_or_p": "x"}})


def test_true_mass_recoveries_keep_period_only_when_asked(tmp_path):
    raw = pd.DataFrame({"inj_msini": [100.0, 200.0], "inj_au": [1.0, 2.0],
                        "inj_e": [0.1, 0.2], "recovered": [True, False],
                        "inj_period": [365.0, 1000.0]})
    raw.to_csv(tmp_path / "raw.csv", index=False)
    cu.recs_msini_converter(tmp_path / "raw.csv", tmp_path / "a.csv")
    cu.recs_msini_converter(tmp_path / "raw.csv", tmp_path / "p.csv",
                            keep_period=True)
    without = pd.read_csv(tmp_path / "a.csv")
    with_period = pd.read_csv(tmp_path / "p.csv")
    assert "inj_period" not in without.columns
    # Ten inclination draws per injection; period does not depend on them
    assert len(with_period) == 20
    np.testing.assert_array_equal(with_period.inj_period,
                                  np.repeat([365.0, 1000.0], 10))


def test_period_map_is_built_from_true_mass_recoveries(tmp_path):
    rng = np.random.RandomState(0)
    n = 3000
    period = 10**rng.uniform(1, 4, n)
    mass = 10**rng.uniform(1, 4, n)
    recoveries = pd.DataFrame({
        "inj_mtrue": mass, "inj_au": rvb.semi_major_axis(period, 1.0),
        "inj_e": np.zeros(n), "recovered": mass > 300,
        "inj_period": period,
    })
    path = tmp_path / "star_recoveries.csv"
    recoveries.to_csv(path, index=False)
    cu.single_map_maker("star", path, tmp_path / "map", 1.0,
                        ycol="inj_mtrue", m_unit="earth",
                        xcol="inj_period")
    xgrid = np.load(tmp_path / "map" / "xgrid.npy")
    assert np.isclose(xgrid.min(), period.min())
    assert np.isclose(xgrid.max(), period.max())


def test_explicit_dimensions_refuse_period_tiers(tmp_path):
    stars = pd.DataFrame({"star_name": ["s1"], "Mstar": [1.0]})
    dimensions = {
        "e": {"role": "fit", "edges": [0.0, 1.0]},
        "a": {"role": "marginalize", "bounds": (0.1, 30.0)},
        "m": {"role": "marginalize", "bounds": (0.4, 50.0)},
    }
    with pytest.raises(NotImplementedError, match="period"):
        run.run_multiple(
            ["mtrue_P"], ["allstars"], ["t3"], star_df=stars,
            tier2_df_cuts_dict={"allstars": ({"star_df_query": None}, "")},
            output_dir=tmp_path, dimensions=dimensions, m_unit="jupiter",
            prepare_missing=False,
        )


def test_period_runs_are_titled_and_labeled_by_period():
    assert run._occurrence_function_title("m", "All Stars", None, "p") == (
        "Period Function (All Stars)")
    assert run._occurrence_function_title("a", "", None, "p") == (
        "Mass Function")
    assert run._axis_style(None, "m", "p") == {
        "x_label": "Orbital period [days]"}
    stack_label = run._axis_style(None, "a", "p")["stack_label"]
    assert stack_label(10, 1000) == "$P$=10-1000 days"
    assert run._axis_style(None, "m", "a") == {}


def test_long_period_tick_labels_rotate_and_short_ones_do_not():
    from occurrence import plotting_utils as pu
    assert pu.sma_tick_formatter(np.geomspace(10.0, 6.0e4, 7)).rotate_labels
    assert not pu.sma_tick_formatter([0.1, 1, 3, 10, 30, 100]).rotate_labels
