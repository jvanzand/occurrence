"""Tests for multi-configuration direct-fit orchestration."""

import numpy as np
import pandas as pd
import pytest

from occurrence import main
from occurrence import run


def test_format_subsample_title_uses_geq_without_stray_math_delimiter():
    assert (
        run._format_subsample_title("$Age>=5.0 Gyr")
        == r"Age $\geq$ 5.0 Gyr"
    )


@pytest.mark.parametrize(
    "m_unit, solar_mass",
    [("jupiter", run.su.Ms2Mj), ("earth", run.su.Ms2Me)],
)
def test_mass_edges_are_converted_for_mass_ratio_tiers(m_unit, solar_mass):
    edges = np.array([1.0, 10.0])
    converted = run._y_edges_for_tier(edges, "q", m_unit)
    np.testing.assert_allclose(converted, edges/solar_mass)
    np.testing.assert_array_equal(edges, [1.0, 10.0])


def test_mass_edges_are_unchanged_for_mass_tiers():
    np.testing.assert_array_equal(
        run._y_edges_for_tier([1.0, 10.0], "m", "jupiter"),
        [1.0, 10.0],
    )


def test_run_multiple_applies_tier2_cuts_and_plot_controls(
        tmp_path, monkeypatch):
    """Each Tier 2 subset should run and plot with its selected star count."""
    prep_calls = []
    fit_calls = []
    plot_calls = []

    def fake_prep(**kwargs):
        prep_calls.append(kwargs)
        return "materials.npz"

    def fake_fit(**kwargs):
        fit_calls.append(kwargs)
        return object()

    def fake_plot(**kwargs):
        plot_calls.append(kwargs)
        return {"occurrence": "occurrence.png"}

    monkeypatch.setattr(main, "prep_fit_materials", fake_prep)
    monkeypatch.setattr(run.mcmc, "fit_piecewise_file", fake_fit)
    monkeypatch.setattr(main, "plot_piecewise", fake_plot)
    monkeypatch.setattr(run, "_tier1_artifacts_exist", lambda *args: True)
    monkeypatch.setattr(run, "_tier2_artifacts_exist", lambda *args: True)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mtrue").mkdir()
    stars = pd.DataFrame({
        "star_name": ["a", "b", "c"],
        "Mstar": [0.8, 1.1, 1.3],
    })
    cuts = {
        "allstars": [{"star_df_query": None}, "All Stars"],
        "highMstar": [{"star_df_query": "Mstar > 1"}, "High Mass"],
    }

    results = run.run_multiple(
        tier1_list=["mtrue"],
        tier2_list=["allstars", "highMstar"],
        tier3_list=["fit"],
        a_edges=[0.1, 10.0],
        m_edges=[0.4, 1.0, 10.0],
        star_df=stars,
        tier2_df_cuts_dict=cuts,
        run_models_list=["piecewise"],
        plot_models_list=["piecewise"],
        stack_dim="a",
        plot_occurrence=True,
        plot_density=False,
        plot_corner=True,
        plot_catalog_roi=True,
        plot_roi_occurrence=True,
        plot_uncorrected_occurrence_mle=True,
        occurrence_legend_loc="lower left",
    )

    assert [result["nstars"] for result in results] == [3, 2]
    assert [len(call["star_df"]) for call in prep_calls] == [3, 2]
    assert len(fit_calls) == 2
    assert [call["nstars"] for call in plot_calls] == [3, 2]
    assert [call["title"] for call in plot_calls] == [
        "Companion Mass Function (All Stars)",
        "Companion Mass Function (High Mass)",
    ]
    assert all(call["plot_density"] is False for call in plot_calls)
    assert all(call["plot_catalog_roi"] is True for call in plot_calls)
    assert all(call["plot_roi_occurrence"] is True for call in plot_calls)
    assert all(
        call["plot_uncorrected_occurrence_mle"] is True
        for call in plot_calls
    )
    assert all(
        call["occurrence_legend_loc"] == "lower left"
        for call in plot_calls
    )


def test_run_multiple_prepares_missing_tier2(tmp_path, monkeypatch):
    """Missing Tier 2 products should be created before direct fitting."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mtrue").mkdir()
    tier2_calls = []
    monkeypatch.setattr(
        run, "make_tier2",
        lambda configuration, *args: tier2_calls.append(configuration),
    )
    monkeypatch.setattr(
        main, "prep_fit_materials", lambda **kwargs: "materials.npz",
    )
    monkeypatch.setattr(
        run.mcmc, "fit_piecewise_file", lambda **kwargs: object(),
    )
    monkeypatch.setattr(run, "_tier1_artifacts_exist", lambda *args: True)

    results = run.run_multiple(
        tier1_list=["mtrue"],
        tier2_list=["highMstar"],
        tier3_list=["fit"],
        a_edges=[0.1, 10.0],
        m_edges=[0.4, 50.0],
        star_df=pd.DataFrame({"star_name": ["a"], "Mstar": [1.2]}),
        tier2_df_cuts_dict={
            "highMstar": [{"star_df_query": "Mstar > 1"}, "High Mass"]
        },
        comp_post_dir="posteriors",
        sampling_func=lambda *args: {},
        plot_models_list=[],
        make_plots=False,
    )

    assert len(tier2_calls) == 1
    assert tier2_calls[0]["t2_dir"] == "highMstar"
    assert results[0]["nstars"] == 1


def test_run_multiple_prepares_missing_tier1(tmp_path, monkeypatch):
    """Missing Tier 1 products should be generated from supplied recoveries."""
    monkeypatch.chdir(tmp_path)
    tier1_calls = []

    def fake_make_tier1(configuration, *args):
        tier1_calls.append((configuration, args))
        (tmp_path / configuration["t1_dir"]).mkdir()

    monkeypatch.setattr(run, "make_tier1", fake_make_tier1)
    monkeypatch.setattr(run, "_tier2_artifacts_exist", lambda *args: True)
    monkeypatch.setattr(
        main, "prep_fit_materials", lambda **kwargs: "materials.npz"
    )
    monkeypatch.setattr(
        run.mcmc, "fit_piecewise_file", lambda **kwargs: object()
    )
    results = run.run_multiple(
        tier1_list=["mtrue"], tier2_list=["allstars"],
        tier3_list=["fit"], a_edges=[0.1, 10.0],
        m_edges=[0.4, 50.0],
        star_df=pd.DataFrame({"star_name": ["a"], "Mstar": [1.0]}),
        tier2_df_cuts_dict={
            "allstars": [{"star_df_query": None}, "All Stars"]
        },
        recoveries_dir="recoveries", recoveries_mtype="msini",
        plot_models_list=[], make_plots=False,
    )
    assert tier1_calls[0][0]["t1_dir"] == "mtrue"
    assert tier1_calls[0][0]["mass_unit"] == "earth"
    assert tier1_calls[0][1][1:] == ("recoveries", "msini", False, False)
    assert results[0]["tier1"] == "mtrue"


def test_missing_tier1_requires_recoveries_directory(tmp_path, monkeypatch):
    """The runner should identify the input needed for Tier 1 preparation."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="recoveries_dir"):
        run.run_multiple(
            tier1_list=["mtrue"], tier2_list=["allstars"],
            tier3_list=["fit"], a_edges=[0.1, 10.0],
            m_edges=[0.4, 50.0],
            star_df=pd.DataFrame({"star_name": ["a"]}),
            tier2_df_cuts_dict={
                "allstars": [{"star_df_query": None}, "All Stars"]
            },
        )


def test_tier2_readiness_requires_catalog_and_average_map(tmp_path):
    """A directory alone should not count as a complete Tier 2 product."""
    tier2 = tmp_path / "mtrue" / "allstars"
    tier2.mkdir(parents=True)
    assert not run._tier2_artifacts_exist(tmp_path / "mtrue", "allstars")
    (tier2 / "sampled_post_prior_compl.npz").touch()
    assert not run._tier2_artifacts_exist(tmp_path / "mtrue", "allstars")
    (tier2 / "avg_map").mkdir()
    (tier2 / "avg_map" / "interp_fn.pkl").touch()
    assert run._tier2_artifacts_exist(tmp_path / "mtrue", "allstars")


def test_tier1_readiness_requires_every_map_interpolator(tmp_path):
    """A partial Tier 1 directory must be rebuilt rather than silently skipped."""
    configuration = {
        "t1_dir": str(tmp_path / "mtrue"),
        "m_or_q": "m",
        "true_or_sini": "true",
    }
    stars = pd.DataFrame({"star_name": ["a", "b"]})
    map_dir = tmp_path / "mtrue" / "saved_maps_mtrue"
    (map_dir / "a").mkdir(parents=True)
    (map_dir / "a" / "interp_fn.pkl").touch()
    assert not run._tier1_artifacts_exist(configuration, stars)
    (map_dir / "b").mkdir()
    (map_dir / "b" / "interp_fn.pkl").touch()
    assert run._tier1_artifacts_exist(configuration, stars)


def test_parallel_progress_paths_support_tier1_and_tier2_configurations():
    assert run._configuration_path({"t1_dir": "mtrue"}) == "mtrue"
    assert run._configuration_path({
        "t1_dir": "qtrue", "t2_dir": "highMstar",
    }) == "qtrue/highMstar"


def test_parallel_prerequisite_runner_handles_multiple_tier1_and_tier2(
        monkeypatch, capsys):
    calls = []

    class FakeFuture:
        def result(self):
            return None

    class FakeExecutor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def submit(self, function, configuration, *shared_args):
            function(configuration, *shared_args)
            return FakeFuture()

    monkeypatch.setattr(run, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(run, "as_completed", lambda futures: list(futures))

    tier1 = [{"t1_dir": "mtrue"}, {"t1_dir": "qtrue"}]
    tier2 = [
        {"t1_dir": "mtrue", "t2_dir": "allstars"},
        {"t1_dir": "qtrue", "t2_dir": "highMstar"},
    ]
    worker = lambda configuration: calls.append(configuration)

    run._run_futures_in_parallel(worker, tier1, (), "Tier 1")
    run._run_futures_in_parallel(worker, tier2, (), "Tier 2")

    assert calls == tier1 + tier2
    output = capsys.readouterr().out
    assert "Finished Tier 1 mtrue" in output
    assert "Finished Tier 1 qtrue" in output
    assert "Finished Tier 2 mtrue/allstars" in output
    assert "Finished Tier 2 qtrue/highMstar" in output


def test_make_tier2_uses_filtered_stars_for_average_map(monkeypatch):
    """Subset average completeness should use the same selected stellar sample."""
    calls = {}
    monkeypatch.setattr(run, "_tier2_artifacts_exist", lambda *args: False)
    monkeypatch.setattr(
        main, "make_average_map",
        lambda **kwargs: calls.setdefault("map_stars", kwargs["star_df"].copy()),
    )
    monkeypatch.setattr(
        main, "prep_post_draws",
        lambda **kwargs: calls.setdefault("post_stars", kwargs["star_df"].copy()),
    )
    monkeypatch.setattr(run.pu, "plot_catalog", lambda **kwargs: None)
    stars = pd.DataFrame({
        "star_name": ["low", "high"],
        "Mstar": [0.8, 1.2],
        "comp_list": [[], []],
    })
    configuration = {
        "t1_dir": "mtrue",
        "t2_dir": "highMstar",
        "star_df_query": "Mstar > 1",
        "t1_mass_unit": "jupiter",
        "t1_true_or_sini": "true",
        "t1_m_or_q": "m",
    }
    run.make_tier2(configuration, stars, "posteriors", lambda *args: {})
    assert calls["map_stars"]["star_name"].tolist() == ["high"]
    assert calls["post_stars"]["star_name"].tolist() == ["high"]


def test_run_multiple_rejects_unknown_model():
    """Model lists should fail clearly for an unregistered direct model."""
    with pytest.raises(ValueError, match="unsupported models"):
        run.run_multiple(
            tier1_list=["mtrue"],
            tier2_list=["allstars"],
            tier3_list=["fit"],
            a_edges=[0.1, 10.0],
            m_edges=[0.4, 50.0],
            star_df=pd.DataFrame({"star_name": ["a"]}),
            tier2_df_cuts_dict={
                "allstars": [{"star_df_query": None}, "All Stars"]
            },
            run_models_list=["not_a_model"],
        )


def test_run_multiple_dispatches_logg_through_smooth_apis(
        tmp_path, monkeypatch):
    """The public runner should send logG through the shared smooth APIs."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mtrue").mkdir()
    fit_calls, plot_calls = [], []
    monkeypatch.setattr(run, "_tier1_artifacts_exist", lambda *args: True)
    monkeypatch.setattr(run, "_tier2_artifacts_exist", lambda *args: True)
    monkeypatch.setattr(
        main, "prep_fit_materials", lambda **kwargs: "materials.npz"
    )

    def fake_fit(**kwargs):
        fit_calls.append(kwargs)
        return [], ["chains_logG_bin0.npz"]

    monkeypatch.setattr(run.mcmc, "fit_smooth_file", fake_fit)
    monkeypatch.setattr(
        main, "plot_smooth",
        lambda **kwargs: plot_calls.append(kwargs) or {"density": "plot.png"},
    )
    results = run.run_multiple(
        tier1_list=["mtrue"], tier2_list=["allstars"],
        tier3_list=["fit"], a_edges=[0.1, 10.0],
        m_edges=[0.4, 50.0],
        star_df=pd.DataFrame({"star_name": ["a"]}),
        tier2_df_cuts_dict={
            "allstars": [{"star_df_query": None}, "All Stars"]
        },
        run_models_list=["logG"], plot_models_list=["logG"],
        logg_amplitude_bounds=(0.001, 2.0),
        logg_sigma_bounds=(0.05, 1.0),
        plot_cumulative=True,
    )
    assert fit_calls[0]["amplitude_bounds"] == (0.001, 2.0)
    assert fit_calls[0]["width_bounds"] == (0.05, 1.0)
    assert fit_calls[0]["model_name"] == "logG"
    assert plot_calls[0]["plot_corner"] is True
    assert plot_calls[0]["plot_cumulative"] is True
    assert results[0]["chains"]["logG"] == ["chains_logG_bin0.npz"]


def test_run_multiple_passes_model_specific_fit_bounds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mtrue").mkdir()
    calls = []
    monkeypatch.setattr(run, "_tier1_artifacts_exist", lambda *args: True)
    monkeypatch.setattr(run, "_tier2_artifacts_exist", lambda *args: True)
    monkeypatch.setattr(
        main, "prep_fit_materials", lambda **kwargs: "materials.npz"
    )
    monkeypatch.setattr(
        run.mcmc, "fit_smooth_file",
        lambda **kwargs: calls.append(kwargs) or ([], ["loglinear.npz"]),
    )

    run.run_multiple(
        tier1_list=["mtrue"], tier2_list=["allstars"], tier3_list=["fit"],
        a_edges=[0.1, 10.0], m_edges=[0.4, 50.0],
        star_df=pd.DataFrame({"star_name": ["a"]}),
        tier2_df_cuts_dict={
            "allstars": [{"star_df_query": None}, "All Stars"]
        },
        run_models_list=["loglinear"], plot_models_list=[], make_plots=False,
        model_fit_bounds={
            "loglinear": {"a": (0.1, 10.0), "m": (1.3, 13.0)}
        },
    )

    assert calls[0]["model_name"] == "loglinear"
    np.testing.assert_allclose(
        calls[0]["model_fit_bounds"]["m"], [1.3, 13.0]
    )


def test_run_multiple_dispatches_new_smooth_models(tmp_path, monkeypatch):
    """All new models should share the registered smooth-fit dispatcher."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mtrue").mkdir()
    calls = []
    monkeypatch.setattr(run, "_tier1_artifacts_exist", lambda *args: True)
    monkeypatch.setattr(run, "_tier2_artifacts_exist", lambda *args: True)
    monkeypatch.setattr(
        main, "prep_fit_materials", lambda **kwargs: "materials.npz"
    )

    def fake_fit(**kwargs):
        calls.append(kwargs)
        return [], [f"{kwargs['model_name']}.npz"]

    monkeypatch.setattr(run.mcmc, "fit_smooth_file", fake_fit)
    results = run.run_multiple(
        tier1_list=["mtrue"], tier2_list=["allstars"],
        tier3_list=["fit"], a_edges=[0.1, 10.0], m_edges=[0.4, 50.0],
        star_df=pd.DataFrame({"star_name": ["a"]}),
        tier2_df_cuts_dict={
            "allstars": [{"star_df_query": None}, "All Stars"]
        },
        run_models_list=["escarpment", "sigmoid", "bpl"], plot_models_list=[],
        sigmoid_amplitude_bounds=(0.002, 3.0),
        sigmoid_width_bounds=(0.02, 0.5),
        make_plots=False,
    )
    assert [call["model_name"] for call in calls] == [
        "escarpment", "sigmoid", "bpl"
    ]
    assert calls[1]["amplitude_bounds"] == (0.002, 3.0)
    assert calls[1]["width_bounds"] == (0.02, 0.5)
    assert set(results[0]["chains"]) == {"escarpment", "sigmoid", "bpl"}


def test_multiple_smooth_plots_do_not_require_piecewise(tmp_path, monkeypatch):
    """Smooth-model overlays must not load or require a piecewise fit."""
    calls = []
    monkeypatch.setattr(
        main.mcmc, "add_smooth_model_to_figures",
        lambda **kwargs: calls.append(kwargs) or {"density": "plot.png"},
    )
    monkeypatch.setattr(main.glob, "glob", lambda pattern: ["chains_bin0.npz"])
    paths = main.plot_models(
        tier1_dir=str(tmp_path), tier2_dir="allstars", tier3_dir="fit",
        nstars=1, stack_dim="a", a_edges=[0.1, 10.0],
        m_edges=[0.4, 50.0], plot_models=["logG", "escarpment"],
        plot_occurrence=False, plot_cumulative=False, plot_density=False,
        plot_corner=False,
    )
    assert [call["model_name"] for call in calls] == ["logG", "escarpment"]
    assert set(paths) == {"logG", "escarpment", "combined"}


def test_supplementary_plots_overlap_later_serial_models(tmp_path, monkeypatch):
    """Independent plots should run early without using a plotting thread."""
    events = []

    class FakeFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            events.append("collect_roi")
            return self.value

    class FakeProcessExecutor:
        def __init__(self, max_workers, mp_context):
            assert max_workers == 1
            assert mp_context.get_start_method() == "spawn"
            events.append("process_executor")

        def submit(self, function, *args):
            events.append("submit_roi")
            return FakeFuture(function(*args))

        def shutdown(self, wait=True):
            assert wait is True
            events.append("release_executor")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "mtrue").mkdir()
    monkeypatch.setattr(run, "_tier1_artifacts_exist", lambda *args: True)
    monkeypatch.setattr(run, "_tier2_artifacts_exist", lambda *args: True)
    monkeypatch.setattr(
        main, "prep_fit_materials", lambda **kwargs: "materials.npz"
    )
    monkeypatch.setattr(
        run.mcmc, "fit_piecewise_file",
        lambda **kwargs: events.append("fit_piecewise"),
    )
    monkeypatch.setattr(
        run.mcmc, "summarize_piecewise_file",
        lambda *args, **kwargs: {"a_m_lims_pairs": np.ones((1, 2, 2))},
    )
    monkeypatch.setattr(
        run.mcmc, "fit_smooth_file",
        lambda **kwargs: (events.append("fit_logG") or ([], ["logG.npz"])),
    )
    monkeypatch.setattr(
        run.mcmc, "plot_catalog_roi_completeness",
        lambda **kwargs: events.append("plot_catalog") or "catalog.png",
    )
    monkeypatch.setattr(
        run.mcmc, "plot_roi_occurrence_completeness",
        lambda *args: events.append("plot_roi") or "roi.png",
    )
    monkeypatch.setattr(run, "ProcessPoolExecutor", FakeProcessExecutor)

    def fake_final_plot(**kwargs):
        events.append("final_plot")
        assert kwargs["plot_catalog_roi"] is False
        assert kwargs["plot_roi_occurrence"] is False
        return {"piecewise": {}}

    monkeypatch.setattr(main, "plot_models", fake_final_plot)
    results = run.run_multiple(
        tier1_list=["mtrue"], tier2_list=["allstars"], tier3_list=["fit"],
        a_edges=[0.1, 10.0], m_edges=[1.0, 10.0],
        star_df=pd.DataFrame({"star_name": ["a"]}),
        tier2_df_cuts_dict={
            "allstars": [{"star_df_query": None}, "All Stars"]
        },
        run_models_list=["piecewise", "logG"],
        plot_models_list=["piecewise", "logG"],
        plot_catalog_roi=True, plot_roi_occurrence=True,
    )

    assert events.index("plot_catalog") < events.index("fit_piecewise")
    assert events.index("fit_piecewise") < events.index("submit_roi")
    assert events.index("submit_roi") < events.index("fit_logG")
    assert events.index("fit_logG") < events.index("collect_roi")
    assert events.index("collect_roi") < events.index("final_plot")
    assert results[0]["plots"]["piecewise"] == {
        "catalog_roi": "catalog.png", "roi_occurrence": "roi.png",
    }
