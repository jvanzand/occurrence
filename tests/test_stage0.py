"""Compact regression and smoke tests for the pre-refactor codebase."""

import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "stage0_baseline.json"


@pytest.fixture(scope="module")
def baseline():
    """Load fixed numerical outputs recorded before the refactor."""
    with FIXTURE_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def test_science_modules_import():
    """The reusable pipeline's principal modules should all import."""
    modules = [
        "occurrence.main",
        "occurrence.run",
        "occurrence.sampling_utils",
        "occurrence.completeness_utils",
        "occurrence.occurrence_utils",
        "occurrence.mcmc_histogram",
        "occurrence.mcmc_powerlaw",
        "occurrence.plotting_utils",
    ]

    for module_name in modules:
        assert importlib.import_module(module_name) is not None


def test_major_function_signatures():
    """Guard the public call interfaces used by the current driver."""
    from occurrence import main
    from occurrence import mcmc_histogram
    from occurrence import mcmc_powerlaw
    from occurrence import run

    expected = {
        run.run_multiple: (
            "tier1_list", "tier2_list", "tier3_list", "a_edges", "m_edges",
            "recoveries_dir", "recoveries_mtype", "star_df",
            "tier2_df_cuts_dict", "comp_post_dir", "sampling_func",
            "run_mcmc", "run_bic_compare", "run_models_list",
            "plot_models_list", "stack_dim", "m_unit", "avg_map_only",
            "plot_only", "do_single_cells", "nwalkers", "nsteps", "burnin",
            "random_seed",
        ),
        main.prep_occurrence_materials: (
            "tier1_dir", "tier2_dir", "tier3_dir", "a_edges",
            "m_or_q_edges", "stack_dim", "star_df", "compl_type",
            "m_unit", "fig_title",
        ),
        main.run_mcmc: (
            "tier1_dir", "tier2_dir", "tier3_dir", "run_models", "a_edges",
            "m_edges", "stack_dim", "nstars", "parallel", "nwalkers",
            "nsteps", "burnin", "random_seed",
        ),
        mcmc_histogram.mcmc: (
            "nstars", "comp_names_inROI", "cell_dict", "bin_lam_dict",
            "nwalkers", "nsteps", "burnin", "parallel", "save_path",
            "random_seed",
        ),
        mcmc_histogram.loglik_hist: (
            "lam", "nstars", "comp_names", "bin_lam_dict", "num_cells",
            "all_binsizes", "avg_cell_compls",
        ),
        mcmc_powerlaw.mcmc: (
            "hist_dict", "model_func_name", "stack_dim", "stack_ind",
            "nwalkers", "nsteps", "burnin", "parallel", "save_path",
            "random_seed",
        ),
        mcmc_powerlaw.loglik_power: (
            "theta", "hist_dict", "model_func", "model_name",
        ),
    }

    for function, expected_parameters in expected.items():
        assert tuple(inspect.signature(function).parameters) == expected_parameters


@pytest.mark.parametrize(
    ("model_name", "function_name"),
    [
        ("logG", "log_gaussian"),
        ("escarpment", "escarpment"),
        ("bpl", "brokenpowerlaw"),
    ],
)
def test_legacy_model_predictions(baseline, model_name, function_name):
    """Fixed physical parameters should reproduce legacy model predictions."""
    from occurrence import mcmc_powerlaw

    model = getattr(mcmc_powerlaw, function_name)
    model_baseline = baseline["models"][model_name]
    actual = model(model_baseline["theta"], np.asarray(baseline["x"]))

    np.testing.assert_allclose(
        actual,
        model_baseline["expected"],
        rtol=1e-7,
        atol=1e-10,
    )


def _histogram_inputs():
    """Return a minimal two-cell legacy likelihood example."""
    bin_lam_dict = {
        "p1_cell0_compl_over_prior_avg_and_weight": np.array([2.0, 0.75]),
        "p1_cell1_compl_over_prior_avg_and_weight": np.array([1.5, 0.25]),
        "p2_cell0_compl_over_prior_avg_and_weight": np.array([1.2, 0.10]),
        "p2_cell1_compl_over_prior_avg_and_weight": np.array([0.8, 0.90]),
    }
    cell_dict = {
        "num_cells": 2,
        "all_binsizes": np.array([0.5, 0.25]),
        "avg_compls": np.array([0.8, 0.5]),
    }
    return ["p1", "p2"], cell_dict, bin_lam_dict


def test_histogram_likelihood_baseline(baseline):
    """The legacy histogram likelihood should retain its recorded value."""
    from occurrence.mcmc_histogram import loglik_hist

    companions, cell_dict, bin_lam_dict = _histogram_inputs()
    actual = loglik_hist(
        np.array([0.2, 0.4]),
        10,
        companions,
        bin_lam_dict,
        cell_dict["num_cells"],
        cell_dict["all_binsizes"],
        cell_dict["avg_compls"],
    )

    assert actual == pytest.approx(baseline["histogram_loglik"], abs=1e-12)


def test_small_histogram_mcmc_is_reproducible(tmp_path):
    """The public seed argument should reproduce a small histogram chain."""
    from occurrence.mcmc_histogram import mcmc

    companions, cell_dict, bin_lam_dict = _histogram_inputs()

    def run_once(filename):
        sampler = mcmc(
            nstars=10,
            comp_names_inROI=companions,
            cell_dict=cell_dict,
            bin_lam_dict=bin_lam_dict,
            nwalkers=8,
            nsteps=12,
            burnin=8,
            parallel=False,
            save_path=tmp_path / filename,
            random_seed=314159,
        )
        return sampler.get_chain(), sampler.get_log_prob()

    chains_a, log_probs_a = run_once("chains_a.npz")
    chains_b, log_probs_b = run_once("chains_b.npz")

    np.testing.assert_array_equal(chains_a, chains_b)
    np.testing.assert_array_equal(log_probs_a, log_probs_b)
    assert chains_a.shape == (12, 8, 2)
    assert np.isfinite(log_probs_a).all()
