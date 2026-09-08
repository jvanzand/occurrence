"""Tests for the Stage 1 model registry and parametric MCMC driver."""

import numpy as np
import pytest


def test_model_registry_is_complete_and_consistent():
    """Every supported model should expose callable fitting metadata."""
    from occurrence import mcmc_powerlaw

    expected_names = {
        "flat", "pp1", "pp2", "logG", "step", "escarpment", "bpl"
    }
    assert set(mcmc_powerlaw.MODEL_REGISTRY) == expected_names

    for name, spec in mcmc_powerlaw.MODEL_REGISTRY.items():
        assert callable(spec.function)
        assert callable(spec.initializer)
        assert callable(spec.prior)
        assert spec.ndim == len(spec.parameter_names)
        assert mcmc_powerlaw.model_dict[name][0] == spec.function.__name__


def test_unknown_model_has_clear_error():
    """Registry lookup should reject unknown model names explicitly."""
    from occurrence.mcmc_powerlaw import get_model_spec

    with pytest.raises(ValueError, match="Unknown model"):
        get_model_spec("not_a_model")


def test_log_gaussian_peak_location():
    """The log Gaussian should peak at x equal to ten to the mu."""
    from occurrence.mcmc_powerlaw import log_gaussian

    amplitude, mu, sigma = 0.7, 0.4, 0.3
    assert log_gaussian((amplitude, mu, sigma), 10**mu) == pytest.approx(amplitude)


def test_escarpment_segments():
    """The escarpment should have two plateaus and a log-linear transition."""
    from occurrence.mcmc_powerlaw import escarpment

    values = escarpment((0.2, 0.6, 0.0, 2.0), np.array([0.1, 1.0, 10.0, 100.0, 1000.0]))
    np.testing.assert_allclose(values, [0.2, 0.2, 0.4, 0.6, 0.6])


def test_broken_powerlaw_turnover_uses_base_ten():
    """The broken-power-law turnover parameter should be log10(a0)."""
    from occurrence.mcmc_powerlaw import brokenpowerlaw

    C, a0, beta, gamma = 0.5, 10.0, 0.0, 1.0
    actual = brokenpowerlaw((C, np.log10(a0), beta, gamma), a0)
    assert actual == pytest.approx(C*(1 - np.exp(-1)))


@pytest.mark.parametrize(
    ("name", "valid", "invalid"),
    [
        ("logG", (0.2, 0.0, 0.5), (-0.2, 0.0, 0.5)),
        ("escarpment", (0.2, 0.6, -0.5, 0.5), (0.2, 0.6, 0.5, -0.5)),
        ("bpl", (0.5, 0.0, -0.5, 1.0), (-0.5, 0.0, -0.5, 1.0)),
    ],
)
def test_target_model_prior_boundaries(name, valid, invalid):
    """Target-model priors should accept valid and reject invalid parameters."""
    from occurrence.mcmc_powerlaw import log_prior

    assert np.isfinite(log_prior(name, valid, 0.1, 10.0))
    assert log_prior(name, invalid, 0.1, 10.0) == -np.inf


def test_parametric_mcmc_seed_is_reproducible(tmp_path):
    """The parametric fitter should honor its public seed argument."""
    from occurrence.mcmc_powerlaw import mcmc

    hist_dict = {
        "bin_centers": np.array([0.2, 0.7, 2.0, 7.0]),
        "lims": (0.1, 10.0),
        "ORD_vals": np.array([0.08, 0.18, 0.16, 0.06]),
        "ORD_errs_high": np.full(4, 0.05),
        "ORD_errs_low": np.full(4, 0.04),
    }

    def run_once(filename):
        sampler = mcmc(
            hist_dict=hist_dict,
            model_func_name="logG",
            stack_dim="m",
            nwalkers=8,
            nsteps=10,
            burnin=6,
            parallel=False,
            save_path=tmp_path / filename,
            random_seed=271828,
        )
        return sampler.get_chain(), sampler.get_log_prob()

    chains_a, log_probs_a = run_once("logG_a.npz")
    chains_b, log_probs_b = run_once("logG_b.npz")
    np.testing.assert_array_equal(chains_a, chains_b)
    np.testing.assert_array_equal(log_probs_a, log_probs_b)
