"""Tests for the direct parametric-model registry."""

import pytest

from occurrence import direct_likelihood as dl
from occurrence import mcmc_powerlaw


def test_model_registry_is_complete_and_consistent():
    """Every direct parametric model should expose function/display metadata."""
    expected = {
        "logG": (dl.log_gaussian_density, 3),
        "escarpment": (dl.escarpment_density, 4),
        "sigmoid": (dl.sigmoid_density, 4),
        "bpl": (dl.broken_powerlaw_density, 4),
    }
    assert set(mcmc_powerlaw.MODEL_REGISTRY) == set(expected)
    for name, (function, ndim) in expected.items():
        spec = mcmc_powerlaw.MODEL_REGISTRY[name]
        assert spec.function is function
        assert spec.ndim == ndim == len(spec.parameter_names)
        assert isinstance(spec.color, str)


def test_unknown_model_has_clear_error():
    """Registry lookup should reject unknown model names explicitly."""
    with pytest.raises(ValueError, match="unknown parametric model"):
        mcmc_powerlaw.get_model_spec("not_a_model")
