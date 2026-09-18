"""Tests for the direct parametric-model registry."""

import numpy as np
import pytest

from occurrence import likelihood as dl
from occurrence import mcmc_powerlaw
from occurrence import plotting_utils


def test_model_registry_is_complete_and_consistent():
    """Every direct parametric model should expose function/display metadata."""
    expected = {
        "logG": (dl.log_gaussian_density, 3),
        "escarpment": (dl.escarpment_density, 4),
        "sigmoid": (dl.sigmoid_density, 4),
        "bpl": (dl.broken_powerlaw_density, 4),
        "loglinear": (dl.log_linear_density, 2),
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


def test_sigmoid_parameter_names_describe_shape():
    assert mcmc_powerlaw.MODEL_REGISTRY["sigmoid"].parameter_names == (
        "C1", "C2", "center", "width",
    )


def test_loglinear_is_red_and_uses_endpoint_rates():
    spec = mcmc_powerlaw.MODEL_REGISTRY["loglinear"]
    assert spec.color == "red"
    assert spec.parameter_names == (r"$C_{\rm low}$", r"$C_{\rm high}$")


def test_corner_explicitly_marks_sigmoid_width_reference(tmp_path, monkeypatch):
    """The final sigmoid parameter should retain its ML vertical marker."""
    import matplotlib.pyplot as plt

    chain_path = tmp_path / "sigmoid.npz"
    np.savez(chain_path, flat_chains=np.ones((20, 4)))
    figure, axes = plt.subplots(4, 4)
    monkeypatch.setattr(
        plotting_utils.corner, "corner", lambda *args, **kwargs: figure
    )
    references = np.array([0.1, 0.2, 0.5, 0.07])
    plotting_utils.plot_corner_from_file(
        chain_path, model_name="sigmoid", outpath=tmp_path / "corner.png",
        reference_values=references,
    )

    width_lines = axes[3, 3].lines
    assert any(np.allclose(line.get_xdata(), references[3]) for line in width_lines)
