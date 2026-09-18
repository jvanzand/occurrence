"""Registry of parametric occurrence-rate-density models."""

from dataclasses import dataclass
from typing import Callable, Tuple

from occurrence import likelihood as dl


@dataclass(frozen=True)
class ModelSpec:
    """Evaluation function and display metadata for one parametric model."""

    function: Callable
    ndim: int
    parameter_names: Tuple[str, ...]
    color: str


MODEL_REGISTRY = {
    "logG": ModelSpec(
        dl.log_gaussian_density,
        3,
        ("A", r"$\mu$", r"$\sigma$"),
        "tomato",
    ),
    "escarpment": ModelSpec(
        dl.escarpment_density,
        4,
        ("C1", "C2", r"$\log_{10}(x_{t,1})$", r"$\log_{10}(x_{t,2})$"),
        "RoyalBlue",
    ),
    "sigmoid": ModelSpec(
        dl.sigmoid_density,
        4,
        ("C1", "C2", "center", "width"),
        "green",
    ),
    "bpl": ModelSpec(
        dl.broken_powerlaw_density,
        4,
        ("C", r"$\log_{10}(x_0)$", r"$\beta$", r"$\gamma$"),
        "deeppink",
    ),
    "loglinear": ModelSpec(
        dl.log_linear_density,
        2,
        (r"$C_{\rm low}$", r"$C_{\rm high}$"),
        "purple",
    ),
}


def get_model_spec(model_name):
    """Return the registered specification for a parametric direct model."""
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"unknown parametric model: {model_name!r}")


def evaluate_density(model_name, theta, x, model_bounds):
    """Evaluate a registered model, supplying bounds when it requires them."""
    function = get_model_spec(model_name).function
    if model_name == "loglinear":
        return function(theta, x, model_bounds=model_bounds)
    return function(theta, x)
