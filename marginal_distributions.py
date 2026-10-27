"""Population distributions for the dimensions a run marginalizes over.

A run fits occurrence along one or two dimensions (semi-major axis ``a``,
companion mass or mass ratio ``m``, eccentricity ``e``) and marginalizes over
the rest. Marginalizing over a dimension means averaging over it with a fixed
population distribution, both in the completeness (the expected number of
detections) and in each detected companion's likelihood term.

DEFAULT: FLAT
-------------
By default a marginalized dimension is flat: uniform per dex in ``a`` and
``m`` and per unit ``e``. That is the same within-range weighting a stacked
bin of a fitted dimension gets, so marginalizing over 0.4-50 M_Jup treats
mass exactly like one stacked mass bin over that range. The literature
distributions in ``POPULATION_MARGINALS`` (Beta in e, broken power law in a,
log-normal in m) are opt-in, through :func:`population_marginal` or a
``distribution`` in the run's ``dimensions`` setting.

INDEPENDENCE ASSUMPTION
-----------------------
Marginalizing with a fixed distribution assumes that, inside the region of
interest, companions follow that distribution in the marginalized dimension
*independently of the fitted dimensions*. For example, fitting the
eccentricity distribution while marginalizing over mass assumes the mass
distribution is the same at every eccentricity. This is the same assumption
the original code made implicitly for eccentricity, whose completeness maps
average over Beta-distributed injections. It is weakest where real
correlations exist (tidal circularization inside ~0.1 AU; possibly higher
eccentricities for more massive giants), so check it by splitting a sample on
the marginalized dimension.

Every density here is normalized to integrate to one over its bounds, with
respect to the dimension's own coordinate: ``log10`` of the value for
logarithmic dimensions (``a`` and ``m``) and the value itself for linear ones
(``e``). That matches the interim priors of the companion posteriors
(log-uniform in ``a`` and ``m``, uniform in ``e``), so a companion's weight is
simply ``density / interim prior``.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import stats

from occurrence import likelihood as dl

# Dimension coordinates: "log" integrates over log10(value), "linear" over
# the value itself.
DIMENSION_SCALES = {"a": "log", "m": "log", "e": "linear"}

_NORMALIZATION_POINTS = 4001


def _flat_density(params, values):
    return np.ones(np.shape(values), dtype=float)


def _beta_density(params, values):
    alpha, beta = params
    return stats.beta.pdf(values, alpha, beta)


def _broken_power_law_density(params, values):
    # Fulton et al. (2021): C a^beta [1 - exp(-(a/a0)^gamma)] per unit log a.
    # The registered bpl model takes (C, log10 a0, beta, gamma).
    amplitude, beta, break_value, gamma = params
    return dl.broken_powerlaw_density(
        [amplitude, np.log10(break_value), beta, gamma], values
    )


def _log_normal_density(params, values):
    # A exp[-(log10 x - mu)^2 / (2 sigma^2)] per unit log10 x.
    return dl.log_gaussian_density(params, values)


# name -> (density function, expected parameter names)
_FORMS = {
    "flat": (_flat_density, ()),
    "beta": (_beta_density, ("alpha", "beta")),
    "bpl": (_broken_power_law_density, ("C", "beta", "a0", "gamma")),
    "lognormal": (_log_normal_density, ("A", "mu", "sigma")),
}


@dataclass(frozen=True)
class MarginalDistribution:
    """One population distribution, normalized over ``bounds``.

    ``form`` names the functional form (see ``_FORMS``); ``params`` are its
    parameters in the order listed there; ``scale`` is ``"log"`` or
    ``"linear"`` and sets the coordinate the density is per unit of.
    """

    form: str
    params: Tuple[float, ...]
    bounds: Tuple[float, float]
    scale: str

    def __post_init__(self):
        if self.form not in _FORMS:
            raise ValueError(
                f"unknown marginal distribution {self.form!r}; choose from "
                f"{sorted(_FORMS)}"
            )
        expected = _FORMS[self.form][1]
        if len(self.params) != len(expected):
            raise ValueError(
                f"{self.form} takes {len(expected)} parameters {expected}, "
                f"got {len(self.params)}"
            )
        if self.scale not in {"log", "linear"}:
            raise ValueError("scale must be 'log' or 'linear'")
        lower, upper = self.bounds
        if not upper > lower or (self.scale == "log" and lower <= 0):
            raise ValueError(f"invalid bounds {self.bounds} for a "
                             f"{self.scale} dimension")
        object.__setattr__(self, "params",
                           tuple(float(value) for value in self.params))
        object.__setattr__(self, "bounds",
                           tuple(float(value) for value in self.bounds))

    def coordinate(self, values):
        """Return the coordinate the density is per unit of."""
        values = np.asarray(values, dtype=float)
        return np.log10(values) if self.scale == "log" else values

    def _raw_density(self, values):
        return np.asarray(_FORMS[self.form][0](self.params, values),
                          dtype=float)

    def _raw_integral(self, lower, upper):
        """Integral of the raw density from ``lower`` to ``upper`` (values).

        The flat form is exact; the Beta form uses its exact CDF, since its
        density diverges at e = 0 when alpha < 1; the others use a fine
        midpoint rule in the dimension's coordinate, which never evaluates
        an endpoint.
        """
        if self.form == "beta":
            return float(stats.beta.cdf(upper, *self.params) -
                         stats.beta.cdf(lower, *self.params))
        start, stop = self.coordinate([lower, upper])
        if self.form == "flat":
            return float(stop - start)
        edges = np.linspace(start, stop, _NORMALIZATION_POINTS)
        centers = 0.5*(edges[1:] + edges[:-1])
        values = 10**centers if self.scale == "log" else centers
        return float(np.sum(self._raw_density(values))*(edges[1] - edges[0]))

    @property
    def normalization(self):
        """Integral of the raw density over the bounds, in its coordinate."""
        return self._raw_integral(*self.bounds)

    def cell_probabilities(self, edges):
        """Return the probability in each cell between consecutive ``edges``.

        Edges outside the bounds are clipped to them, so cells outside get
        zero probability. Independence assumption: see the module docstring.
        """
        edges = np.clip(np.asarray(edges, dtype=float), *self.bounds)
        if np.any(np.diff(edges) < 0):
            raise ValueError("edges must be increasing")
        total = self.normalization
        return np.array([
            self._raw_integral(lower, upper)/total if upper > lower else 0.0
            for lower, upper in zip(edges[:-1], edges[1:])
        ])

    def density(self, values):
        """Return the normalized density at ``values`` (zero outside bounds).

        Independence assumption: see the module docstring.
        """
        values = np.asarray(values, dtype=float)
        inside = (values >= self.bounds[0]) & (values <= self.bounds[1])
        result = np.zeros(values.shape, dtype=float)
        if np.any(inside):
            result[inside] = (
                self._raw_density(values[inside])/self.normalization
            )
        return result

    def record(self):
        """Return a JSON-serializable description."""
        return {"form": self.form, "params": list(self.params),
                "bounds": list(self.bounds), "scale": self.scale}

    @classmethod
    def from_record(cls, record):
        return cls(record["form"], tuple(record["params"]),
                   tuple(record["bounds"]), record["scale"])


# Bounds of the default marginalized ranges. Masses are in Jupiter masses and
# separations in AU.
DEFAULT_BOUNDS = {"e": (0.0, 1.0), "a": (0.1, 30.0), "m": (0.4, 50.0)}

# Defaults used whenever a dimension is marginalized and the run does not
# override them: flat in each dimension's coordinate (see the module
# docstring).
DEFAULT_MARGINALS = {
    name: MarginalDistribution("flat", (), bounds, DIMENSION_SCALES[name])
    for name, bounds in DEFAULT_BOUNDS.items()
}

# Literature population distributions, opt-in through population_marginal:
#   e: Beta(0.867, 3.03), Kipping (2013); also the injections' distribution
#   a: broken power law, C = 350, beta = -0.86, a0 = 3.6 AU, gamma = 1.59,
#      Fulton et al. (2021)
#   m: log-normal in log10 M with A = 0.09, mu = 0.23, sigma = 0.48 dex, the
#      full-sample logG fit of the stellar-properties mass-function paper
POPULATION_MARGINALS = {
    "e": MarginalDistribution("beta", (0.867, 3.03), DEFAULT_BOUNDS["e"],
                              "linear"),
    "a": MarginalDistribution("bpl", (350.0, -0.86, 3.6, 1.59),
                              DEFAULT_BOUNDS["a"], "log"),
    "m": MarginalDistribution("lognormal", (0.09, 0.23, 0.48),
                              DEFAULT_BOUNDS["m"], "log"),
}


def _rebounded(distributions, kind, name, bounds):
    try:
        distribution = distributions[name]
    except KeyError:
        raise ValueError(f"no {kind} marginal distribution for {name!r}")
    if bounds is None:
        return distribution
    return MarginalDistribution(distribution.form, distribution.params,
                                tuple(bounds), distribution.scale)


def default_marginal(name, bounds=None):
    """Return the default (flat) distribution for a dimension.

    ``bounds`` replaces the default range; the density is renormalized over
    it, i.e. the population is truncated to the region of interest.
    """
    return _rebounded(DEFAULT_MARGINALS, "default", name, bounds)


def population_marginal(name, bounds=None):
    """Return a dimension's literature distribution, optionally re-bounded.

    Pass the result as a dimension's ``distribution`` to opt in. Re-bounding
    keeps the functional form and renormalizes over the new bounds.
    """
    return _rebounded(POPULATION_MARGINALS, "population", name, bounds)
