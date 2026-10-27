"""Which companion dimensions a run fits, and which it marginalizes over.

The occurrence engine (``fit_utils``, ``likelihood``, ``mcmc``) works in two
slots, ``x`` and ``y``, on logarithmic axes. Originally ``x`` was semi-major
axis and ``y`` companion mass, with eccentricity marginalized implicitly by
the completeness maps. A :class:`DimensionSpec` generalizes that:

* Each of the three dimensions, ``a`` (semi-major axis, AU), ``m``
  (companion mass in M_Jup, true or M sin i per the Tier 1 folder), and ``e``
  (eccentricity), is either **fit** (with bin edges) or **marginalized**
  (with bounds and a population distribution, flat by default; see
  ``marginal_distributions``).
* One or two dimensions are fit. They occupy the engine slots: ``x`` is the
  first fitted dimension (in the order a, m, e) and ``y`` the second. With one
  fitted dimension, ``y`` holds one marginalized dimension as a single bin
  (the engine then stacks over ``y`` and fits along ``x``).
* Dimensions on a linear axis (``e``) enter the engine as ``u = 10**value``,
  so the engine's ``log10`` axes are linear in the value: a bin uniform in e is
  uniform in log10(u), and the posteriors' uniform-in-e prior is uniform in
  log10(u). :meth:`DimensionSpec.to_engine` and :meth:`from_engine` convert.

Marginalized dimensions enter through the completeness (averaged over their
population distribution) and through each companion draw's weight
(distribution / interim prior). That assumes the marginalized dimensions are
distributed independently of the fitted ones; see ``marginal_distributions``.

Runs without a ``dimensions`` setting use :func:`legacy_spec`, which keeps the
original behavior and file layout exactly (``a`` and ``m`` fit, ``e``
marginalized implicitly over the injections' Beta distribution).
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from occurrence import marginal_distributions as md

DIMENSION_ORDER = ("a", "m", "e")
DIMENSION_SCALES = md.DIMENSION_SCALES
ROLES = ("fit", "marginalize")
# Bounds every dimension must stay inside
_LIMITS = {"a": (0.0, np.inf), "m": (0.0, np.inf), "e": (0.0, 1.0)}
# Preference for the marginalized dimension placed in the y slot of a 1D fit
_Y_SLOT_PREFERENCE = ("m", "a", "e")


@dataclass(frozen=True)
class Dimension:
    """One dimension's role, range, and (if marginalized) distribution."""

    name: str
    role: str
    edges: Tuple[float, ...]
    distribution: Optional[md.MarginalDistribution] = None

    @property
    def scale(self):
        return DIMENSION_SCALES[self.name]

    @property
    def bounds(self):
        return (self.edges[0], self.edges[-1])

    def record(self):
        record = {"role": self.role, "scale": self.scale,
                  "edges": list(self.edges)}
        if self.distribution is not None:
            record["distribution"] = self.distribution.record()
        return record


@dataclass(frozen=True)
class DimensionSpec:
    """The roles of ``a``, ``m``, and ``e`` for one run."""

    dimensions: Dict[str, Dimension] = field(default_factory=dict)
    x: str = "a"
    y: str = "m"
    legacy: bool = False

    # -- structure ----------------------------------------------------------
    @property
    def fitted(self):
        return tuple(name for name in DIMENSION_ORDER
                     if self.dimensions[name].role == "fit")

    @property
    def marginalized(self):
        """Marginalized dimensions, including one placed in the y slot."""
        return tuple(name for name in DIMENSION_ORDER
                     if self.dimensions[name].role == "marginalize")

    @property
    def stack_dim(self):
        """The engine's stack setting: 'a' stacks over x, 'm' over y.

        A 1D fit always stacks over y (one bin) and fits along x. A 2D fit
        returns None, and the run's own ``stack_dim`` decides.
        """
        return "m" if len(self.fitted) == 1 else None

    @property
    def y_is_marginalized(self):
        return self.dimensions[self.y].role == "marginalize"

    # -- coordinates --------------------------------------------------------
    def to_engine(self, name, values):
        """Convert physical values of a dimension to engine coordinates."""
        values = np.asarray(values, dtype=float)
        return 10**values if DIMENSION_SCALES[name] == "linear" else values

    def from_engine(self, name, values):
        values = np.asarray(values, dtype=float)
        return np.log10(values) if DIMENSION_SCALES[name] == "linear" else values

    def engine_edges(self, slot):
        """Bin edges of the x or y slot, in engine coordinates."""
        name = self.x if slot == "x" else self.y
        return [float(value) for value in
                self.to_engine(name, self.dimensions[name].edges)]

    # -- bookkeeping --------------------------------------------------------
    @property
    def token(self):
        """Label for Tier 2 file names, e.g. 'fit-e_marg-a-m_3f9c2a'.

        The hash covers everything Tier 2 products depend on (the slots and
        the marginalized dimensions' bounds and distributions, but not the
        fitted bin edges), so runs that differ there never share files.
        Legacy specs return '' so their file layout is unchanged.
        """
        if self.legacy:
            return ""
        digest = hashlib.sha1(
            json.dumps(self.record(include_fit_edges=False),
                       sort_keys=True).encode()
        ).hexdigest()[:6]
        return ("fit-" + "-".join(self.fitted) + "_marg-" +
                "-".join(self.marginalized) + "_" + digest)

    def record(self, include_fit_edges=True):
        """Return a JSON-serializable description of the roles.

        ``include_fit_edges=False`` drops the fitted dimensions' bin edges,
        which affect Tier 3 fits but not the Tier 2 maps and samples.
        """
        dimensions = {}
        for name, dimension in self.dimensions.items():
            entry = dimension.record()
            if dimension.role == "fit" and not include_fit_edges:
                entry.pop("edges")
            dimensions[name] = entry
        return {"legacy": self.legacy, "x": self.x, "y": self.y,
                "dimensions": dimensions}

    def to_json(self):
        return json.dumps(self.record(), sort_keys=True)

    def describe(self):
        """One-line human-readable summary."""
        if self.legacy:
            return ("legacy: a and m fit, e marginalized implicitly over the "
                    "injections' Beta distribution")
        parts = []
        for name in DIMENSION_ORDER:
            dimension = self.dimensions[name]
            low, high = dimension.bounds
            if dimension.role == "fit":
                parts.append(f"{name} fit ({len(dimension.edges) - 1} bins, "
                             f"{low:g}-{high:g})")
            else:
                parts.append(f"{name} marginalized ({dimension.distribution.form},"
                             f" {low:g}-{high:g})")
        return "; ".join(parts) + f" [engine x={self.x}, y={self.y}]"


def spec_from_json(text):
    """Rebuild a spec from :meth:`DimensionSpec.to_json` output."""
    record = json.loads(str(text))
    dimensions = {}
    for name, entry in record["dimensions"].items():
        distribution = entry.get("distribution")
        dimensions[name] = Dimension(
            name, entry["role"], tuple(entry["edges"]),
            md.MarginalDistribution.from_record(distribution)
            if distribution else None,
        )
    return DimensionSpec(dimensions, record["x"], record["y"],
                         bool(record["legacy"]))


def check_matching(expected, found, context, include_fit_edges=True):
    """Raise if a saved product's roles differ from the run's.

    ``found`` is None for products written before roles were recorded;
    those are legacy and match only a legacy run.
    """
    if found is None:
        if expected.legacy:
            return
        raise ValueError(
            f"{context} has no dimension record, so it was made by the legacy "
            f"code (a and m fit, e implicit); this run is: "
            f"{expected.describe()}. Use a different folder or rebuild it."
        )
    if (found.record(include_fit_edges) !=
            expected.record(include_fit_edges)):
        raise ValueError(
            f"{context} was made with different dimension roles than this "
            f"run.\n  run:   {expected.describe()}\n  saved: "
            f"{found.describe()}\nUse a different folder or rebuild it."
        )


def legacy_spec(a_edges, m_edges):
    """The original roles: ``a`` (x) and ``m`` (y) fit, ``e`` implicit."""
    dimensions = {
        "a": Dimension("a", "fit", tuple(float(v) for v in a_edges)),
        "m": Dimension("m", "fit", tuple(float(v) for v in m_edges)),
        "e": Dimension("e", "marginalize", (0.0, 1.0)),
    }
    return DimensionSpec(dimensions, "a", "m", legacy=True)


def parse_dimensions(settings):
    """Build a spec from a run's ``dimensions`` setting.

    ``settings`` maps each of ``a``, ``m``, and ``e`` to either
    ``{"role": "fit", "edges": [...]}`` or
    ``{"role": "marginalize", "bounds": (low, high)}``, optionally with
    ``"distribution": MarginalDistribution`` (or its record) to override the
    default, which is flat in the dimension's coordinate (e.g.
    ``marginal_distributions.population_marginal("m", bounds)`` for the
    log-normal mass function). All three dimensions must be listed.

    Marginalizing assumes the marginalized dimension is distributed
    independently of the fitted ones (see ``marginal_distributions``).
    """
    if set(settings) != set(DIMENSION_ORDER):
        raise ValueError(
            "dimensions must list exactly a, m, and e; got "
            f"{sorted(settings)}"
        )
    dimensions = {}
    for name in DIMENSION_ORDER:
        entry = dict(settings[name])
        role = entry.pop("role", None)
        if role not in ROLES:
            raise ValueError(f"{name}: role must be 'fit' or 'marginalize'")
        if role == "fit":
            if "edges" not in entry:
                raise ValueError(f"{name}: a fitted dimension needs edges")
            edges = tuple(float(value) for value in entry.pop("edges"))
            if len(edges) < 2 or np.any(np.diff(edges) <= 0):
                raise ValueError(f"{name}: edges must be increasing, with at "
                                 "least two values")
            distribution = None
        else:
            if "bounds" not in entry:
                raise ValueError(f"{name}: a marginalized dimension needs bounds")
            edges = tuple(float(value) for value in entry.pop("bounds"))
            if len(edges) != 2 or edges[1] <= edges[0]:
                raise ValueError(f"{name}: bounds must be (low, high)")
            distribution = entry.pop("distribution", None)
            if isinstance(distribution, dict):
                distribution = md.MarginalDistribution.from_record(distribution)
            if distribution is None:
                distribution = md.default_marginal(name, edges)
            if tuple(distribution.bounds) != edges:
                raise ValueError(
                    f"{name}: distribution bounds {distribution.bounds} differ "
                    f"from the dimension bounds {edges}"
                )
        if entry:
            raise ValueError(f"{name}: unknown settings {sorted(entry)}")
        low, high = _LIMITS[name]
        if edges[0] < low or edges[-1] > high or (
                DIMENSION_SCALES[name] == "log" and edges[0] <= 0):
            raise ValueError(f"{name}: range {edges[0]}-{edges[-1]} is outside "
                             "its allowed limits")
        dimensions[name] = Dimension(name, role, edges, distribution)

    fitted = [name for name in DIMENSION_ORDER
              if dimensions[name].role == "fit"]
    if not 1 <= len(fitted) <= 2:
        raise ValueError("fit one or two dimensions")
    if len(fitted) == 2:
        x, y = fitted
    else:
        x = fitted[0]
        y = next(name for name in _Y_SLOT_PREFERENCE
                 if dimensions[name].role == "marginalize")
    return DimensionSpec(dimensions, x, y, legacy=False)
