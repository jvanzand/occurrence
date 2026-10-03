"""Tests for completeness plotting helpers."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from occurrence import plotting_utils as pu


class _StopAfterCompleteness(Exception):
    pass


def _write_average_map(tier1_dir, tier2_dir):
    average_map = tier1_dir / tier2_dir / "avg_map"
    average_map.mkdir(parents=True)
    for name in ("parent_xgrid.npy", "parent_ygrid.npy", "parent_zgrid.npy"):
        np.save(average_map / name, np.ones((2, 2)))


def _capture_ycol(monkeypatch):
    captured = {}

    def fake_completeness_plotter(*args, **kwargs):
        captured["ycol"] = kwargs["ycol"]
        raise _StopAfterCompleteness

    monkeypatch.setattr(pu, "completeness_plotter", fake_completeness_plotter)
    return captured


@pytest.mark.parametrize("tier1_name", ["mtrue", "mtrue/"])
def test_plot_catalog_infers_ycol_from_tier1_path(
        tmp_path, monkeypatch, tier1_name):
    tier1_dir = tmp_path / "results" / "mtrue"
    _write_average_map(tier1_dir, "allstars")
    captured = _capture_ycol(monkeypatch)

    with pytest.raises(_StopAfterCompleteness):
        pu.plot_catalog(
            tier1_dir=str(tmp_path / "results" / tier1_name),
            tier2_dir="allstars",
            catalog_path="unused.npz",
        )

    assert captured["ycol"] == "inj_mtrue"


def test_plot_catalog_uses_explicit_ycol(tmp_path, monkeypatch):
    tier1_dir = tmp_path / "custom_name"
    _write_average_map(tier1_dir, "allstars")
    captured = _capture_ycol(monkeypatch)

    with pytest.raises(_StopAfterCompleteness):
        pu.plot_catalog(
            tier1_dir=str(tier1_dir),
            tier2_dir="allstars",
            catalog_path="unused.npz",
            ycol="inj_qtrue",
        )

    assert captured["ycol"] == "inj_qtrue"


def test_completeness_plotter_rejects_unknown_ycol():
    xgrid = np.logspace(0, 2, 5)
    ygrid = np.logspace(0, 3, 6)
    zgrid = np.linspace(0, 1, 30).reshape(6, 5)

    with pytest.raises(ValueError, match="unsupported ycol"):
        pu.completeness_plotter(
            xgrid, ygrid, zgrid,
            save_path="", title="", save_plot=False,
            ycol="inj_results/mtrue",
        )
