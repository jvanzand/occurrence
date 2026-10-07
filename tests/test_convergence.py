"""Tests for chain convergence summaries."""

import numpy as np
import pytest

from occurrence import convergence


def _ar1_chain(n_steps, n_walkers, n_parameters, correlation, seed=0):
    """Return an AR(1) chain whose autocorrelation time grows with correlation."""
    rng = np.random.RandomState(seed)
    chain = np.zeros((n_steps, n_walkers, n_parameters))
    for step in range(1, n_steps):
        chain[step] = (correlation*chain[step - 1] +
                       rng.standard_normal((n_walkers, n_parameters)))
    return chain


def _write_smooth(path, chain, tau=None):
    data = {"transformed_chains": chain, "chains": chain}
    if tau is not None:
        data["autocorrelation_time"] = np.asarray(tau, dtype=float)
    np.savez(path, **data)


def test_well_mixed_chains_converge(tmp_path):
    path = tmp_path / "chains_sigmoid_bin0.npz"
    _write_smooth(path, _ar1_chain(2000, 20, 4, 0.0))

    result = convergence.chain_convergence(path)

    assert result["model"] == "sigmoid"
    assert result["converged"] and result["steps_per_tau"] > 50
    assert [row[0] for row in result["parameters"]] == [
        "ln B1", "ln B2", "log10 x_t", "ln W"]
    assert result["tau_source"] == "computed from the chain"


def test_stored_tau_is_used_and_flags_short_chains(tmp_path):
    path = tmp_path / "chains_logG_bin0.npz"
    _write_smooth(path, _ar1_chain(500, 10, 3, 0.0), tau=[20.0, 25.0, 10.0])

    result = convergence.chain_convergence(path)

    assert result["tau_source"] == "saved by the fit"
    assert not result["converged"]
    assert result["steps_per_tau"] == pytest.approx(500/25)
    assert result["required_steps"] == 50*25


def test_piecewise_chains_use_the_sampled_parameters(tmp_path):
    gp = tmp_path / "gp" / "chains_piecewise.npz"
    gp.parent.mkdir()
    np.savez(gp, chains=np.ones((300, 10, 3)),
             latent_chains=_ar1_chain(300, 10, 3 + 4, 0.0))
    independent = tmp_path / "ind" / "chains_piecewise.npz"
    independent.parent.mkdir()
    np.savez(independent, chains=_ar1_chain(300, 10, 3, 0.0),
             latent_chains=np.empty((0,)))

    gp_names = [row[0] for row in convergence.chain_convergence(gp)["parameters"]]
    ind = convergence.chain_convergence(independent)

    assert gp_names[:4] == ["total occurrence", "ln GP amplitude",
                            "ln length scale (a)", "ln length scale (M)"]
    assert gp_names[4:] == ["bin 0 (latent)", "bin 1 (latent)", "bin 2 (latent)"]
    assert ind["sampled"] == "chains (independent bins)"
    assert [row[0] for row in ind["parameters"]] == [
        "bin 0 height", "bin 1 height", "bin 2 height"]


def test_summary_file_and_warning_name_unconverged_fits(tmp_path):
    chain_dir = tmp_path / "mtrue" / "allstars" / "run" / "saved_chains"
    chain_dir.mkdir(parents=True)
    _write_smooth(chain_dir / "chains_sigmoid_bin0.npz",
                  _ar1_chain(2000, 20, 4, 0.0))
    _write_smooth(chain_dir / "chains_logG_bin0.npz",
                  _ar1_chain(500, 10, 3, 0.0), tau=[30.0, 30.0, 30.0])

    results = convergence.write_convergence_summary(chain_dir)

    text = (chain_dir / "convergence.txt").read_text()
    assert "Convergence summary for mtrue/allstars/run" in text
    assert "Result: 1 of 2 fits converged: logG did not" in text
    assert "NOT converged: spans 17 autocorrelation times" in text
    assert "TOO SHORT" in text and "Verdict: converged" in text
    warning = convergence.convergence_warning(results)
    assert warning.startswith("WARNING: 1 of 2 fits span fewer than 50")
    assert "mtrue/allstars/run (logG, 17 tau; needs >= 1500 steps)" in warning
    assert convergence.convergence_warning(results[:0]) is None


def test_summarize_skips_linked_chain_folders(tmp_path, capsys):
    source = tmp_path / "mtrue" / "allstars" / "source" / "saved_chains"
    source.mkdir(parents=True)
    _write_smooth(source / "chains_sigmoid_bin0.npz", _ar1_chain(2000, 20, 4, 0.0))
    derived = tmp_path / "mtrue" / "allstars" / "derived"
    derived.mkdir()
    (derived / "saved_chains").symlink_to("../source/saved_chains")

    results = convergence.summarize(tmp_path)

    assert len(results) == 1
    assert "All 1 fits span at least 50" in capsys.readouterr().out
