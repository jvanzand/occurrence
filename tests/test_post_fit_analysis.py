import numpy as np
import pytest
from types import SimpleNamespace
import json

from occurrence import post_fit_analysis


def _write_summary(root, tier2, tier3, **overrides):
    directory = root / tier2 / tier3 / "saved_dicts"
    directory.mkdir(parents=True)
    values = {
        "nstars": 123,
        "cell_weights": np.array([40.1, 58.63]),
        "cell_compls": np.array([0.4, 0.5]),
        "cell_compl_single": 0.456,
        "n_abins": 2,
        "n_mbins": 1,
        "a_m_lims_pairs": np.array([
            [[1.0, 10.0], [1.0, 10.0]],
            [[10.0, 100.0], [1.0, 10.0]],
        ]),
    }
    values.update(overrides)
    np.savez(directory / post_fit_analysis.SUMMARY_FILENAME, **values)


def test_make_variables_writes_allstars_statistics(tmp_path):
    tier1 = tmp_path / "mtrue"
    _write_summary(tier1, "allstars", "paper_bounds")

    output = post_fit_analysis.make_variables(
        tmp_path, ["mtrue"], ["allstars"], ["paper_bounds"],
    )

    assert output == tmp_path / "variables.tex"
    assert output.read_text().splitlines()[4:] == [
        r"\newcommand{\McallstarsNstars}{\ensuremath{123}}",
        r"\newcommand{\McallstarsNeff}{\ensuremath{98.7}}",
        r"\newcommand{\McallstarsAvgCompl}{\ensuremath{0.46}}",
        r"\newcommand{\McallstarsNeffBinAZero}{\ensuremath{40.1}}",
        r"\newcommand{\McallstarsAvgComplBinAZero}{\ensuremath{0.40}}",
        r"\newcommand{\McallstarsNeffBinAOne}{\ensuremath{58.6}}",
        r"\newcommand{\McallstarsAvgComplBinAOne}{\ensuremath{0.50}}",
    ]


def test_make_variables_expands_high_and_low_tier2_directories(tmp_path):
    tier1 = tmp_path / "qtrue"
    _write_summary(tier1, "highFeH", "roi", nstars=60)
    _write_summary(
        tier1, "lowFeH", "roi", nstars=63,
        cell_weights=np.array([12.345, 0.0]), cell_compl_single=0.8,
    )

    output = post_fit_analysis.make_variables(
        tmp_path, ["qtrue"], ["FeH"], ["roi"]
    )
    text = output.read_text()

    assert r"\QFeHNstarsHigh" in text
    assert r"\QFeHNeffHigh" in text
    assert r"\QFeHAvgComplLow}{\ensuremath{0.80}}" in text
    assert text.count("%"*72) == 2
    assert "\n\n" + "%"*72 in text


def test_make_variables_aggregates_nonstack_dimension_by_area(tmp_path):
    tier1 = tmp_path / "mtrue"
    _write_summary(
        tier1, "allstars", "roi",
        cell_weights=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_compls=np.array([0.2, 0.4, 0.8, 1.0]),
        n_abins=2, n_mbins=2,
        a_m_lims_pairs=np.array([
            [[1.0, 10.0], [1.0, 10.0]],
            [[10.0, 100.0], [1.0, 10.0]],
            [[1.0, 10.0], [10.0, 1000.0]],
            [[10.0, 100.0], [10.0, 1000.0]],
        ]),
    )

    text = post_fit_analysis.make_variables(
        tmp_path, ["mtrue"], ["allstars"], ["roi"], stack_dim="a"
    ).read_text()

    assert r"\McallstarsNeffBinAZero}{\ensuremath{4.0}}" in text
    assert r"\McallstarsNeffBinAOne}{\ensuremath{6.0}}" in text
    assert r"\McallstarsAvgComplBinAZero}{\ensuremath{0.60}}" in text
    assert r"\McallstarsAvgComplBinAOne}{\ensuremath{0.80}}" in text


def test_make_variables_reports_missing_summary(tmp_path):
    with pytest.raises(FileNotFoundError, match="summary_dict_direct_piecewise"):
        post_fit_analysis.make_variables(
            tmp_path, ["mtrue"], ["allstars"], ["roi"],
        )


def test_parameter_precision_uses_tighter_error():
    assert post_fit_analysis._format_parameter(
        5.23423, 5.18023, 5.35723
    ) == r"5.23^{+0.12}_{-0.05}"


def test_make_variables_collects_piecewise_integrated_occurrence(tmp_path):
    tier1 = tmp_path / "mtrue"
    _write_summary(tier1, "allstars", "roi")
    chain_dir = tier1 / "allstars" / "roi" / "saved_chains"
    chain_dir.mkdir()
    scale = np.linspace(0.5, 1.5, 101)
    np.savez(
        chain_dir / "chains_direct_piecewise.npz",
        flat_chains=np.column_stack([scale, 2.0*scale]),
        x_edges=np.array([1.0, 10.0, 100.0]),
        y_edges=np.array([1.0, 10.0]),
    )

    text = post_fit_analysis.make_variables(
        tmp_path, ["mtrue"], ["allstars"], ["roi"]
    ).read_text()

    assert r"\McallstarsPiecewiseIntOccBinaZero" in text
    assert r"\McallstarsPiecewiseIntOccBinaOne" in text
    assert "% Integrated piecewise occurrence" in text


def test_make_variables_collects_parametric_fit_chains(tmp_path, monkeypatch):
    tier1 = tmp_path / "mtrue"
    _write_summary(tier1, "allstars", "roi")
    chain_dir = tier1 / "allstars" / "roi" / "saved_chains"
    chain_dir.mkdir()
    sample_axis = np.linspace(0.0, 1.0, 101)
    for index in range(2):
        np.savez(
            chain_dir / f"chains_direct_escarpment_bin{index}.npz",
            flat_chains=np.column_stack([
                1.0 + 0.1*sample_axis,
                2.0 + 0.2*sample_axis,
                0.4 + 0.1*sample_axis,
                1.2 + 0.2*sample_axis,
            ]),
            model_bounds=np.array([1.0, 10.0]),
            stack_bounds=np.array([1.0, 10.0]),
        )
    monkeypatch.setattr(
        post_fit_analysis, "calculate_all_delta_bics",
        lambda *args: {
            "mtrue/allstars/roi": {"escarpment": [4.25, -1.04]}
        },
    )

    text = post_fit_analysis.make_variables(
        tmp_path, ["mtrue"], ["allstars"], ["roi"]
    ).read_text()

    assert r"\McallstarsEscarpmentParamBPTwoBinaZero" in text
    assert r"\ensuremath{1.30^{+0.07}_{-0.07}}" in text
    assert r"\McallstarsEscarpmentIntOccBinaZero" in text
    assert "% Parametric model: escarpment" in text
    assert r"\McallstarsEscarpmentDbicBinaZero}{\ensuremath{4.2}}" in text


def test_calculate_delta_bic_uses_flat_minus_model_convention(
        tmp_path, monkeypatch):
    chain_dir = tmp_path / "chains"
    chain_dir.mkdir()
    np.savez(
        chain_dir / "chains_direct_logG_bin0.npz",
        stack_bounds=np.array([1.0, 10.0]),
    )
    cache = SimpleNamespace(companion_names=("one", "two", "three"))
    monkeypatch.setattr(
        post_fit_analysis.dfu, "load_direct_fit_data",
        lambda path: ({}, object()),
    )
    monkeypatch.setattr(
        post_fit_analysis.dl, "build_smooth_cache",
        lambda *args: cache,
    )
    monkeypatch.setattr(
        post_fit_analysis, "_optimize_flat_model", lambda cache: (0.2, -12.0)
    )
    monkeypatch.setattr(
        post_fit_analysis, "_maximum_likelihood_draw",
        lambda path, model: np.array([1.0, 2.0, 3.0]),
    )
    monkeypatch.setattr(
        post_fit_analysis.dl, "cached_smooth_log_likelihood",
        lambda *args: -8.0,
    )

    result = post_fit_analysis.calculate_delta_bic(
        tmp_path / "data.npz", chain_dir
    )

    expected = (np.log(3) + 24.0) - (3*np.log(3) + 16.0)
    np.testing.assert_allclose(result["logG"], [expected])


def test_calculate_all_delta_bics_saves_results_for_every_folder(
        tmp_path, monkeypatch):
    for tier2 in ("highMass", "lowMass"):
        chain_dir = tmp_path / "mtrue" / tier2 / "roi" / "saved_chains"
        chain_dir.mkdir(parents=True)
        (chain_dir / "chains_direct_logG_bin0.npz").touch()
    calls = []

    def fake_calculate(direct_fit_path, chain_dir, stack_dim):
        calls.append((direct_fit_path, chain_dir, stack_dim))
        return {"logG": np.array([3.25])}

    monkeypatch.setattr(
        post_fit_analysis, "calculate_delta_bic", fake_calculate
    )
    result = post_fit_analysis.calculate_all_delta_bics(
        tmp_path, ["mtrue"], ["Mass"], ["roi"], stack_dim="a"
    )

    assert result == {
        "mtrue/highMass/roi": {"logG": [3.25]},
        "mtrue/lowMass/roi": {"logG": [3.25]},
    }
    assert len(calls) == 2
    saved = json.loads(
        (tmp_path / post_fit_analysis.DELTA_BIC_FILENAME).read_text()
    )
    assert saved["stack_dim"] == "a"
    assert saved["delta_bic_convention"] == "BIC_flat - BIC_model"
    assert saved["results"] == result


def test_make_parameter_table_references_variables_commands(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    commands = [
        "McMstarLogGParamAHighBinaZero",
        "McMstarLogGParamMuHighBinaZero",
        "McMstarLogGParamSigmaHighBinaZero",
        "McMstarLogGIntOccHighBinaZero",
        "McMstarEscarpmentParamCOneHighBinaZero",
        "McMstarEscarpmentParamCTwoHighBinaZero",
        "McMstarEscarpmentParamBPOneHighBinaZero",
        "McMstarEscarpmentParamBPTwoHighBinaZero",
        "McMstarEscarpmentIntOccHighBinaZero",
    ]
    (tmp_path / "variables.tex").write_text("\n".join(
        rf"\newcommand{{\{name}}}{{\ensuremath{{1.0}}}}" for name in commands
    ))

    output = post_fit_analysis.make_parameter_table(
        tmp_path, "mtrue", "highMstar", "paper_bounds",
        ["logG", "escaprment"], caption="Custom Fit Caption",
    )
    text = output.read_text()

    assert output == (
        tmp_path / "latex_tables" /
        "model_params_mtrue_highMstar_paper_bounds.tex"
    )
    assert r"\caption{Custom Fit Caption}" in text
    assert r"\label{tab:model_params}" in text
    assert r"$\mu$ & \McMstarLogGParamMuHighBinaZero \\" in text
    assert (
        r"Occurrence & \McMstarEscarpmentIntOccHighBinaZero \\" in text
    )
    assert "1.0" not in text
