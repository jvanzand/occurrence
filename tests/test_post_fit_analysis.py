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

    assert output == tmp_path / "paper_items" / "variables.tex"
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

    assert r"\QHighFeHNstars" in text
    assert r"\QHighFeHNeff" in text
    assert r"\QLowFeHAvgCompl}{\ensuremath{0.80}}" in text
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
    with pytest.raises(FileNotFoundError, match="summary_dict_piecewise"):
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
        chain_dir / "chains_piecewise.npz",
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
            chain_dir / f"chains_escarpment_bin{index}.npz",
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
    assert r"\McallstarsEscarpmentParamSlopeBinaZero" in text
    assert r"\ensuremath{1.30^{+0.07}_{-0.07}}" in text
    assert r"\McallstarsEscarpmentIntOccBinaZero" in text
    assert "% Parametric model: escarpment" in text
    assert r"\McallstarsEscarpmentDbicBinaZero}{\ensuremath{4.2}}" in text


def test_make_variables_derives_loglinear_slope_from_fit_bounds(
        tmp_path, monkeypatch):
    tier1 = tmp_path / "mtrue"
    _write_summary(
        tier1, "allstars", "roi", n_abins=1, n_mbins=1,
        cell_weights=np.array([4.0]), cell_compls=np.array([0.5]),
        a_m_lims_pairs=np.array([[[1.0, 10.0], [1.0, 10.0]]]),
    )
    chain_dir = tier1 / "allstars" / "roi" / "saved_chains"
    chain_dir.mkdir()
    offsets = np.linspace(-0.1, 0.1, 101)
    np.savez(
        chain_dir / "chains_loglinear_bin0.npz",
        flat_chains=np.column_stack([1.0 + offsets, 3.0 + 2.0*offsets]),
        model_bounds=np.array([1.0, 100.0]),
        stack_bounds=np.array([1.0, 10.0]),
    )
    monkeypatch.setattr(
        post_fit_analysis, "calculate_all_delta_bics",
        lambda *args: {"mtrue/allstars/roi": {"loglinear": [2.0]}},
    )

    text = post_fit_analysis.make_variables(
        tmp_path, ["mtrue"], ["allstars"], ["roi"]
    ).read_text()

    assert r"\McallstarsLogLinearParamSlopeBinaZero" in text
    # The median slope is (3 - 1) / log10(100 / 1) = 1.
    assert (
        r"\McallstarsLogLinearParamSlopeBinaZero}"
        r"{\ensuremath{1.00^{+0.03}_{-0.03}}}"
    ) in text


def test_calculate_delta_bic_uses_flat_minus_model_convention(
        tmp_path, monkeypatch):
    chain_dir = tmp_path / "chains"
    chain_dir.mkdir()
    np.savez(
        chain_dir / "chains_logG_bin0.npz",
        stack_bounds=np.array([1.0, 10.0]),
        model_bounds=np.array([1.0, 10.0]),
    )
    cache = SimpleNamespace(companion_names=("one", "two", "three"))
    monkeypatch.setattr(
        post_fit_analysis.dfu, "load_fit_data",
        lambda path: ({}, object()),
    )
    monkeypatch.setattr(
        post_fit_analysis.dl, "build_smooth_cache",
        lambda *args, **kwargs: cache,
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
        (chain_dir / "chains_logG_bin0.npz").touch()
    calls = []

    def fake_calculate(fit_path, chain_dir, stack_dim):
        calls.append((fit_path, chain_dir, stack_dim))
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
        (tmp_path / "paper_items" /
         post_fit_analysis.DELTA_BIC_FILENAME).read_text()
    )
    assert saved["stack_dim"] == "a"
    assert saved["delta_bic_convention"] == "BIC_flat - BIC_model"
    assert saved["results"] == result


def test_make_parameter_table_references_variables_commands(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    commands = [
        "McHighMstarLogGParamABinaZero",
        "McHighMstarLogGParamMuBinaZero",
        "McHighMstarLogGParamSigmaBinaZero",
        "McHighMstarLogGIntOccBinaZero",
        "McHighMstarEscarpmentParamCOneBinaZero",
        "McHighMstarEscarpmentParamCTwoBinaZero",
        "McHighMstarEscarpmentParamBPOneBinaZero",
        "McHighMstarEscarpmentParamBPTwoBinaZero",
        "McHighMstarEscarpmentIntOccBinaZero",
        "McHighMstarSigmoidParamCOneBinaZero",
        "McHighMstarSigmoidParamCTwoBinaZero",
        "McHighMstarSigmoidParamCenterBinaZero",
        "McHighMstarSigmoidParamWidthBinaZero",
        "McHighMstarSigmoidIntOccBinaZero",
    ]
    paper_items = tmp_path / "paper_items"
    paper_items.mkdir()
    (paper_items / "variables.tex").write_text("\n".join(
        rf"\newcommand{{\{name}}}{{\ensuremath{{1.0}}}}" for name in commands
    ))

    output = post_fit_analysis.make_parameter_table(
        tmp_path, "mtrue", "highMstar", "paper_bounds",
        ["logG", "escaprment", "sigmoid"], caption="Custom Fit Caption",
    )
    text = output.read_text()
    assert r"$\log_{10}(x_t)$" in text
    assert r"$W$" in text
    assert "Center &" not in text
    assert "Width &" not in text

    assert output == (
        tmp_path / "paper_items" /
        "model_params_mtrue_highMstar_paper_bounds.tex"
    )
    assert r"\caption{Custom Fit Caption}" in text
    assert r"\label{tab:model_params}" in text
    assert r"$\mu$ & \McHighMstarLogGParamMuBinaZero \\" in text
    assert (
        r"Occurrence & \McHighMstarEscarpmentIntOccBinaZero \\" in text
    )
    assert "1.0" not in text


def test_make_appendix_parameter_table_preserves_order_and_uses_commands(
        tmp_path):
    models = ("logG", "escarpment", "sigmoid", "bpl", "loglinear")
    for tier1 in ("mtrue", "qtrue"):
        for tier2 in ("allstars", "highMstar", "lowMstar"):
            chain_dir = tmp_path / tier1 / tier2 / "paper_bounds" / "saved_chains"
            chain_dir.mkdir(parents=True)
            for model in models:
                if (tier1, tier2, model) != ("qtrue", "lowMstar", "loglinear"):
                    (chain_dir / f"chains_{model}_bin0.npz").touch()
    output = post_fit_analysis.make_appendix_parameter_table(
        tmp_path,
        tier1_dirs=["mtrue", "qtrue"],
        tier2_types=["allstars", "Mstar"],
        t3="paper_bounds",
    )
    text = output.read_text()

    assert output == (
        tmp_path / "paper_items" /
        "model_params_appendix_paper_bounds.tex"
    )
    assert r"\begin{deluxetable*}{cccccccccccc}" in text
    assert r"$\theta_1$ & $\theta_2$ & $\theta_3$ & $\theta_4$" in text
    assert "\\startdata\n\n" not in text
    assert "\n\n\\enddata" not in text
    rows = [line for line in text.splitlines() if line.startswith(("mtrue", "qtrue"))]
    assert [row.split(" & ")[:2] for row in rows] == [
        ["mtrue", "allstars"],
        ["mtrue", "highMstar"],
        ["mtrue", "lowMstar"],
        ["qtrue", "allstars"],
        ["qtrue", "highMstar"],
        ["qtrue", "lowMstar"],
    ]
    assert r"\McallstarsNstars" in rows[0]
    assert r"\McallstarsNeffBinAZero" in rows[0]
    assert r"\McallstarsAvgComplBinAZero" in rows[0]
    assert r"\McHighMstarNeffBinAZero" in rows[1]
    data = text.split(r"\startdata", 1)[1].split(r"\enddata", 1)[0]
    all_data_rows = [
        line for line in data.splitlines()
        if line.strip() and line != r"\hline"
    ]
    assert len(all_data_rows) == 23
    first_block = all_data_rows[:4]
    expected_models = [
        rf"\textbf{{{model}}}"
        for model in post_fit_analysis._APPENDIX_MODEL_ORDER
    ]
    assert [row.split(" & ")[5] for row in first_block] == expected_models
    model_rows = dict(zip(post_fit_analysis._APPENDIX_MODEL_ORDER, first_block))
    assert r"\McallstarsLogGIntOccBinaZero" in model_rows["logG"]
    assert r"\McallstarsLogGParamABinaZero" in model_rows["logG"]
    assert r"\McallstarsSigmoidParamCenterBinaZero" in model_rows["sigmoid"]
    assert r"\McallstarsEscarpmentParamBPOneBinaZero" in model_rows["escarpment"]
    assert r"\McallstarsLogLinearParamCHighBinaZero" in model_rows["loglinear"]
    assert r"\nodata" in model_rows["logG"]
    assert r"\McLowMstarLogLinearDbicBinaZero" in all_data_rows[11]
    assert r"\QallstarsLogLinearIntOccBinaZero" in text
    assert r"\QLowMstarLogLinearIntOccBinaZero" not in text
    assert "tablecomments" not in text


def test_make_three_parameter_tables_create_both_forms(tmp_path):
    outputs = post_fit_analysis.make_three_parameter_tables(tmp_path)
    text = outputs["reordered"].read_text()

    assert outputs["reordered"] == (
        tmp_path / "paper_items" /
        "three_parameter_OR_reordered_mtrue_stellar3params.tex"
    )
    assert outputs["original"] == (
        tmp_path / "paper_items" /
        "three_parameter_OR_mtrue_stellar3params.tex"
    )
    assert (
        r"\tablecaption{Occurrence Rates with Two Stellar Parameters Held Fixed}"
        in text
    )
    assert r"\label{tab:three_param_OR_reordered}" in text
    lho = post_fit_analysis._three_parameter_command_name(
        "mtrue", "stellar3params", ("low", "high", "old")
    )
    hho = post_fit_analysis._three_parameter_command_name(
        "mtrue", "stellar3params", ("high", "high", "old")
    )
    hhy = post_fit_analysis._three_parameter_command_name(
        "mtrue", "stellar3params", ("high", "high", "young")
    )
    assert rf"high & old & \{lho} & \{hho} \\" in text
    assert rf"high & high & \{hhy} & \{hho} \\" in text
    assert text.count(rf"\{hhy}") == 3

    original = outputs["original"].read_text()
    assert r"\begin{deluxetable*}{lcccccc}" in original
    assert r"\tablecaption{Occurrence Rates by Stellar Mass, Metallicity, and Age}" in original
    assert original.index(r"\colhead{$N_{\star}$}") < original.index(
        r"\colhead{$N_{\mathrm{eff}}$}"
    ) < original.index(r"\colhead{Completeness}")
    first_row = next(
        line for line in original.splitlines() if line.startswith("high")
    )
    assert first_row.startswith("high & high & young &")
    assert first_row.index("Nstars") < first_row.index("Neff") < first_row.index(
        "AvgCompl"
    )
    for statistic in ("Neff", "Nstars", "AvgCompl", "IntOcc"):
        name = post_fit_analysis._three_parameter_command_name(
            "mtrue", "stellar3params", ("high", "high", "young"),
            statistic,
        )
        assert rf"\{name}" in first_row


def test_make_three_parameter_tables_can_embed_numerical_values(
        tmp_path, monkeypatch):
    values = {
        "Nstars": "47",
        "Neff": "4.0",
        "AvgCompl": "0.64",
        "IntOcc": r"0.12^{+0.08}_{-0.05}",
    }
    monkeypatch.setattr(
        post_fit_analysis, "_three_parameter_statistics",
        lambda result_dir: values,
    )

    outputs = post_fit_analysis.make_three_parameter_tables(
        tmp_path, use_latex_variables=False
    )
    reordered = outputs["reordered"].read_text()
    original = outputs["original"].read_text()

    formatted_occurrence = r"$0.12^{+0.08}_{-0.05}$"
    assert formatted_occurrence in reordered
    assert reordered.count(formatted_occurrence) == 24
    assert (
        "high & high & young & 47 & 4.0 & 0.64 & "
        + formatted_occurrence
    ) in original
    assert r"\McHighMstarHighFeHYoungIntOcc" not in reordered


def test_make_variables_adds_available_three_parameter_results(
        tmp_path, monkeypatch, capsys):
    tier1 = tmp_path / "mtrue"
    _write_summary(tier1, "allstars", "roi")
    subset = "highMstarhighFeHhighAct"
    result_dir = tier1 / subset / "stellar3params"
    summary_dir = result_dir / "saved_dicts"
    summary_dir.mkdir(parents=True)
    (summary_dir / post_fit_analysis.SUMMARY_FILENAME).touch()
    monkeypatch.setattr(
        post_fit_analysis, "_three_parameter_statistics",
        lambda path: {
            "Nstars": "47", "Neff": "4.0", "AvgCompl": "0.64",
            "IntOcc": r"0.12^{+0.08}_{-0.05}",
        },
    )

    text = post_fit_analysis.make_variables(
        tmp_path, ["mtrue"], ["allstars"], ["roi"]
    ).read_text()
    for statistic in ("Nstars", "Neff", "AvgCompl", "IntOcc"):
        name = post_fit_analysis._three_parameter_command_name(
            "mtrue", "stellar3params", ("high", "high", "young"),
            statistic,
        )
        assert rf"\newcommand{{\{name}}}" in text
    output = capsys.readouterr().out
    assert "three-parameter occurrence results have not been calculated" in output
    assert "lowMstarlowFeHlowAct" in output
