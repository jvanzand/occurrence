import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
import json
import re

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

    assert output == tmp_path / "paper_tables" / "variables.tex"
    assert output.read_text().splitlines()[4:] == [
        r"\newcommand{\McAllstarsPaperBoundsNstars}{\ensuremath{123}}",
        r"\newcommand{\McAllstarsPaperBoundsNeff}{\ensuremath{98.7}}",
        r"\newcommand{\McAllstarsPaperBoundsAvgCompl}{\ensuremath{0.46}}",
        r"\newcommand{\McAllstarsPaperBoundsNeffBinAZero}{\ensuremath{40.1}}",
        r"\newcommand{\McAllstarsPaperBoundsAvgComplBinAZero}{\ensuremath{0.40}}",
        r"\newcommand{\McAllstarsPaperBoundsNeffBinAOne}{\ensuremath{58.6}}",
        r"\newcommand{\McAllstarsPaperBoundsAvgComplBinAOne}{\ensuremath{0.50}}",
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

    assert r"\QHighFeHRoiNstars" in text
    assert r"\QHighFeHRoiNeff" in text
    assert r"\QLowFeHRoiAvgCompl}{\ensuremath{0.80}}" in text
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

    assert r"\McAllstarsRoiNeffBinAZero}{\ensuremath{4.0}}" in text
    assert r"\McAllstarsRoiNeffBinAOne}{\ensuremath{6.0}}" in text
    assert r"\McAllstarsRoiAvgComplBinAZero}{\ensuremath{0.60}}" in text
    assert r"\McAllstarsRoiAvgComplBinAOne}{\ensuremath{0.80}}" in text


def test_make_variables_reports_missing_summary(tmp_path):
    (tmp_path / "mtrue" / "allstars" / "roi").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="summary_dict_piecewise"):
        post_fit_analysis.make_variables(
            tmp_path, ["mtrue"], ["allstars"], ["roi"],
        )


def test_make_variables_rejects_tier3_without_any_results(tmp_path):
    _write_summary(tmp_path / "mtrue", "allstars", "roi")
    with pytest.raises(FileNotFoundError, match="missing_run"):
        post_fit_analysis.make_variables(
            tmp_path, ["mtrue"], ["allstars"], ["roi", "missing_run"],
        )


def test_make_variables_combines_tier3_runs_covering_different_samples(
        tmp_path, capsys):
    for tier1 in ("mtrue", "qtrue"):
        for tier2 in ("allstars", "highMstar", "lowMstar"):
            _write_summary(tmp_path / tier1, tier2, "paper_bounds")
        _write_summary(tmp_path / tier1, "allstars", "paper_bounds_noGP")

    text = post_fit_analysis.make_variables(
        tmp_path, ["mtrue", "qtrue"], ["allstars", "Mstar"],
        ["paper_bounds", "paper_bounds_noGP"], three_parameter_t3=None,
    ).read_text()

    for name in ("McAllstarsPaperBoundsNstars", "QLowMstarPaperBoundsNstars",
                 "McAllstarsPaperBoundsNoGPNstars",
                 "QAllstarsPaperBoundsNoGPNstars"):
        assert rf"\newcommand{{\{name}}}" in text
    assert "HighMstarPaperBoundsNoGP" not in text
    output = capsys.readouterr().out
    assert "skipped experiments without result folders" in output
    assert "mtrue/highMstar/paper_bounds_noGP" in output
    assert "qtrue/lowMstar/paper_bounds_noGP" in output


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

    assert r"\McAllstarsRoiPiecewiseIntOccBinaZero" in text
    assert r"\McAllstarsRoiPiecewiseIntOccBinaOne" in text
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

    assert r"\McAllstarsRoiEscarpmentParamBPTwoBinaZero" in text
    assert r"\McAllstarsRoiEscarpmentParamSlopeBinaZero" in text
    assert r"\ensuremath{1.30^{+0.07}_{-0.07}}" in text
    assert r"\McAllstarsRoiEscarpmentIntOccBinaZero" in text
    assert "% Parametric model: escarpment" in text
    assert r"\McAllstarsRoiEscarpmentDbicBinaZero}{\ensuremath{4.2}}" in text


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

    assert r"\McAllstarsRoiLogLinearParamSlopeBinaZero" in text
    # The median slope is (3 - 1) / log10(100 / 1) = 1.
    assert (
        r"\McAllstarsRoiLogLinearParamSlopeBinaZero}"
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
        (tmp_path / "paper_tables" /
         post_fit_analysis.DELTA_BIC_FILENAME).read_text()
    )
    assert saved["stack_dim"] == "a"
    assert saved["delta_bic_convention"] == "BIC_flat - BIC_model"
    assert saved["results"] == result


def test_make_parameter_table_references_variables_commands(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    commands = [
        "McHighMstarPaperBoundsLogGParamABinaZero",
        "McHighMstarPaperBoundsLogGParamMuBinaZero",
        "McHighMstarPaperBoundsLogGParamSigmaBinaZero",
        "McHighMstarPaperBoundsLogGIntOccBinaZero",
        "McHighMstarPaperBoundsEscarpmentParamCOneBinaZero",
        "McHighMstarPaperBoundsEscarpmentParamCTwoBinaZero",
        "McHighMstarPaperBoundsEscarpmentParamBPOneBinaZero",
        "McHighMstarPaperBoundsEscarpmentParamBPTwoBinaZero",
        "McHighMstarPaperBoundsEscarpmentIntOccBinaZero",
        "McHighMstarPaperBoundsSigmoidParamCOneBinaZero",
        "McHighMstarPaperBoundsSigmoidParamCTwoBinaZero",
        "McHighMstarPaperBoundsSigmoidParamCenterBinaZero",
        "McHighMstarPaperBoundsSigmoidParamWidthBinaZero",
        "McHighMstarPaperBoundsSigmoidIntOccBinaZero",
    ]
    paper_tables = tmp_path / "paper_tables"
    paper_tables.mkdir()
    (paper_tables / "variables.tex").write_text("\n".join(
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
        tmp_path / "paper_tables" /
        "model_params_mtrue_highMstar_paper_bounds.tex"
    )
    assert r"\caption{Custom Fit Caption}" in text
    assert r"\label{tab:model_params}" in text
    assert r"$\mu$ & \McHighMstarPaperBoundsLogGParamMuBinaZero \\" in text
    assert (
        r"Occurrence & \McHighMstarPaperBoundsEscarpmentIntOccBinaZero \\" in text
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
        tmp_path / "paper_tables" /
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
    assert r"\McAllstarsPaperBoundsNstars" in rows[0]
    assert r"\McAllstarsPaperBoundsNeffBinAZero" in rows[0]
    assert r"\McAllstarsPaperBoundsAvgComplBinAZero" in rows[0]
    assert r"\McHighMstarPaperBoundsNeffBinAZero" in rows[1]
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
    assert r"\McAllstarsPaperBoundsLogGIntOccBinaZero" in model_rows["logG"]
    assert r"\McAllstarsPaperBoundsLogGParamABinaZero" in model_rows["logG"]
    assert r"\McAllstarsPaperBoundsSigmoidParamCenterBinaZero" in model_rows["sigmoid"]
    assert r"\McAllstarsPaperBoundsEscarpmentParamBPOneBinaZero" in model_rows["escarpment"]
    assert r"\McAllstarsPaperBoundsLogLinearParamCHighBinaZero" in model_rows["loglinear"]
    assert r"\nodata" in model_rows["logG"]
    assert r"\McLowMstarPaperBoundsLogLinearDbicBinaZero" in all_data_rows[11]
    assert r"\QAllstarsPaperBoundsLogLinearIntOccBinaZero" in text
    assert r"\QLowMstarPaperBoundsLogLinearIntOccBinaZero" not in text
    assert "tablecomments" not in text


def test_make_three_parameter_tables_create_both_forms(tmp_path):
    outputs = post_fit_analysis.make_three_parameter_tables(tmp_path)
    text = outputs["reordered"].read_text()

    assert outputs["reordered"] == (
        tmp_path / "paper_tables" /
        "three_parameter_OR_reordered_mtrue_stellar3params.tex"
    )
    assert outputs["original"] == (
        tmp_path / "paper_tables" /
        "three_parameter_OR_mtrue_stellar3params.tex"
    )
    assert (
        r"\tablecaption{Occurrence Rates with Two Stellar Parameters Held Fixed}"
        in text
    )
    assert r"\label{tab:three_param_OR_reordered}" in text
    assert r"\begin{deluxetable*}{cccccc}" in text
    assert r"\multicolumn{2}{c}{Occurrence Rate}" in text
    assert r"\colhead{Significance}" in text
    assert r"\colhead{High/Low}" in text
    assert r"\textbf{Significance}" not in text
    assert r"\textbf{Dynamic Range}" not in text
    assert r"\colhead{Low}" not in text
    assert r"\colhead{High}" not in text
    assert r"\textbf{Low Mass} & \textbf{High Mass}" in text
    assert r"\textbf{Low [Fe/H]} & \textbf{High [Fe/H]}" in text
    assert r"\textbf{Young} & \textbf{Old}" in text
    assert "Mass OR" not in text
    assert "[Fe/H] OR" not in text
    assert "Young OR" not in text
    lho = post_fit_analysis._three_parameter_command_name(
        "mtrue", "stellar3params", ("low", "high", "old")
    )
    hho = post_fit_analysis._three_parameter_command_name(
        "mtrue", "stellar3params", ("high", "high", "old")
    )
    hhy = post_fit_analysis._three_parameter_command_name(
        "mtrue", "stellar3params", ("high", "high", "young")
    )
    mass_significance = (
        post_fit_analysis._three_parameter_significance_command_name(
            "mtrue", "stellar3params", "Mass", ("high", "old")
        )
    )
    age_significance = (
        post_fit_analysis._three_parameter_significance_command_name(
            "mtrue", "stellar3params", "Age", ("high", "high")
        )
    )
    mass_dynamic_range = (
        post_fit_analysis._three_parameter_dynamic_range_command_name(
            "mtrue", "stellar3params", "Mass", ("high", "old")
        )
    )
    age_dynamic_range = (
        post_fit_analysis._three_parameter_dynamic_range_command_name(
            "mtrue", "stellar3params", "Age", ("high", "high")
        )
    )
    assert (
        rf"high & old & \{lho} & \{hho} & \{mass_significance} & "
        rf"\{mass_dynamic_range} \\"
        in text
    )
    assert (
        rf"high & high & \{hhy} & \{hho} & \{age_significance} & "
        rf"\{age_dynamic_range} \\"
        in text
    )
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
    monkeypatch.setattr(
        post_fit_analysis, "_three_parameter_occurrence_samples",
        lambda result_dir: np.array([0.1, 0.2, 0.3]),
    )
    catalog_path = tmp_path / "stars.csv"
    rows = [
        {"Mstar": mass, "feh": feh, "age": age}
        for mass in (0.82, 1.21)
        for feh in (-0.3, 0.2)
        for age in (2.0, 8.0)
    ]
    rows.extend([
        {"Mstar": 0.5, "feh": 0.2, "age": 8.0},
        {"Mstar": 2.0, "feh": 0.2, "age": 8.0},
    ])
    pd.DataFrame(rows).to_csv(catalog_path, index=False)

    outputs = post_fit_analysis.make_three_parameter_tables(
        tmp_path, use_latex_variables=False,
        stellar_catalog_path=catalog_path,
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
    assert r"\McStellarThreeParamsHighMstarHighFeHYoungIntOcc" not in reordered
    assert reordered.count(r"$0.0\,\sigma$") == 12
    assert reordered.count(" & 1.5 ") == 4
    assert reordered.count(" & 3.2 ") == 4
    assert reordered.count(" & 4.0 ") == 4


def test_make_three_parameter_tables_can_select_occurrence_model(
        tmp_path, monkeypatch):
    outputs = post_fit_analysis.make_three_parameter_tables(
        tmp_path, occurrence_model="logG"
    )
    reordered = outputs["reordered"].read_text()
    original = outputs["original"].read_text()
    occurrence_name = (
        post_fit_analysis._three_parameter_occurrence_command_name(
            "mtrue", "stellar3params", ("high", "high", "young"),
            "logG", "a", 0,
        )
    )
    significance_name = (
        post_fit_analysis._three_parameter_significance_command_name(
            "mtrue", "stellar3params", "Mass", ("high", "old"),
            "logG", "a", 0,
        )
    )
    assert rf"\{occurrence_name}" in reordered
    assert rf"\{occurrence_name}" in original
    assert rf"\{significance_name}" in reordered
    assert r"\McStellarThreeParamsHighMstarHighFeHYoungIntOcc" not in reordered

    monkeypatch.setattr(
        post_fit_analysis, "_three_parameter_statistics",
        lambda result_dir: {
            "Nstars": "47", "Neff": "4.0", "AvgCompl": "0.64",
            "IntOcc": r"0.12^{+0.08}_{-0.05}",
        },
    )

    def occurrence_samples(result_dir, occurrence_model, stack_bin):
        assert occurrence_model == "logG"
        assert stack_bin == 0
        tier2_dir = result_dir.parent.name
        offset = post_fit_analysis._THREE_PARAMETER_TIER2_DIRS.index(
            tier2_dir
        )
        return np.linspace(0.1, 0.3, 101) + 0.01*offset

    monkeypatch.setattr(
        post_fit_analysis, "_model_occurrence_samples", occurrence_samples,
    )
    monkeypatch.setattr(
        post_fit_analysis, "_three_parameter_dynamic_ranges",
        lambda *args: {
            (varied, fixed): 2.0
            for varied, fixed, _, _ in
            post_fit_analysis._three_parameter_comparisons()
        },
    )
    outputs = post_fit_analysis.make_three_parameter_tables(
        tmp_path, occurrence_model="logG", use_latex_variables=False
    )
    reordered = outputs["reordered"].read_text()
    assert r"$0.20^{+0.07}_{-0.07}$" in reordered
    assert r"$0.12^{+0.08}_{-0.05}$" not in reordered


def test_make_two_parameter_tables_create_both_forms(tmp_path):
    outputs = post_fit_analysis.make_two_parameter_tables(tmp_path)
    reordered = outputs["reordered"].read_text()
    original = outputs["original"].read_text()

    assert outputs["reordered"] == (
        tmp_path / "paper_tables" /
        "two_parameter_OR_reordered_mtrue_stellar2params.tex"
    )
    assert outputs["original"] == (
        tmp_path / "paper_tables" /
        "two_parameter_OR_mtrue_stellar2params.tex"
    )
    assert r"\begin{deluxetable*}{ccccc}" in reordered
    assert r"\colhead{Fixed Parameter}" in reordered
    assert r"\colhead{Significance}" in reordered
    assert r"\colhead{High/Low}" in reordered
    low_occurrence = post_fit_analysis._two_parameter_command_name(
        "mtrue", "stellar2params", ("low", "high")
    )
    high_occurrence = post_fit_analysis._two_parameter_command_name(
        "mtrue", "stellar2params", ("high", "high")
    )
    significance = post_fit_analysis._two_parameter_comparison_command_name(
        "mtrue", "stellar2params", "Mass", "high", "Significance"
    )
    ratio = post_fit_analysis._two_parameter_comparison_command_name(
        "mtrue", "stellar2params", "Mass", "high", "DynamicRange"
    )
    assert (
        rf"high & \{low_occurrence} & \{high_occurrence} & "
        rf"\{significance} & \{ratio} \\" in reordered
    )
    assert r"\begin{deluxetable*}{lccccc}" in original
    assert r"\McStellarTwoParamsHighMstarHighFeHNstars" in original
    assert r"\McStellarTwoParamsHighMstarHighFeHPiecewiseIntOcc" in original
    assert not (tmp_path / "paper_tables" / "variables.tex").exists()


def test_make_two_parameter_tables_can_embed_values_without_mass_range(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        post_fit_analysis, "_three_parameter_statistics",
        lambda result_dir: {
            "Nstars": "50", "Neff": "5.0", "AvgCompl": "0.70",
            "IntOcc": r"0.10^{+0.02}_{-0.02}",
        },
    )

    def occurrence_samples(result_dir):
        tier2_dir = result_dir.parent.name
        offset = post_fit_analysis._TWO_PARAMETER_TIER2_DIRS.index(tier2_dir)
        return np.arange(1.0, 11.0) + offset

    monkeypatch.setattr(
        post_fit_analysis, "_three_parameter_occurrence_samples",
        occurrence_samples,
    )
    catalog_path = tmp_path / "stars.csv"
    pd.DataFrame([
        {"Mstar": mass, "feh": feh}
        for mass in (0.5, 0.8, 1.2, 2.0)
        for feh in (-0.3, 0.2)
    ]).to_csv(catalog_path, index=False)

    outputs = post_fit_analysis.make_two_parameter_tables(
        tmp_path, use_latex_variables=False,
        stellar_catalog_path=catalog_path,
    )
    reordered = outputs["reordered"].read_text()
    original = outputs["original"].read_text()

    assert reordered.count(" & 2.5 ") == 2
    assert reordered.count(" & 3.2 ") == 2
    assert r"\McStellarTwoParamsHighMstarHighFeHIntOcc" not in reordered
    assert r"$0.10^{+0.02}_{-0.02}$" in reordered
    assert "high & high & 50 & 5.0 & 0.70" in original


def test_posterior_difference_significance_uses_sign_probability():
    probability, z_score, lower_bound = (
        post_fit_analysis._posterior_difference_significance(
            low_samples=[0.0, 2.0], high_samples=[1.0, 3.0]
        )
    )
    assert probability == pytest.approx(0.75)
    assert z_score == pytest.approx(0.67448975)
    assert not lower_bound

    probability, z_score, lower_bound = (
        post_fit_analysis._posterior_difference_significance(
            low_samples=[0.0, 1.0, 2.0, 3.0],
            high_samples=[4.0, 5.0, 6.0, 7.0],
        )
    )
    assert probability == 1.0
    assert z_score == pytest.approx(0.67448975)
    assert lower_bound
    assert post_fit_analysis._format_posterior_significance(
        z_score, lower_bound
    ) == r">0.7\,\sigma"


def test_three_parameter_dynamic_ranges_use_stellar_medians(tmp_path):
    catalog_path = tmp_path / "stars.csv"
    pd.DataFrame([
        {"Mstar": mass, "feh": feh, "age": age}
        for mass in (0.82, 1.21)
        for feh in (-0.3, 0.2)
        for age in (2.0, 8.0)
    ]).to_csv(catalog_path, index=False)

    dynamic_ranges = post_fit_analysis._three_parameter_dynamic_ranges(
        catalog_path
    )

    assert dynamic_ranges[("Mass", ("high", "old"))] == pytest.approx(
        1.21/0.82
    )
    assert dynamic_ranges[("FeH", ("high", "old"))] == pytest.approx(
        10**0.2/10**-0.3
    )
    assert dynamic_ranges[("Age", ("high", "high"))] == pytest.approx(4.0)


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


def test_make_variables_adds_three_parameter_parametric_occurrence(tmp_path):
    tier1 = tmp_path / "mtrue"
    _write_summary(tier1, "allstars", "roi")
    subset = "highMstarlowFeHhighAct"
    _write_summary(
        tier1, subset, "stellar3params",
        cell_weights=np.array([4.0]),
        cell_compls=np.array([0.6]),
        n_abins=1,
        n_mbins=1,
        a_m_lims_pairs=np.array([[[1.0, 10.0], [1.0, 10.0]]]),
        mode_OR_single=np.array([0.2]),
        hdi_low_OR_single=np.array([0.15]),
        hdi_high_OR_single=np.array([0.25]),
    )
    chain_dir = tier1 / subset / "stellar3params" / "saved_chains"
    chain_dir.mkdir()
    sample_axis = np.linspace(0.0, 1.0, 101)
    np.savez(
        chain_dir / "chains_logG_bin0.npz",
        flat_chains=np.column_stack([
            0.1 + 0.01*sample_axis,
            0.4 + 0.02*sample_axis,
            0.3 + 0.01*sample_axis,
        ]),
        model_bounds=np.array([1.0, 10.0]),
        stack_bounds=np.array([1.0, 10.0]),
    )

    text = post_fit_analysis.make_variables(
        tmp_path, ["mtrue"], ["allstars"], ["roi"]
    ).read_text()

    assert r"\McStellarThreeParamsHighMstarLowFeHYoungPiecewiseIntOcc" in text
    assert r"\McStellarThreeParamsHighMstarLowFeHYoungLogGIntOccBinaZero" in text
    assert r"\McStellarThreeParamsHighMstarLowFeHYoungLogGParamABinaZero" not in text
    assert "% Parametric integrated occurrence: logG" in text


def test_make_variables_adds_three_parameter_significances(
        tmp_path, monkeypatch):
    tier1 = tmp_path / "mtrue"
    _write_summary(tier1, "allstars", "roi")
    for index, tier2_dir in enumerate(
            post_fit_analysis._THREE_PARAMETER_TIER2_DIRS):
        result_dir = tier1 / tier2_dir / "stellar3params"
        summary_dir = result_dir / "saved_dicts"
        chain_dir = result_dir / "saved_chains"
        summary_dir.mkdir(parents=True)
        chain_dir.mkdir()
        (summary_dir / post_fit_analysis.SUMMARY_FILENAME).touch()
        (chain_dir / "chains_piecewise.npz").touch()

    monkeypatch.setattr(
        post_fit_analysis, "_three_parameter_statistics",
        lambda path: {
            "Nstars": "47", "Neff": "4.0", "AvgCompl": "0.64",
            "IntOcc": r"0.12^{+0.08}_{-0.05}",
        },
    )

    def samples_for_result(result_dir):
        tier2_dir = result_dir.parent.name
        offset = post_fit_analysis._THREE_PARAMETER_TIER2_DIRS.index(
            tier2_dir
        )
        return np.arange(1.0, 11.0) + offset

    monkeypatch.setattr(
        post_fit_analysis, "_three_parameter_occurrence_samples",
        samples_for_result,
    )
    catalog_path = tmp_path / "stars.csv"
    pd.DataFrame([
        {"Mstar": mass, "feh": feh, "age": age}
        for mass in (0.82, 1.21)
        for feh in (-0.3, 0.2)
        for age in (2.0, 8.0)
    ]).to_csv(catalog_path, index=False)

    text = post_fit_analysis.make_variables(
        tmp_path, ["mtrue"], ["allstars"], ["roi"],
        stellar_catalog_path=catalog_path,
    ).read_text()
    comparison_names = [
        post_fit_analysis._three_parameter_significance_command_name(
            "mtrue", "stellar3params", varied_parameter, fixed_levels
        )
        for varied_parameter, fixed_levels, _, _ in
        post_fit_analysis._three_parameter_comparisons()
    ]
    assert len(comparison_names) == 12
    for name in comparison_names:
        assert rf"\newcommand{{\{name}}}" in text
    dynamic_range_names = [
        post_fit_analysis._three_parameter_dynamic_range_command_name(
            "mtrue", "stellar3params", varied_parameter, fixed_levels
        )
        for varied_parameter, fixed_levels, _, _ in
        post_fit_analysis._three_parameter_comparisons()
    ]
    for name in dynamic_range_names:
        assert rf"\newcommand{{\{name}}}" in text
    mass_name = post_fit_analysis._three_parameter_dynamic_range_command_name(
        "mtrue", "stellar3params", "Mass", ("high", "old")
    )
    assert (
        rf"\newcommand{{\{mass_name}}}{{\ensuremath{{1.5}}}}" in text
    )


def test_plot_companions_by_stellar_parameter_uses_saved_hosts(
        tmp_path, monkeypatch):
    import pandas as pd
    from matplotlib.axes import Axes

    result_dir = tmp_path / "mtrue" / "allstars" / "fit"
    saved_dicts = result_dir / "saved_dicts"
    saved_dicts.mkdir(parents=True)
    np.savez(
        saved_dicts / "fit_data.npz",
        companion_names=np.array(["host1_0"]),
        x_bounds=np.array([0.1, 10.0]),
        y_bounds=np.array([0.5, 10.0]),
    )
    average_map = result_dir.parent / "avg_map"
    average_map.mkdir()
    np.save(average_map / "parent_xgrid.npy", [0.1, 1.0, 10.0])
    np.save(average_map / "parent_ygrid.npy", [0.5, 2.0, 10.0])
    np.save(
        average_map / "parent_zgrid.npy",
        [[0.1, 0.3, 0.5], [0.2, 0.5, 0.7], [0.4, 0.7, 0.9]],
    )
    catalog_path = tmp_path / "companions.csv"
    pd.DataFrame({
        "CPS Name": ["Host 1 b", "Host 1 c", "Host 2 b"],
        "cps_identifier": ["host1", "host1", "host2"],
        "comp_ind": [0, 1, 0],
        "a_post_med": [0.5, np.nan, 1.0],
        "a_pre_med": [0.4, 2.5, 0.9],
        "Mtrue_post_med": [np.nan, 4.0, 2.0],
        "Msini_pre_med": [1.5, 3.5, 1.8],
        "Mstar": [0.8, 0.8, 1.1],
        "feh": [-0.2, -0.2, 0.1],
        "age": [5.0, 5.0, 2.0],
    }).to_csv(catalog_path, index=False)

    plotted = []
    original_scatter = Axes.scatter

    def capture_scatter(axis, x, y, *args, **kwargs):
        plotted.append({
            "x": np.asarray(x),
            "y": np.asarray(y),
            "color": np.asarray(kwargs["c"]),
        })
        return original_scatter(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    outputs = post_fit_analysis.plot_companions_by_stellar_parameter(
        tmp_path,
        tier1_dirs=["mtrue"],
        tier2_types=["allstars"],
        tier3_dirs=["fit"],
        stellar_parameters=["FeH", "Mstar"],
        catalog_path=catalog_path,
    )
    feh_output = outputs["mtrue/allstars/fit/FeH"]
    mass_output = outputs["mtrue/allstars/fit/Mstar"]

    assert feh_output == result_dir / "plots" / "companions_by_feh.png"
    assert mass_output == result_dir / "plots" / "companions_by_mstar.png"
    assert feh_output.is_file()
    assert mass_output.is_file()
    np.testing.assert_allclose(plotted[0]["x"], [0.5, 2.5])
    np.testing.assert_allclose(plotted[0]["y"], [1.5, 4.0])
    np.testing.assert_allclose(plotted[0]["color"], [-0.2, -0.2])
    np.testing.assert_allclose(plotted[1]["color"], [0.8, 0.8])


def test_plot_companions_by_age_handles_missing_age_and_mass_cut(
        tmp_path, monkeypatch):
    import pandas as pd
    from matplotlib.axes import Axes

    result_dir = tmp_path / "mtrue" / "allstars" / "fit"
    saved_dicts = result_dir / "saved_dicts"
    saved_dicts.mkdir(parents=True)
    np.savez(
        saved_dicts / "fit_data.npz",
        companion_names=np.array(["host1_0", "host2_0", "host3_0"]),
        x_bounds=np.array([0.1, 10.0]),
        y_bounds=np.array([0.5, 10.0]),
    )
    average_map = result_dir.parent / "avg_map"
    average_map.mkdir()
    np.save(average_map / "parent_xgrid.npy", [0.01, 0.1, 10.0, 100.0])
    np.save(average_map / "parent_ygrid.npy", [0.1, 0.5, 10.0, 100.0])
    np.save(
        average_map / "parent_zgrid.npy",
        np.linspace(0.1, 0.9, 16).reshape(4, 4),
    )
    catalog_path = tmp_path / "companions.csv"
    pd.DataFrame({
        "cps_identifier": ["host1", "host2", "host3"],
        "a_post_med": [0.5, 1.0, 2.0],
        "a_pre_med": [0.5, 1.0, 2.0],
        "Mtrue_post_med": [1.0, 2.0, 3.0],
        "Msini_pre_med": [1.0, 2.0, 3.0],
        "Mstar": [0.82, 1.21, 1.3],
        "age": [5.0, np.nan, 7.0],
    }).to_csv(catalog_path, index=False)

    calls = []
    original_scatter = Axes.scatter

    def capture_scatter(axis, x, y, *args, **kwargs):
        calls.append({
            "x": np.asarray(x),
            "color": kwargs.get("color"),
            "marker": kwargs.get("marker"),
            "c": np.asarray(kwargs["c"]) if "c" in kwargs else None,
        })
        return original_scatter(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    output = post_fit_analysis._plot_companions_for_result(
        result_dir, stellar_parameter="Age", catalog_path=catalog_path,
        tier1_name="mtrue",
    )

    assert output.is_file()
    np.testing.assert_allclose(calls[0]["x"], [0.5])
    np.testing.assert_allclose(calls[0]["c"], [5.0])
    np.testing.assert_allclose(calls[1]["x"], [1.0])
    assert calls[1]["color"] == "gray"
    np.testing.assert_allclose(calls[2]["x"], [2.0])
    assert calls[2]["color"] == "black"
    assert calls[2]["marker"] == "x"


def test_latex_token_spells_out_digits():
    assert post_fit_analysis._latex_token("stellar_3params_Miyazaki") == (
        "StellarThreeParamsMiyazaki"
    )
    assert post_fit_analysis._latex_token("bin10") == "BinOneZero"
    assert post_fit_analysis._latex_token("highMstar") == "HighMstar"


def test_make_variables_accepts_several_three_parameter_runs(
        tmp_path, monkeypatch):
    tier1 = tmp_path / "mtrue"
    _write_summary(tier1, "allstars", "roi")
    subset = "highMstarhighFeHhighAct"
    for t3 in ("stellar3params", "stellar_3params_Miyazaki"):
        summary_dir = tier1 / subset / t3 / "saved_dicts"
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
        tmp_path, ["mtrue"], ["allstars"], ["roi"],
        three_parameter_t3=["stellar3params", "stellar_3params_Miyazaki"],
    ).read_text()

    assert r"\newcommand{\McStellarThreeParamsHighMstarHighFeHYoungNstars}" in text
    assert (
        r"\newcommand{\McStellarThreeParamsMiyazakiHighMstarHighFeHYoungNstars}"
        in text
    )
    command_names = re.findall(r"\\newcommand\{\\([^}]*)\}", text)
    assert all(re.fullmatch(r"[A-Za-z]+", name) for name in command_names)


def test_make_variables_can_skip_three_parameter_results(tmp_path, capsys):
    _write_summary(tmp_path / "mtrue", "allstars", "roi")
    text = post_fit_analysis.make_variables(
        tmp_path, ["mtrue"], ["allstars"], ["roi"], three_parameter_t3=None,
    ).read_text()
    assert "HighFeHYoung" not in text
    assert "three-parameter" not in capsys.readouterr().out


def _write_model_chain(path, model, samples, bounds=(0.4, 50.0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, flat_chains=np.asarray(samples, dtype=float),
             model_bounds=np.array(bounds))


def test_model_cdf_samples_normalize_each_posterior_sample(tmp_path):
    chain = tmp_path / "chains_sigmoid_bin0.npz"
    _write_model_chain(chain, "sigmoid", [
        [0.1, 0.01, 0.5, 0.2], [0.2, 0.001, 1.0, 0.1], [0.05, 0.05, 0.0, 0.3],
    ])

    grid, cdfs = post_fit_analysis.model_cdf_samples(chain, "sigmoid",
                                                     n_grid=200)

    assert grid[0] == pytest.approx(0.4) and grid[-1] == pytest.approx(50.0)
    assert cdfs.shape == (3, 200)
    assert np.allclose(cdfs[:, 0], 0) and np.allclose(cdfs[:, -1], 1)
    assert np.all(np.diff(cdfs, axis=1) >= 0)
    # Equal plateaus make a flat density, whose CDF is linear in log mass.
    assert np.allclose(
        cdfs[2], np.log10(grid/0.4)/np.log10(50/0.4), atol=1e-6
    )


def test_model_cdf_samples_thin_long_chains(tmp_path):
    chain = tmp_path / "chains_logG_bin0.npz"
    _write_model_chain(chain, "logG", [[0.1, 0.3, 0.4]]*50)
    _, cdfs = post_fit_analysis.model_cdf_samples(chain, "logG",
                                                  max_samples=7)
    assert len(cdfs) == 7


def _write_cdf_chains(root, tier2s=("highMstar", "lowMstar", "highFeH",
                                      "lowFeH"),
                      models=("sigmoid", "logG")):
    parameters = {"sigmoid": [[0.1, 0.01, 0.5, 0.2], [0.12, 0.02, 0.6, 0.25]],
                  "logG": [[0.1, 0.3, 0.4], [0.12, 0.35, 0.45]]}
    for tier2 in tier2s:
        for model in models:
            _write_model_chain(
                root / "mtrue" / tier2 / "paper_bounds" / "saved_chains" /
                f"chains_{model}_bin0.npz", model, parameters[model],
            )


def _cdf_row(*tier2s):
    return [{"label": tier2, "t1": "mtrue", "t2": tier2, "t3": "paper_bounds"}
            for tier2 in tier2s]


_CDF_ROWS = [_cdf_row("highMstar", "lowMstar"), _cdf_row("highFeH", "lowFeH")]


def _capture_cdf_figure(monkeypatch):
    from matplotlib.figure import Figure
    figures = []
    original = Figure.savefig
    monkeypatch.setattr(
        Figure, "savefig",
        lambda self, *a, **k: figures.append(self) or original(self, *a, **k),
    )
    return figures


def test_plot_model_cdf_comparison_draws_models_by_sample_pairs(
        tmp_path, monkeypatch):
    _write_cdf_chains(tmp_path)
    figures = _capture_cdf_figure(monkeypatch)

    output = post_fit_analysis.plot_model_cdf_comparison(
        tmp_path, _CDF_ROWS, ["sigmoid", "logG"], "cdf_comparison",
        credible=0.95,
    )

    assert output == tmp_path / "cdf_comparisons" / "cdf_comparison.png"
    assert output.is_file()
    axes = figures[0].axes
    assert len(axes) == 4
    assert axes[0].get_subplotspec().get_gridspec().get_geometry() == (2, 2)
    assert [axis.get_title() for axis in axes] == [
        "Sigmoid CDF", "Log-Gaussian CDF", "", "",
    ]
    assert all(len(axis.get_lines()) == 2 for axis in axes)
    assert [axis.get_legend() is not None for axis in axes] == [
        True, False, True, False,
    ]
    assert [line.get_label() for line in axes[2].get_lines()] == [
        "highFeH", "lowFeH",
    ]
    assert [bool(axis.get_xlabel()) for axis in axes] == [
        False, False, True, True,
    ]
    assert [bool(axis.get_ylabel()) for axis in axes] == [
        True, False, True, False,
    ]


def test_plot_model_cdf_comparison_custom_title_heads_each_column(
        tmp_path, monkeypatch):
    _write_cdf_chains(tmp_path)
    figures = _capture_cdf_figure(monkeypatch)
    post_fit_analysis.plot_model_cdf_comparison(
        tmp_path, _CDF_ROWS[:1], ["sigmoid", "logG"], "fixed",
        title="Fit: {model}",
    )
    post_fit_analysis.plot_model_cdf_comparison(
        tmp_path, _CDF_ROWS[:1], "logG", "untitled", title=None,
    )
    assert [axis.get_title() for axis in figures[0].axes] == [
        "Fit: Sigmoid", "Fit: Log-Gaussian",
    ]
    assert [axis.get_title() for axis in figures[1].axes] == [""]


def test_plot_model_cdf_comparison_needs_every_model_for_every_sample(
        tmp_path):
    _write_cdf_chains(tmp_path, tier2s=("highMstar", "lowMstar"))
    _write_cdf_chains(tmp_path, tier2s=("highFeH", "lowFeH"),
                      models=("sigmoid",))
    with pytest.raises(FileNotFoundError, match="every model") as error:
        post_fit_analysis.plot_model_cdf_comparison(
            tmp_path, _CDF_ROWS, ["sigmoid", "logG"], "missing",
        )
    message = str(error.value)
    assert "highFeH/paper_bounds/saved_chains/chains_logG_bin0" in message
    assert "lowFeH/paper_bounds/saved_chains/chains_logG_bin0" in message
    assert message.count("chains_") == 2
    assert not (tmp_path / "cdf_comparisons").exists()


def test_plot_model_cdf_comparison_rejects_empty_rows(tmp_path):
    with pytest.raises(ValueError, match="rows"):
        post_fit_analysis.plot_model_cdf_comparison(
            tmp_path, [_cdf_row("highMstar"), []], ["sigmoid"], "x",
        )
