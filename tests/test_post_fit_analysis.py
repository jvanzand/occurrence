import numpy as np
import pandas as pd
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
    assert r"\begin{deluxetable*}{cccccc}" in text
    assert r"\multicolumn{2}{c}{Occurrence Rate}" in text
    assert r"\colhead{Significance}" in text
    assert r"\colhead{Dynamic Range}" in text
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
    assert r"\McHighMstarHighFeHYoungIntOcc" not in reordered
    assert reordered.count(r"$0.0\,\sigma$") == 12
    assert reordered.count(" & 1.5 ") == 4
    assert reordered.count(" & 3.2 ") == 4
    assert reordered.count(" & 4.0 ") == 4


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
