"""Fuentes reales: parseo de los reportes de la CNE (frase, tabla y OCR) y series versionadas."""
import numpy as np
import pandas as pd
import pytest

from src.data import real_sources as RS


def test_sentence_format_maps_report_to_previous_month():
    text = ("Por su parte la barra Crucero registró un costo marginal promedio de 60,7 USD/MWh, "
            "lo que representó un incremento de 25,7% con respecto al mes anterior.")
    obs = RS.parse_crucero(text, "2026-06")
    direct = [o for o in obs if o["source"] == "texto"][0]
    assert direct["month"] == "2026-05" and direct["cmg_usd_mwh"] == 60.7
    prev = [o for o in obs if o["source"] == "derivado_mensual"][0]
    assert prev["month"] == "2026-04" and prev["cmg_usd_mwh"] == pytest.approx(60.7 / 1.257, abs=0.05)


def test_sentence_with_decrease_word_and_positive_number_is_a_decrease():
    text = ("la barra Crucero registró un costo marginal promedio de 106,3 USD/MWh, lo que representó una "
            "disminución de 2,3% con respecto al mes anterior.")
    prev = [o for o in RS.parse_crucero(text, "2023-06", "ocr") if o["source"] == "derivado_mensual"][0]
    assert prev["cmg_usd_mwh"] > 106.3


def test_table_format_reconstructs_previous_month_and_year():
    text = """Barra       [USD/MWh]  Mensual      Anual
    Quillota           34,5         -0,9%    -35,6%
    Crucero           34,0          -0,5%    -34,8%
    PNE      $/kWh Mensual"""
    obs = {o["source"]: o for o in RS.parse_crucero(text, "2020-01")}
    assert obs["tabla"]["month"] == "2019-12" and obs["tabla"]["cmg_usd_mwh"] == 34.0
    assert obs["derivado_mensual"]["month"] == "2019-11"
    assert obs["derivado_anual"]["month"] == "2018-12"
    assert obs["derivado_anual"]["cmg_usd_mwh"] == pytest.approx(34.0 / (1 - 0.348), abs=0.05)


def test_published_value_wins_over_derived_one():
    obs = [{"month": "2022-05", "cmg_usd_mwh": 90.0, "source": "derivado_anual"},
           {"month": "2022-05", "cmg_usd_mwh": 88.1, "source": "ocr"},
           {"month": "2022-05", "cmg_usd_mwh": 87.0, "source": "derivado_mensual"}]
    out = RS.combine_observations(obs)
    assert len(out) == 1 and out.iloc[0]["source"] == "ocr" and out.iloc[0]["cmg_usd_mwh"] == 88.1


def test_no_crucero_means_no_observation():
    assert RS.parse_crucero("Reporte sin datos de barras", "2024-01") == []


def test_versioned_price_series_is_monthly_plausible_and_traceable():
    cmg = RS.load_cmg()
    months = pd.PeriodIndex(cmg["month"], freq="M")
    assert months.is_monotonic_increasing and months.is_unique
    assert cmg["cmg_usd_mwh"].between(10, 300).all()
    assert set(cmg["source"]) <= set(RS.SOURCE_PRIORITY)
    assert (cmg["source"].isin(["texto", "tabla", "ocr"])).mean() > 0.6   # mayoria publicada, no derivada


def test_versioned_cochilco_series_matches_its_axis():
    coc = RS.load_cochilco()
    assert list(coc["year"]) == list(range(2010, 2025))
    assert coc["mj_per_t"].between(70, 85).all()
    np.testing.assert_allclose(coc["kwh_per_t"], coc["mj_per_t"] / RS.MJ_PER_KWH, atol=0.01)


def test_a_report_that_copies_the_previous_one_is_discarded():
    """Caso real: el reporte de enero de 2024 repite la frase de diciembre de 2023 (45,6 USD/MWh, +3,9%)."""
    obs = [{"month": "2023-11", "cmg_usd_mwh": 45.6, "source": "ocr", "change_pct": 3.9},
           {"month": "2023-12", "cmg_usd_mwh": 45.6, "source": "ocr", "change_pct": 3.9},
           {"month": "2024-01", "cmg_usd_mwh": 50.0, "source": "ocr", "change_pct": 9.6}]
    out = RS.combine_observations(RS.flag_repeated_reports(obs))
    assert list(out["month"]) == ["2023-11", "2024-01"]


def test_published_and_reconstructed_values_agree_across_reports():
    """Validacion de la extraccion: dos reportes distintos implican el mismo valor para el mismo mes."""
    cv = RS.cross_validation(RS.load_cmg_observations())
    assert cv["pairs"] >= 50
    assert cv["p95_rel_diff"] < 0.01 and cv["max_rel_diff"] < 0.02
    assert cv["ocr_pairs"] >= 5 and cv["ocr_max_rel_diff"] < 0.01
    assert "RMensual_v202401.pdf" in cv["discarded_reports"]
