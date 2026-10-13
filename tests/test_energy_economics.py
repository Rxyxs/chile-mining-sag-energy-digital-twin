"""Economia de la energia con precios reales y verificacion del simulador."""
import pandas as pd
import pytest

from src.models import energy_economics as EE


def _cmg():
    return pd.DataFrame({"month": ["2024-01", "2024-02", "2024-03"], "cmg_usd_mwh": [40.0, 60.0, 80.0],
                         "source": ["texto", "tabla", "ocr"]})


def test_annual_cost_and_value_of_one_percent():
    assert EE.annual_energy_cost(25.0, 60.0) == pytest.approx(25 * 8760 * 60)
    v = EE.efficiency_value(25.0, {"a": 60.0})
    assert v["a"] == pytest.approx(25 * 8760 * 60 * 0.01)


def test_price_summary_reports_extremes_with_their_months():
    p = EE.price_summary(_cmg())
    assert p["mean"] == 60.0 and p["min_month"] == "2024-01" and p["max_month"] == "2024-03"
    assert p["by_year"] == {"2024": 60.0}


def test_monthly_cost_uses_calendar_hours():
    out = EE.monthly_cost_series(10.0, _cmg())
    feb = out.set_index("month").loc["2024-02"]
    assert feb["energy_mwh"] == 10.0 * 29 * 24          # 2024 es bisiesto
    assert feb["cost_usd"] == feb["energy_mwh"] * 60.0


def test_simulator_check_reports_implied_grinding_share():
    sim = pd.DataFrame({"specific_energy_kwh_t": [10.0, 12.0], "mill_power_mw": [25.0, 25.0]})
    coc = pd.DataFrame({"year": [2023, 2024], "mj_per_t": [72.0, 79.2], "kwh_per_t": [20.0, 22.0]})
    chk = EE.simulator_check(sim, coc)
    assert chk["cochilco_year"] == 2024 and chk["implied_grinding_share"] == pytest.approx(11.0 / 22.0)


def test_run_energy_economics_keeps_the_old_assumption_for_comparison():
    sim = pd.DataFrame({"specific_energy_kwh_t": [11.0] * 4, "mill_power_mw": [25.0] * 4})
    coc = pd.DataFrame({"year": [2024], "mj_per_t": [76.0], "kwh_per_t": [21.1]})
    out = EE.run_energy_economics(sim, _cmg(), coc)
    assert out["annual_cost_usd"]["supuesto_anterior"] == pytest.approx(25 * 8760 * 70)
    assert set(out["value_of_1pct_usd"]) >= {"p10", "mediana", "p90"}
