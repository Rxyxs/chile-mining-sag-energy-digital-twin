"""Economia de la energia del molino con precios reales y verificacion del simulador contra Cochilco.

Dos preguntas que el simulador solo no puede responder:

1. **Cuanto vale la energia que consume el molino.** Antes se usaba un precio supuesto de
   70 USD/MWh. Ahora se usa el costo marginal real de la barra Crucero 220 kV (CNE), mes a mes.
   El costo marginal es el precio spot del sistema: una minera grande compra la mayor parte de su
   energia con contratos de largo plazo, asi que este precio mide el costo de oportunidad de la
   energia ahorrada, no la tarifa de una faena en particular.

2. **Si el simulador produce consumos plausibles.** Cochilco publica el consumo electrico unitario
   de todas las concentradoras de cobre de Chile (MJ por tonelada procesada). La energia de
   molienda del simulador debe ser una fraccion de ese total, porque la concentradora tambien
   incluye chancado, flotacion, remolienda, espesamiento y bombeo. Se reporta esa fraccion
   implicita; no se ajusta el simulador para forzarla.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 8760


def price_summary(cmg: pd.DataFrame) -> dict:
    """Estadisticos del costo marginal mensual (USD/MWh), completo y de los ultimos 12 meses."""
    s = cmg.set_index("month")["cmg_usd_mwh"].sort_index()
    last12 = s.iloc[-12:]
    return {
        "months": int(len(s)),
        "from": s.index[0],
        "to": s.index[-1],
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p10": float(s.quantile(0.10)),
        "p90": float(s.quantile(0.90)),
        "min": float(s.min()),
        "min_month": s.idxmin(),
        "max": float(s.max()),
        "max_month": s.idxmax(),
        "last12_mean": float(last12.mean()),
        "by_year": {str(y): float(v) for y, v in s.groupby(s.index.str[:4]).mean().items()},
        "sources": cmg["source"].value_counts().to_dict(),
    }


def annual_energy_cost(mean_power_mw: float, price_usd_mwh: float) -> float:
    """Costo anual (USD) de operar a una potencia media constante."""
    return mean_power_mw * HOURS_PER_YEAR * price_usd_mwh


def efficiency_value(mean_power_mw: float, prices: dict[str, float], gain: float = 0.01) -> dict:
    """Valor anual de reducir la energia especifica en ``gain`` (fraccion), a distintos precios."""
    return {k: annual_energy_cost(mean_power_mw, p) * gain for k, p in prices.items()}


def monthly_cost_series(mean_power_mw: float, cmg: pd.DataFrame) -> pd.DataFrame:
    """Costo mensual de la energia del molino si hubiera operado a su potencia media en cada mes real."""
    out = cmg[["month", "cmg_usd_mwh", "source"]].copy()
    hours = pd.PeriodIndex(out["month"], freq="M").days_in_month * 24
    out["energy_mwh"] = mean_power_mw * np.asarray(hours)
    out["cost_usd"] = out["energy_mwh"] * out["cmg_usd_mwh"]
    return out


def simulator_check(sim: pd.DataFrame, cochilco: pd.DataFrame) -> dict:
    """Energia especifica de molienda simulada frente al consumo real de las concentradoras."""
    grinding = float(sim["specific_energy_kwh_t"].mean())
    last = cochilco.sort_values("year").iloc[-1]
    return {
        "sim_grinding_kwh_t": grinding,
        "sim_grinding_p10_p90": [float(sim["specific_energy_kwh_t"].quantile(q)) for q in (0.1, 0.9)],
        "cochilco_year": int(last["year"]),
        "cochilco_concentrator_kwh_t": float(last["kwh_per_t"]),
        "cochilco_range_kwh_t": [float(cochilco["kwh_per_t"].min()), float(cochilco["kwh_per_t"].max())],
        "implied_grinding_share": grinding / float(last["kwh_per_t"]),
    }


def run_energy_economics(sim: pd.DataFrame, cmg: pd.DataFrame, cochilco: pd.DataFrame, assumed_price: float = 70.0) -> dict:
    mean_power = float(sim["mill_power_mw"].mean())
    prices = price_summary(cmg)
    levels = {"supuesto_anterior": assumed_price, "p10": prices["p10"], "mediana": prices["median"],
              "promedio": prices["mean"], "p90": prices["p90"], "ultimos_12_meses": prices["last12_mean"]}
    return {
        "mean_power_mw": mean_power,
        "annual_energy_mwh": mean_power * HOURS_PER_YEAR,
        "price": prices,
        "annual_cost_usd": {k: annual_energy_cost(mean_power, p) for k, p in levels.items()},
        "value_of_1pct_usd": efficiency_value(mean_power, levels),
        "simulator_check": simulator_check(sim, cochilco),
    }
