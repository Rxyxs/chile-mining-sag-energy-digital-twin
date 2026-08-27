"""Analisis de supervivencia (Cox Proportional Hazards) para tiempo restante
hasta falla mecanica del Molino SAG (RUL / RTF -- Remaining Useful Life).

No existe un log de fallas reales en el simulador (`src.data.simulation`
genera series continuas de operacion, no eventos de mantenimiento), asi que
este modulo construye un dataset de supervivencia sintetico pero fisicamente
consistente: la serie horaria completa se divide en ciclos de operacion de
longitud fija (ventanas deslizantes), cada ciclo se resume en covariables de
estres mecanico/termico (desviacion de carga respecto al optimo metalurgico,
desviacion de carga de bolas, variabilidad de la energia especifica, dureza
promedio del mineral), y el tiempo hasta falla de cada ciclo se genera con un
modelo de riesgos proporcionales de base Weibull (creciente en el tiempo,
consistente con desgaste mecanico tipo "wear-out") cuyo hazard depende
linealmente de esas covariables -- exactamente la relacion que CoxPH debe
recuperar al ajustar sobre los datos. Los ciclos que exceden el horizonte de
observacion se censuran administrativamente (censura por la derecha), como
ocurriria con equipos que siguen operando al cierre del periodo de estudio.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

CYCLE_LENGTH_HOURS = 72  # ventana de resumen por ciclo (3 dias de operacion)
CYCLE_STRIDE_HOURS = 24  # paso entre ciclos consecutivos (solapados)
SURVIVAL_HISTORY_HOURS = 24 * 720  # 2 anios de historia simulada: los 180 dias
# del dataset operacional principal no acumulan ciclos suficientes para que un
# CoxPH identifique con estabilidad efectos de largo plazo sobre la falla
# mecanica, asi que el modulo de supervivencia simula su propia historia mas
# larga con la misma fisica de `simulate_sag_mill_operation`.
ADMIN_CENSOR_HOURS = 1_400.0  # horizonte de observacion (~58 dias) -> censura administrativa
WEIBULL_SHAPE = 1.8  # k > 1: hazard creciente en el tiempo (desgaste mecanico)
BASELINE_SCALE_HOURS = 700.0  # vida caracteristica en condiciones de referencia

COVARIATE_COLS = [
    "load_deviation_from_optimum",
    "ball_charge_deviation",
    "specific_energy_std",
    "hardness_proxy_wi_mean",
]

# Coeficientes verdaderos del generador sintetico (log-hazard ratio por unidad
# de covariable, centrada en su referencia de operacion optima) -- CoxPH debe
# recuperarlos al ajustar sobre los ciclos generados. `load_deviation` y
# `ball_charge_deviation` llevan peso deliberadamente bajo: son setpoints que
# el operador mantiene cerca del optimo, por lo que su varianza ciclo a ciclo
# es intrinsecamente pequena y su efecto es dificil de identificar -- igual
# que en una planta real, donde la variabilidad de un setpoint bien controlado
# rara vez alcanza significancia estadistica frente a la carga termica/energetica.
TRUE_COEFS = {
    "load_deviation_from_optimum": 0.10,
    "ball_charge_deviation": 0.12,
    "specific_energy_std": 1.40,
    "hardness_proxy_wi_mean": 0.14,
}
REFERENCE_VALUES = {
    "load_deviation_from_optimum": 2.4,
    "ball_charge_deviation": 0.8,
    "specific_energy_std": 1.0,
    "hardness_proxy_wi_mean": 13.0,
}

RANDOM_STATE = 42


def build_cycle_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Resume la serie horaria en ciclos de operacion de ``CYCLE_LENGTH_HOURS``
    (ventanas deslizantes cada ``CYCLE_STRIDE_HOURS``), calculando por ciclo
    las covariables de estres mecanico que alimentan el modelo de riesgo."""
    d = df.sort_values("timestamp").reset_index(drop=True)
    n = len(d)
    rows = []
    start = 0
    cycle_id = 0
    while start + CYCLE_LENGTH_HOURS <= n:
        window = d.iloc[start:start + CYCLE_LENGTH_HOURS]
        rows.append({
            "cycle_id": cycle_id,
            "cycle_start": window["timestamp"].iloc[0],
            "load_deviation_from_optimum": float((window["mill_load_pct"] - 30.0).abs().mean()),
            "ball_charge_deviation": float((window["ball_charge_pct"] - 11.0).abs().mean()),
            "specific_energy_std": float(window["specific_energy_kwh_t"].std()),
            "hardness_proxy_wi_mean": float(window["hardness_proxy_wi"].mean()),
        })
        cycle_id += 1
        start += CYCLE_STRIDE_HOURS
    return pd.DataFrame(rows)


def simulate_survival_times(covariates: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Genera tiempo-hasta-falla sintetico por ciclo con un modelo de riesgos
    proporcionales de base Weibull: bajo PH, S(t|x) = S0(t)^exp(eta), y para
    una base Weibull(shape=k, scale=b0) la inversa de la CDF es

        T = b0 * (-ln(U))^(1/k) * exp(-eta / k),  U ~ Uniform(0, 1)

    con eta = suma_j coef_j * (x_j - referencia_j). Ciclos con T mayor al
    horizonte de observacion se censuran por la derecha en ese horizonte.
    """
    rng = np.random.default_rng(seed)
    out = covariates.copy()

    eta = np.zeros(len(out))
    for col, coef in TRUE_COEFS.items():
        eta += coef * (out[col].to_numpy() - REFERENCE_VALUES[col])

    u = rng.uniform(1e-9, 1.0, size=len(out))
    true_failure_time = (
        BASELINE_SCALE_HOURS * (-np.log(u)) ** (1.0 / WEIBULL_SHAPE) * np.exp(-eta / WEIBULL_SHAPE)
    )

    out["duration_hours"] = np.minimum(true_failure_time, ADMIN_CENSOR_HOURS)
    out["event_observed"] = (true_failure_time <= ADMIN_CENSOR_HOURS).astype(int)
    return out


def build_survival_dataset(df: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    covariates = build_cycle_covariates(df)
    return simulate_survival_times(covariates, seed=seed)


def simulate_survival_history(seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Genera la historia operacional larga (`SURVIVAL_HISTORY_HOURS`) usada
    solo para construir ciclos de supervivencia -- independiente de los 180
    dias del dataset principal de forecasting/regresion."""
    from src.data.simulation import simulate_sag_mill_operation

    return simulate_sag_mill_operation(n_hours=SURVIVAL_HISTORY_HOURS, seed=seed)


def fit_coxph(survival_df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[CoxPHFitter, dict]:
    """Ajusta CoxPH sobre una particion cronologica (por ``cycle_start``) de
    los ciclos y evalua discriminacion con el indice de concordancia (C-index)
    sobre los ciclos held-out -- el analogo, en supervivencia, de un AUC."""
    ordered = survival_df.sort_values("cycle_start").reset_index(drop=True)
    split_idx = int(len(ordered) * (1 - test_fraction))
    train_df, test_df = ordered.iloc[:split_idx], ordered.iloc[split_idx:]

    cols = COVARIATE_COLS + ["duration_hours", "event_observed"]
    cph = CoxPHFitter()
    cph.fit(train_df[cols], duration_col="duration_hours", event_col="event_observed")

    test_risk = cph.predict_partial_hazard(test_df[COVARIATE_COLS])
    c_index = concordance_index(
        test_df["duration_hours"], -test_risk, test_df["event_observed"]
    )

    metrics = {
        "n_train_cycles": int(len(train_df)),
        "n_test_cycles": int(len(test_df)),
        "n_events_train": int(train_df["event_observed"].sum()),
        "n_events_test": int(test_df["event_observed"].sum()),
        "concordance_index_test": float(c_index),
        "log_likelihood_ratio_test_p_value": float(cph.log_likelihood_ratio_test().p_value),
        "coefficients": {
            col: {
                "estimated_log_hazard_ratio": float(cph.params_[col]),
                "true_log_hazard_ratio": TRUE_COEFS[col],
                "hazard_ratio": float(np.exp(cph.params_[col])),
                "p_value": float(cph.summary.loc[col, "p"]),
            }
            for col in COVARIATE_COLS
        },
    }
    return cph, metrics


def predict_remaining_life(cph: CoxPHFitter, covariates_row: pd.DataFrame) -> dict:
    """Predice tiempo restante hasta falla (RUL, en horas) para un ciclo dado:
    mediana de la curva de supervivencia individual y hazard ratio relativo a
    un ciclo en condiciones de referencia (operacion optima)."""
    median_rul = cph.predict_median(covariates_row[COVARIATE_COLS])
    median_raw = median_rul.iloc[0] if hasattr(median_rul, "iloc") else float(median_rul)
    hazard_ratio = float(cph.predict_partial_hazard(covariates_row[COVARIATE_COLS]).iloc[0])
    median_value = float(median_raw) if np.isfinite(median_raw) else float(ADMIN_CENSOR_HOURS * 2)
    return {
        "median_remaining_life_hours": median_value,
        "median_remaining_life_days": median_value / 24.0,
        "hazard_ratio_vs_reference": hazard_ratio,
    }


def run_survival_pipeline(models_dir: Path, reports_dir: Path, seed: int = RANDOM_STATE) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    history = simulate_survival_history(seed=seed)
    survival_df = build_survival_dataset(history, seed=seed)
    cph, metrics = fit_coxph(survival_df)

    joblib.dump(cph, models_dir / "coxph_survival_model.joblib")
    survival_df.to_csv(reports_dir / "survival_cycles_dataset.csv", index=False)

    survival_curve = cph.predict_survival_function(
        survival_df[COVARIATE_COLS].iloc[[0]]
    ).reset_index()
    survival_curve.columns = ["hours", "survival_probability"]
    survival_curve.to_csv(reports_dir / "survival_curve_reference_cycle.csv", index=False)

    with open(reports_dir / "survival_model_report.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    report = run_survival_pipeline(base / "outputs" / "models", base / "outputs" / "reports")
    print(json.dumps(report, indent=2))
