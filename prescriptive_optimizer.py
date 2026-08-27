"""Motor de optimizacion prescriptiva de setpoints energeticos del Molino SAG.

Dado un contexto operacional (F80/P80, dureza estimada, estadisticas de
rolling window recientes, hora del dia) que el operador no controla en el
horizonte inmediato, este modulo busca los setpoints controlables --
alimentacion fresca, % de llenado, % de carga de bolas, agua de adicion --
que minimizan el consumo especifico de energia predicho por el modelo
multi-output de `src.models.train_multioutput`, sujeto a un piso minimo de
throughput y a un termino de penalizacion por desgaste mecanico tomado del
modelo de supervivencia CoxPH de `src.models.train_survival`: un setpoint que
ahorra energia empujando la carga o la carga de bolas lejos de su optimo
metalurgico acorta la vida util del equipo, y el optimizador debe balancear
ambos objetivos, no solo minimizar energia a cualquier costo.

Es "prescriptivo" en el sentido de investigacion de operaciones: no describe
ni predice el estado de la planta (eso ya lo hacen los modelos de regresion y
supervivencia), sino que recomienda una accion -- el vector de setpoints --
que optimiza un objetivo de negocio explicito sujeto a restricciones fisicas.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.train_multioutput import FEATURE_COLS

MODEL_DIR = Path(__file__).resolve().parent / "outputs" / "models"

CONTROLLABLE_COLS = ["fresh_feed_tph", "mill_load_pct", "ball_charge_pct", "water_addition_m3h"]

SETPOINT_BOUNDS = {
    "fresh_feed_tph": (1_800.0, 2_900.0),
    "mill_load_pct": (24.0, 36.0),
    "ball_charge_pct": (8.0, 14.0),
    "water_addition_m3h": (700.0, 1_000.0),
}

# Peso relativo del termino de desgaste mecanico (hazard ratio del CoxPH)
# frente al ahorro de energia especifica (kWh/t) en la funcion objetivo.
# lambda=0 -> optimiza energia pura, ignorando el impacto en vida util.
DEFAULT_WEAR_PENALTY_WEIGHT = 0.15


class PrescriptiveOptimizer:
    """Envuelve el modelo de regresion multi-output (energia/throughput) y el
    modelo de supervivencia CoxPH (riesgo de falla) para recomendar setpoints
    optimos dado un contexto operacional puntual."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.regression_model = joblib.load(model_dir / "best_multioutput_model.joblib")
        self.feature_cols = joblib.load(model_dir / "feature_columns.joblib")
        coxph_path = model_dir / "coxph_survival_model.joblib"
        self.survival_model = joblib.load(coxph_path) if coxph_path.exists() else None

    def _build_feature_row(self, context: dict, setpoints: np.ndarray) -> pd.DataFrame:
        """Reconstruye la fila completa de features consumida por el modelo de
        regresion, combinando el contexto fijo (no controlable en el
        horizonte de decision) con el candidato de setpoints propuesto por el
        optimizador -- recalculando las features derivadas que dependen de
        los setpoints, exactamente como `src.features.preprocessing.engineer_features`.
        """
        row = dict(context)
        for col, value in zip(CONTROLLABLE_COLS, setpoints):
            row[col] = value

        row["reduction_ratio"] = row["f80_um"] / row["p80_um"]
        row["load_deviation_from_optimum"] = abs(row["mill_load_pct"] - 30.0)
        row["ball_charge_deviation"] = abs(row["ball_charge_pct"] - 11.0)

        return pd.DataFrame([row])[self.feature_cols]

    def _predict_energy_throughput(self, context: dict, setpoints: np.ndarray) -> tuple[float, float]:
        X = self._build_feature_row(context, setpoints)
        specific_energy, throughput = self.regression_model.predict(X)[0]
        return float(specific_energy), float(throughput)

    def _wear_hazard_ratio(self, setpoints: np.ndarray, context: dict) -> float:
        """Hazard ratio relativo (CoxPH) del ciclo implicado por estos
        setpoints frente a un ciclo en condiciones de referencia. Aproxima la
        desviacion de carga/bolas del setpoint candidato como si se sostuviera
        durante todo el ciclo; la energia especifica std y la dureza se toman
        del contexto (no son controlables en el horizonte inmediato)."""
        if self.survival_model is None:
            return 1.0

        load_pct, ball_pct = setpoints[1], setpoints[2]
        covariates = pd.DataFrame([{
            "load_deviation_from_optimum": abs(load_pct - 30.0),
            "ball_charge_deviation": abs(ball_pct - 11.0),
            "specific_energy_std": context.get("specific_energy_std_recent", 1.0),
            "hardness_proxy_wi_mean": context.get("hardness_proxy_wi", 13.0),
        }])
        return float(self.survival_model.predict_partial_hazard(covariates).iloc[0])

    def _objective(
        self, setpoints: np.ndarray, context: dict, wear_penalty_weight: float,
    ) -> float:
        specific_energy, _ = self._predict_energy_throughput(context, setpoints)
        hazard_ratio = self._wear_hazard_ratio(setpoints, context)
        return specific_energy + wear_penalty_weight * (hazard_ratio - 1.0)

    def optimize_setpoints(
        self,
        context: dict,
        min_throughput_tph: float | None = None,
        wear_penalty_weight: float = DEFAULT_WEAR_PENALTY_WEIGHT,
    ) -> dict:
        """Resuelve el problema de optimizacion restringida (SLSQP) y devuelve
        los setpoints recomendados junto con las metricas antes/despues."""
        x0 = np.array([context.get(c, np.mean(SETPOINT_BOUNDS[c])) for c in CONTROLLABLE_COLS])
        bounds = [SETPOINT_BOUNDS[c] for c in CONTROLLABLE_COLS]

        constraints = []
        if min_throughput_tph is not None:
            def throughput_constraint(x, context=context, floor=min_throughput_tph):
                _, throughput = self._predict_energy_throughput(context, x)
                return throughput - floor
            constraints.append({"type": "ineq", "fun": throughput_constraint})

        result = minimize(
            self._objective, x0, args=(context, wear_penalty_weight),
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6},
        )

        baseline_energy, baseline_throughput = self._predict_energy_throughput(context, x0)
        baseline_hazard = self._wear_hazard_ratio(x0, context)

        optimal_setpoints = result.x
        opt_energy, opt_throughput = self._predict_energy_throughput(context, optimal_setpoints)
        opt_hazard = self._wear_hazard_ratio(optimal_setpoints, context)

        return {
            "converged": bool(result.success),
            "message": str(result.message),
            "baseline": {
                "setpoints": dict(zip(CONTROLLABLE_COLS, x0.tolist())),
                "specific_energy_kwh_t": baseline_energy,
                "throughput_tph": baseline_throughput,
                "wear_hazard_ratio": baseline_hazard,
            },
            "recommended": {
                "setpoints": dict(zip(CONTROLLABLE_COLS, optimal_setpoints.tolist())),
                "specific_energy_kwh_t": opt_energy,
                "throughput_tph": opt_throughput,
                "wear_hazard_ratio": opt_hazard,
            },
            "energy_savings_pct": float((baseline_energy - opt_energy) / baseline_energy * 100),
            "wear_penalty_weight": wear_penalty_weight,
            "min_throughput_constraint_tph": min_throughput_tph,
        }


def _sample_context_from_processed(df_path: Path, row_index: int = -1) -> dict:
    """Extrae un contexto de ejemplo (features no controlables + valores
    actuales de setpoints) desde el dataset procesado, para demostrar el
    optimizador end-to-end sin necesitar un servicio en linea."""
    df = pd.read_parquet(df_path)
    row = df.iloc[row_index]
    context = row[FEATURE_COLS].to_dict()
    context["specific_energy_std_recent"] = float(
        df["specific_energy_kwh_t"].iloc[max(0, row_index - 72):row_index].std()
    )
    return context


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    context = _sample_context_from_processed(base / "data" / "processed" / "sag_mill_operation_with_kf.parquet")

    optimizer = PrescriptiveOptimizer()
    result = optimizer.optimize_setpoints(context, min_throughput_tph=context["fresh_feed_tph"] * 0.97)

    out_dir = base / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "prescriptive_optimization_example.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
