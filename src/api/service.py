"""Servicio FastAPI que expone los tres componentes del gemelo digital:

- ``POST /predict``           -- energia especifica / throughput predichos a
  partir de una lectura operacional cruda, con contribucion SHAP punto a
  punto de cada feature (explicabilidad, no solo el numero final).
- ``POST /survival/predict``  -- vida util remanente (RUL) y hazard ratio del
  modelo CoxPH para un ciclo de operacion dado.
- ``POST /optimize``          -- setpoints prescriptivos recomendados
  (`prescriptive_optimizer.PrescriptiveOptimizer`) para minimizar energia sin
  sacrificar throughput ni acelerar el desgaste mecanico.

Arranca cargando una sola vez el modelo de regresion, el modelo CoxPH, el
optimizador prescriptivo y el background set de SHAP (evita reconstruir
explainers en cada request).

Uso local:
    uvicorn src.api.service:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prescriptive_optimizer import PrescriptiveOptimizer
from src.api.explain import build_target_explainers, explain_point
from src.features.kalman_filter import add_kalman_hardness_estimate
from src.features.preprocessing import run_preprocessing_pipeline
from src.models.train_multioutput import FEATURE_COLS, TARGET_COLS
from src.models.train_survival import COVARIATE_COLS, predict_remaining_life

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "outputs" / "models"
SHAP_BACKGROUND_SIZE = 100

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="SAG Energy Digital Twin API",
    description="Prediccion, explicabilidad SHAP, supervivencia CoxPH y optimizacion prescriptiva de setpoints para un molino SAG.",
    version="1.0.0",
    lifespan=lifespan,
)


class OperationalReading(BaseModel):
    timestamp: str = Field(..., description="ISO 8601, ej. 2025-06-01T14:00:00")
    fresh_feed_tph: float
    mill_load_pct: float
    ball_charge_pct: float
    water_addition_m3h: float
    f80_um: float
    p80_um: float
    hardness_proxy_wi: float
    lab_assay_wi: float | None = None
    mill_power_mw: float
    specific_energy_kwh_t: float
    throughput_tph: float


class SurvivalContext(BaseModel):
    load_deviation_from_optimum: float
    ball_charge_deviation: float
    specific_energy_std: float
    hardness_proxy_wi_mean: float


class OptimizeRequest(BaseModel):
    context: dict = Field(..., description="Features del ultimo punto procesado (ver /predict) mas 'specific_energy_std_recent'")
    min_throughput_tph: float | None = None
    wear_penalty_weight: float = 0.15


class _State:
    predictor_model = None
    feature_cols: list[str] | None = None
    explainers: list | None = None
    survival_model = None
    prescriptive_optimizer: PrescriptiveOptimizer | None = None


state = _State()


def load_artifacts() -> None:
    state.predictor_model = joblib.load(MODEL_DIR / "best_multioutput_model.joblib")
    state.feature_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")

    processed = pd.read_parquet(BASE_DIR / "data" / "processed" / "sag_mill_operation_with_kf.parquet")
    background = processed[state.feature_cols].sample(
        min(SHAP_BACKGROUND_SIZE, len(processed)), random_state=42
    )
    state.explainers = build_target_explainers(state.predictor_model, background)

    coxph_path = MODEL_DIR / "coxph_survival_model.joblib"
    state.survival_model = joblib.load(coxph_path) if coxph_path.exists() else None

    state.prescriptive_optimizer = PrescriptiveOptimizer(model_dir=MODEL_DIR)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "regression_model_loaded": state.predictor_model is not None,
        "survival_model_loaded": state.survival_model is not None,
    }


@app.post("/predict")
def predict(reading: OperationalReading) -> dict:
    raw_df = pd.DataFrame([reading.model_dump()])
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])

    processed = run_preprocessing_pipeline(raw_df)
    # fit_params=False: una sola lectura no tiene historia suficiente para
    # re-estimar Q/R del filtro (ver HardnessKalmanFilter.from_data), asi que
    # la inferencia en linea usa los parametros de fabrica pre-calibrados,
    # igual que en produccion real (Q/R se calibran offline, no por request).
    with_kf = add_kalman_hardness_estimate(processed, fit_params=False)

    missing = set(FEATURE_COLS) - set(with_kf.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"Faltan columnas derivadas: {sorted(missing)}")

    X = with_kf[FEATURE_COLS]
    preds = state.predictor_model.predict(X)[0]
    prediction = {target: float(value) for target, value in zip(TARGET_COLS, preds)}

    shap_contributions = explain_point(state.explainers, TARGET_COLS, X)

    return {
        "prediction": prediction,
        "shap_explanation": shap_contributions,
        "features_used": X.iloc[0].to_dict(),
    }


@app.post("/survival/predict")
def survival_predict(context: SurvivalContext) -> dict:
    if state.survival_model is None:
        raise HTTPException(status_code=503, detail="Modelo de supervivencia no disponible: ejecuta src/models/train_survival.py primero.")

    row = pd.DataFrame([context.model_dump()])[COVARIATE_COLS]
    return predict_remaining_life(state.survival_model, row)


@app.post("/optimize")
def optimize(request: OptimizeRequest) -> dict:
    if state.prescriptive_optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizador prescriptivo no disponible.")

    try:
        return state.prescriptive_optimizer.optimize_setpoints(
            context=request.context,
            min_throughput_tph=request.min_throughput_tph,
            wear_penalty_weight=request.wear_penalty_weight,
        )
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Falta feature de contexto requerida: {exc}")
