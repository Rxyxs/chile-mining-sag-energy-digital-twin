"""Entrypoint de inferencia: carga artefactos serializados y predice
energia especifica (kWh/t) y throughput (t/h) sobre datos operacionales
nuevos, replicando exactamente el pipeline de features de entrenamiento.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.features.kalman_filter import add_kalman_hardness_estimate
from src.features.preprocessing import run_preprocessing_pipeline
from src.models.train_multioutput import TARGET_COLS

MODEL_DIR = Path(__file__).resolve().parents[2] / "outputs" / "models"


class SagMillPredictor:
    """Envuelve el modelo entrenado y el pipeline de features completo,
    para que la inferencia en produccion reciba datos crudos (los mismos
    campos que entrega el SCADA/simulador) y no features pre-calculadas."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model = joblib.load(model_dir / "best_multioutput_model.joblib")
        self.feature_cols = joblib.load(model_dir / "feature_columns.joblib")

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        processed = run_preprocessing_pipeline(raw_df)
        with_kf = add_kalman_hardness_estimate(processed)

        missing = set(self.feature_cols) - set(with_kf.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas para inferencia: {sorted(missing)}")

        X = with_kf[self.feature_cols]
        preds = self.model.predict(X)

        result = with_kf[["timestamp"]].copy()
        for i, target in enumerate(TARGET_COLS):
            result[f"{target}_pred"] = preds[:, i]
        return result


def main():
    parser = argparse.ArgumentParser(description="Inferencia de energia/throughput SAG")
    parser.add_argument("input_path", type=str, help="Parquet o CSV con datos operacionales crudos")
    parser.add_argument("--output", type=str, default=None, help="Ruta de salida (default: stdout)")
    args = parser.parse_args()

    path = Path(args.input_path)
    raw_df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, parse_dates=["timestamp"])

    predictor = SagMillPredictor()
    predictions = predictor.predict(raw_df)

    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"Predicciones guardadas -> {args.output}")
    else:
        print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
