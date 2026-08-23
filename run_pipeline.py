"""Orquestador end-to-end: simulacion -> preprocesamiento -> Kalman ->
entrenamiento (multi-output + forecasting) -> visualizaciones.

Uso:
    python run_pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.simulation import simulate_sag_mill_operation
from src.features.kalman_filter import add_kalman_hardness_estimate
from src.features.preprocessing import run_preprocessing_pipeline
from src.models.train_forecasting import run_forecasting_pipeline
from src.models.train_multioutput import run_training_pipeline
from src.visualization.plots import generate_all_plots

BASE_DIR = Path(__file__).resolve().parent


def main():
    raw_dir = BASE_DIR / "data" / "raw"
    processed_dir = BASE_DIR / "data" / "processed"
    models_dir = BASE_DIR / "outputs" / "models"
    reports_dir = BASE_DIR / "outputs" / "reports"
    for d in (raw_dir, processed_dir, models_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("[1/6] Simulando operacion del molino SAG (180 dias horarios)...")
    raw = simulate_sag_mill_operation()
    raw.to_parquet(raw_dir / "sag_mill_operation_raw.parquet", index=False)
    print(f"       {len(raw)} registros generados.")

    print("[2/6] Preprocesamiento: outliers, imputacion, feature engineering...")
    processed = run_preprocessing_pipeline(raw)
    processed.to_parquet(processed_dir / "sag_mill_operation_clean.parquet", index=False)

    print("[3/6] Filtro de Kalman: fusion de sensores de dureza...")
    with_kf = add_kalman_hardness_estimate(processed)
    with_kf.to_parquet(processed_dir / "sag_mill_operation_with_kf.parquet", index=False)

    true_wi = raw["ore_hardness_wi_true"].to_numpy()
    proxy_rmse = float(((with_kf["hardness_proxy_wi"] - true_wi) ** 2).mean() ** 0.5)
    kf_rmse = float(((with_kf["wi_hat"] - true_wi) ** 2).mean() ** 0.5)
    print(f"       RMSE proxy crudo: {proxy_rmse:.3f} | RMSE Kalman: {kf_rmse:.3f} "
          f"({(1 - kf_rmse / proxy_rmse) * 100:.1f}% de reduccion de error)")

    print("[4/6] Entrenando modelos multi-output (energia especifica + throughput)...")
    multioutput_report = run_training_pipeline(with_kf, models_dir, tune=True)
    print(f"       Mejor modelo: {multioutput_report['best_model']}")

    print("[5/6] Entrenando y comparando modelos de forecasting (24h ahead)...")
    forecasting_report = run_forecasting_pipeline(with_kf, reports_dir)
    print(f"       LightGBM RMSE: {forecasting_report['lightgbm']['summary']['rmse_mean']:.3f} MW | "
          f"Holt-Winters RMSE: {forecasting_report['holt_winters']['summary']['rmse_mean']:.3f} MW")

    print("[6/6] Generando graficos de resultados...")
    generate_all_plots(BASE_DIR)

    summary = {
        "n_records": len(raw),
        "kalman_filter": {"proxy_rmse": proxy_rmse, "kf_rmse": kf_rmse,
                           "error_reduction_pct": (1 - kf_rmse / proxy_rmse) * 100},
        "multioutput_best_model": multioutput_report["best_model"],
        "multioutput_metrics": multioutput_report["metrics"][multioutput_report["best_model"]],
        "forecasting_lightgbm": forecasting_report["lightgbm"]["summary"],
        "forecasting_holt_winters": forecasting_report["holt_winters"]["summary"],
    }
    with open(reports_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nPipeline completo. Resumen guardado en outputs/reports/pipeline_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
