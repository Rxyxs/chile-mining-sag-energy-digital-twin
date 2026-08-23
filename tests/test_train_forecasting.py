from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.data.simulation import simulate_sag_mill_operation
from src.features.kalman_filter import add_kalman_hardness_estimate
from src.features.preprocessing import run_preprocessing_pipeline
from src.models.train_forecasting import (
    HORIZON_HOURS,
    build_supervised_forecast_dataset,
    evaluate_holt_winters_forecast,
    evaluate_lightgbm_forecast,
    run_forecasting_pipeline,
)


def _prepared_dataset(n_hours=24 * 40, seed=13):
    raw = simulate_sag_mill_operation(n_hours=n_hours, seed=seed)
    processed = run_preprocessing_pipeline(raw)
    processed["lab_assay_wi"] = raw["lab_assay_wi"].to_numpy()
    return add_kalman_hardness_estimate(processed)


def test_supervised_dataset_target_is_shifted_forward():
    df = _prepared_dataset()
    feats = build_supervised_forecast_dataset(df)
    assert "target" in feats.columns
    assert feats.isna().sum().sum() == 0
    assert len(feats) < len(df)


def test_lightgbm_and_holt_winters_folds_produce_metrics():
    df = _prepared_dataset()
    feats = build_supervised_forecast_dataset(df)
    splits = list(TimeSeriesSplit(n_splits=3).split(feats))

    lgbm_folds, lgbm_preview = evaluate_lightgbm_forecast(feats, splits)
    hw_folds, hw_preview = evaluate_holt_winters_forecast(df, feats, splits)

    assert len(lgbm_folds) == 3
    assert len(hw_folds) == 3
    for fold in lgbm_folds + hw_folds:
        assert fold["rmse"] >= 0
        assert fold["mape_pct"] >= 0
    assert len(lgbm_preview) > 0
    assert len(hw_preview) > 0


def test_run_forecasting_pipeline_writes_report(tmp_path):
    df = _prepared_dataset(n_hours=24 * 50)
    report = run_forecasting_pipeline(df, tmp_path, n_splits=3)

    assert (tmp_path / "forecasting_model_comparison.json").exists()
    assert (tmp_path / "forecast_last_fold_preview.csv").exists()
    assert report["horizon_hours"] == HORIZON_HOURS
    assert "rmse_mean" in report["lightgbm"]["summary"]
    assert "rmse_mean" in report["holt_winters"]["summary"]
