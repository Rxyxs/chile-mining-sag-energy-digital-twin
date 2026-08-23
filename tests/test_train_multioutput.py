import pandas as pd

from src.data.simulation import simulate_sag_mill_operation
from src.features.kalman_filter import add_kalman_hardness_estimate
from src.features.preprocessing import run_preprocessing_pipeline
from src.models.train_multioutput import (
    FEATURE_COLS,
    TARGET_COLS,
    build_candidate_models,
    chronological_split,
    evaluate_model,
    run_training_pipeline,
)


def _prepared_dataset(n_hours=400, seed=9):
    raw = simulate_sag_mill_operation(n_hours=n_hours, seed=seed)
    processed = run_preprocessing_pipeline(raw)
    processed["lab_assay_wi"] = raw["lab_assay_wi"].to_numpy()
    return add_kalman_hardness_estimate(processed)


def test_chronological_split_preserves_order_and_no_leakage():
    df = _prepared_dataset()
    train, test = chronological_split(df, test_fraction=0.2)
    assert train["timestamp"].max() < test["timestamp"].min()
    assert len(train) + len(test) == len(df)


def test_all_feature_columns_present_after_pipeline():
    df = _prepared_dataset()
    missing = set(FEATURE_COLS) - set(df.columns)
    assert not missing, f"Faltan columnas: {missing}"


def test_linear_regression_baseline_trains_and_predicts():
    df = _prepared_dataset()
    train, test = chronological_split(df)
    model = build_candidate_models()["linear_regression"]
    model.fit(train[FEATURE_COLS], train[TARGET_COLS])
    metrics = evaluate_model(model, test[FEATURE_COLS], test[TARGET_COLS])
    for target in TARGET_COLS:
        assert metrics[target]["rmse"] > 0


def test_run_training_pipeline_end_to_end(tmp_path):
    df = _prepared_dataset(n_hours=600)
    report = run_training_pipeline(df, tmp_path / "models", tune=False)

    assert report["best_model"] in report["metrics"]
    assert (tmp_path / "models" / "best_multioutput_model.joblib").exists()
    assert (tmp_path / "reports" / "feature_importance.csv").exists()
    assert (tmp_path / "reports" / "test_residuals.csv").exists()

    for target in TARGET_COLS:
        assert report["metrics"][report["best_model"]][target]["r2"] > -1.0
