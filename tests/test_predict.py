from pathlib import Path

from src.data.simulation import simulate_sag_mill_operation
from src.features.kalman_filter import add_kalman_hardness_estimate
from src.features.preprocessing import run_preprocessing_pipeline
from src.inference.predict import SagMillPredictor
from src.models.train_multioutput import TARGET_COLS, run_training_pipeline


def test_predictor_round_trip(tmp_path):
    train_raw = simulate_sag_mill_operation(n_hours=600, seed=21)
    train_processed = run_preprocessing_pipeline(train_raw)
    train_processed["lab_assay_wi"] = train_raw["lab_assay_wi"].to_numpy()
    train_with_kf = add_kalman_hardness_estimate(train_processed)

    model_dir = tmp_path / "models"
    run_training_pipeline(train_with_kf, model_dir, tune=False)

    predictor = SagMillPredictor(model_dir=model_dir)

    new_raw = simulate_sag_mill_operation(n_hours=48, seed=22)
    predictions = predictor.predict(new_raw)

    assert len(predictions) == 48
    for target in TARGET_COLS:
        assert f"{target}_pred" in predictions.columns
    assert predictions[f"{TARGET_COLS[0]}_pred"].notna().all()
