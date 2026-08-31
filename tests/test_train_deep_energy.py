import numpy as np
import pandas as pd
import torch

from src.data.simulation import simulate_sag_mill_operation
from src.features.kalman_filter import add_kalman_hardness_estimate
from src.features.preprocessing import run_preprocessing_pipeline
from src.models.train_deep_energy import (
    ACTIVATIONS,
    EnergyThroughputMLP,
    compare_against_tree_benchmark,
    evaluate_mlp,
    run_activation_comparison,
    train_mlp,
    weighted_huber_loss,
)


def _prepared_dataset(n_hours=400, seed=9):
    raw = simulate_sag_mill_operation(n_hours=n_hours, seed=seed)
    processed = run_preprocessing_pipeline(raw)
    processed["lab_assay_wi"] = raw["lab_assay_wi"].to_numpy()
    return add_kalman_hardness_estimate(processed)


def test_mlp_forward_pass_shape():
    model = EnergyThroughputMLP(n_features=10, n_targets=2, activation="relu")
    x = torch.randn(5, 10)
    out = model(x)
    assert out.shape == (5, 2)


def test_all_activation_layers_build_and_run():
    for activation in ACTIVATIONS:
        model = EnergyThroughputMLP(n_features=6, n_targets=2, activation=activation)
        out = model(torch.randn(3, 6))
        assert out.shape == (3, 2)


def test_weighted_huber_loss_is_nonnegative_and_zero_when_exact():
    target_std = torch.tensor([1.0, 2.0])
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    loss_zero = weighted_huber_loss(pred, pred, target_std)
    assert loss_zero.item() == 0.0

    other = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    loss_positive = weighted_huber_loss(pred, other, target_std)
    assert loss_positive.item() > 0.0


def test_train_mlp_reduces_loss_over_epochs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8)).astype(np.float32)
    true_w = rng.normal(size=(8, 2)).astype(np.float32)
    y = X @ true_w + rng.normal(scale=0.05, size=(200, 2)).astype(np.float32)

    model, history = train_mlp(X, y, activation="relu", n_epochs=20, batch_size=32)
    assert len(history) == 20
    assert history[-1] < history[0]

    metrics, preds = evaluate_mlp(model, X, y)
    assert preds.shape == (200, 2)
    for target_metrics in metrics.values():
        if isinstance(target_metrics, dict):
            assert target_metrics["rmse"] >= 0


def test_run_activation_comparison_end_to_end(tmp_path):
    df = _prepared_dataset(n_hours=500)
    report = run_activation_comparison(df, tmp_path / "models", n_epochs=8)

    assert report["best_activation"] in ACTIVATIONS
    assert set(report["metrics"].keys()) == set(ACTIVATIONS)
    assert (tmp_path / "models" / "deep_energy_mlp.pt").exists()
    assert (tmp_path / "models" / "deep_energy_scaler.joblib").exists()
    assert (tmp_path / "reports" / "deep_energy_loss_curves.csv").exists()
    assert (tmp_path / "reports" / "deep_learning_activation_comparison.json").exists()

    for target_metrics in report["metrics"][report["best_activation"]].values():
        if isinstance(target_metrics, dict):
            assert target_metrics["rmse"] >= 0


def test_compare_against_tree_benchmark(tmp_path):
    import json

    deep_report = {
        "best_activation": "gelu",
        "metrics": {"gelu": {"specific_energy_kwh_t": {"rmse": 1.0, "mae": 0.8, "r2": 0.5},
                              "throughput_tph": {"rmse": 5.0, "mae": 4.0, "r2": 0.6}}},
    }
    tree_report = {
        "best_model": "lightgbm_multioutput_tuned",
        "metrics": {"lightgbm_multioutput_tuned": {
            "specific_energy_kwh_t": {"rmse": 0.6, "mae": 0.4, "r2": 0.8},
            "throughput_tph": {"rmse": 3.0, "mae": 2.0, "r2": 0.79},
        }},
    }
    tree_json_path = tmp_path / "multioutput_model_comparison.json"
    with open(tree_json_path, "w") as f:
        json.dump(tree_report, f)

    comparison = compare_against_tree_benchmark(deep_report, tree_json_path)
    assert comparison["tree_model"] == "lightgbm_multioutput_tuned"
    assert comparison["deep_model"] == "pytorch_mlp_gelu"
    assert comparison["tree_metrics"]["specific_energy_kwh_t"]["rmse"] == 0.6
    assert comparison["deep_metrics"]["specific_energy_kwh_t"]["rmse"] == 1.0
