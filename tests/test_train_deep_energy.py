import numpy as np
import pytest
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


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8)).astype(np.float32)
    y = X @ rng.normal(size=(8, 2)).astype(np.float32) + rng.normal(scale=0.05, size=(n, 2)).astype(np.float32)
    return X, y


def test_train_mlp_reduces_loss_over_epochs():
    X, y = _toy()
    model, history = train_mlp(X, y, activation="relu", n_epochs=20, batch_size=32)
    assert history["epochs_run"] == 20 and len(history["train"]) == 20
    assert history["train"][-1] < history["train"][0]

    metrics, preds = evaluate_mlp(model, X, y)
    assert preds.shape == (200, 2)
    for target_metrics in metrics.values():
        if isinstance(target_metrics, dict):
            assert target_metrics["rmse"] >= 0


def test_early_stopping_respects_minimum_epochs_and_restores_best():
    X, y = _toy()
    Xv, yv = _toy(80, seed=1)
    model, h = train_mlp(X, y, "relu", X_val=Xv, y_val=yv, n_epochs=60, min_epochs=30, patience=3)
    assert h["epochs_run"] >= 30
    assert h["best_epoch"] == int(np.argmin(h["val"])) + 1
    best_val = min(h["val"])
    target_std = torch.as_tensor(y.std(axis=0), dtype=torch.float32)
    with torch.no_grad():
        restored = weighted_huber_loss(model(torch.as_tensor(Xv)), torch.as_tensor(yv), target_std).item()
    assert restored == pytest.approx(best_val, rel=1e-5)


def test_training_resumes_from_checkpoint_after_interruption(tmp_path, monkeypatch):
    X, y = _toy()
    Xv, yv = _toy(80, seed=1)
    kwargs = dict(X_val=Xv, y_val=yv, n_epochs=12, min_epochs=12, patience=50, batch_size=32)
    reference, ref_h = train_mlp(X, y, "gelu", **kwargs)

    ckpt = tmp_path / "gelu.ckpt"
    real_step = torch.optim.Adam.step
    calls = {"n": 0}

    def flaky_step(self, *a, **k):
        calls["n"] += 1
        if calls["n"] > 7 * 4:  # 200 filas / 32 = 7 batches: corta durante la epoca 5
            raise KeyboardInterrupt
        return real_step(self, *a, **k)

    monkeypatch.setattr(torch.optim.Adam, "step", flaky_step)
    with pytest.raises(KeyboardInterrupt):
        train_mlp(X, y, "gelu", checkpoint_path=ckpt, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", real_step)
    assert ckpt.exists()

    resumed, h = train_mlp(X, y, "gelu", checkpoint_path=ckpt, **kwargs)
    assert h["resumed_from"] == 4
    assert h["val"] == pytest.approx(ref_h["val"], rel=1e-5)
    for a, b in zip(reference.state_dict().values(), resumed.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6)


def test_checkpoint_from_other_data_is_not_reused(tmp_path):
    X, y = _toy()
    ckpt = tmp_path / "relu.ckpt"
    train_mlp(X, y, "relu", n_epochs=3, checkpoint_path=ckpt)
    X2, y2 = _toy(seed=5)
    _, h = train_mlp(X2, y2, "relu", n_epochs=3, checkpoint_path=ckpt)
    assert h["resumed_from"] == 0


def test_run_activation_comparison_end_to_end(tmp_path):
    df = _prepared_dataset(n_hours=500)
    report = run_activation_comparison(df, tmp_path / "models", n_epochs=8, min_epochs=4, patience=2)

    assert report["best_activation"] in ACTIVATIONS
    assert report["selected_on"] == "validation"
    assert set(report["metrics"].keys()) == set(ACTIVATIONS)
    best = report["best_activation"]
    assert all(report["val_metrics"][best]["overall_r2_uniform_average"]
               >= report["val_metrics"][a]["overall_r2_uniform_average"] for a in ACTIVATIONS)
    assert all(4 <= report["training"][a]["epochs_run"] <= 8 for a in ACTIVATIONS)
    assert (tmp_path / "models" / "deep_energy_mlp.pt").exists()
    assert (tmp_path / "models" / "checkpoints" / f"deep_energy_{best}.ckpt").exists()
    assert (tmp_path / "models" / "deep_energy_scaler.joblib").exists()
    assert (tmp_path / "reports" / "deep_energy_loss_curves.csv").exists()
    assert (tmp_path / "reports" / "deep_learning_activation_comparison.json").exists()

    for target_metrics in report["metrics"][best].values():
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
