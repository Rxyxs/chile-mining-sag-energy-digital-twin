"""Tercer enfoque de modelado: MLP en PyTorch para energia especifica
(kWh/t) y throughput (t/h), sobre el mismo dataset y particion cronologica
que ``train_multioutput.py``.

Este modulo no busca reemplazar el ganador basado en arboles (LightGBM
tuneado, ver ``train_multioutput.py``) -- lo usa como benchmark. El objetivo
es (a) verificar si una red densa con loss robusto (Huber) captura la misma
relacion no lineal con una arquitectura mas liviana en inferencia, y (b)
medir el efecto de la funcion de activacion (ReLU vs GELU vs Swish/SiLU) en
un problema tabular pequeno-mediano, donde la literatura no da una respuesta
unica de antemano.

Loss custom: Huber (SmoothL1) ponderado por target, normalizado por el
desvio estandar de cada objetivo en train -- sin esto, throughput (t/h,
escala ~100) domina el gradiente sobre energia especifica (kWh/t, escala
~15) y el modelo ignora el segundo objetivo.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.train_multioutput import FEATURE_COLS, TARGET_COLS, chronological_split

RANDOM_STATE = 42
HIDDEN_DIMS = (64, 32)
ACTIVATIONS = ("relu", "gelu", "swish")
N_EPOCHS = 120
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
HUBER_DELTA = 1.0


def _activation_layer(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return nn.SiLU()  # Swish == x * sigmoid(x) == SiLU
    raise ValueError(f"Activacion desconocida: {name}")


class EnergyThroughputMLP(nn.Module):
    """MLP feed-forward con activacion configurable, para regresion
    multi-salida (energia especifica + throughput) a partir de features
    operacionales del molino SAG."""

    def __init__(self, n_features: int, n_targets: int = 2,
                 hidden_dims: tuple[int, ...] = HIDDEN_DIMS, activation: str = "relu"):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(_activation_layer(activation))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor,
                         target_std: torch.Tensor, delta: float = HUBER_DELTA) -> torch.Tensor:
    """Huber loss aplicado sobre residuos normalizados por el desvio
    estandar de cada target en train, para que ambos objetivos contribuyan
    en escala comparable al gradiente."""
    resid = (pred - target) / target_std
    huber = torch.nn.functional.smooth_l1_loss(resid, torch.zeros_like(resid), beta=delta, reduction="mean")
    return huber


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, activation: str,
              n_epochs: int = N_EPOCHS, batch_size: int = BATCH_SIZE,
              lr: float = LEARNING_RATE, seed: int = RANDOM_STATE) -> tuple[EnergyThroughputMLP, list[float]]:
    torch.manual_seed(seed)
    model = EnergyThroughputMLP(n_features=X_train.shape[1], n_targets=y_train.shape[1], activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    target_std = _to_tensor(y_train.std(axis=0)).clamp_min(1e-6)
    dataset = TensorDataset(_to_tensor(X_train), _to_tensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))

    history = []
    model.train()
    for _ in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = weighted_huber_loss(pred, yb, target_std)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        history.append(epoch_loss / max(n_batches, 1))
    return model, history


def evaluate_mlp(model: EnergyThroughputMLP, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        preds = model(_to_tensor(X_test)).numpy()
    metrics = {}
    for i, target in enumerate(TARGET_COLS):
        y_true = y_test[:, i]
        y_pred = preds[:, i]
        metrics[target] = {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    metrics["overall_r2_uniform_average"] = float(r2_score(y_test, preds, multioutput="uniform_average"))
    return metrics, preds


def run_activation_comparison(df: pd.DataFrame, out_dir: Path,
                               n_epochs: int = N_EPOCHS) -> dict:
    """Entrena un MLP identico (misma arquitectura, mismo loss, misma
    semilla) para cada activacion y compara metricas de test, para aislar
    el efecto de la funcion de activacion del resto del pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = chronological_split(df)
    X_train_raw = train_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    X_test_raw = test_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_train = train_df[TARGET_COLS].to_numpy(dtype=np.float64)
    y_test = test_df[TARGET_COLS].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    results = {}
    histories = {}
    fitted = {}
    for activation in ACTIVATIONS:
        model, history = train_mlp(X_train, y_train, activation, n_epochs=n_epochs)
        metrics, _ = evaluate_mlp(model, X_test, y_test)
        results[activation] = metrics
        histories[activation] = history
        fitted[activation] = model

    best_activation = max(results, key=lambda k: results[k]["overall_r2_uniform_average"])
    best_model = fitted[best_activation]

    torch.save(best_model.state_dict(), out_dir / "deep_energy_mlp.pt")
    joblib.dump(scaler, out_dir / "deep_energy_scaler.joblib")
    joblib.dump({"activation": best_activation, "hidden_dims": HIDDEN_DIMS}, out_dir / "deep_energy_mlp_config.joblib")

    _, best_preds = evaluate_mlp(best_model, X_test, y_test)
    residuals_df = test_df[["timestamp"] + TARGET_COLS].copy()
    for i, target in enumerate(TARGET_COLS):
        residuals_df[f"{target}_pred"] = best_preds[:, i]
        residuals_df[f"{target}_residual"] = residuals_df[target] - best_preds[:, i]
    residuals_df.to_csv(reports_dir / "deep_energy_test_residuals.csv", index=False)

    loss_curve_df = pd.DataFrame({activation: histories[activation] for activation in ACTIVATIONS})
    loss_curve_df.insert(0, "epoch", range(1, n_epochs + 1))
    loss_curve_df.to_csv(reports_dir / "deep_energy_loss_curves.csv", index=False)

    report = {
        "best_activation": best_activation,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_epochs": n_epochs,
        "loss": "weighted_huber (per-target std normalized)",
        "metrics": results,
    }
    with open(reports_dir / "deep_learning_activation_comparison.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


def compare_against_tree_benchmark(deep_report: dict, multioutput_comparison_json: Path) -> dict:
    """Benchmark del MLP (mejor activacion) contra el mejor modelo de
    arboles ya entrenado en train_multioutput.py -- el MLP no reemplaza al
    ganador, sirve como segundo pilar del ensamble de enfoques y como
    verificacion cruzada."""
    with open(multioutput_comparison_json) as f:
        tree_report = json.load(f)
    tree_best_name = tree_report["best_model"]
    tree_metrics = tree_report["metrics"][tree_best_name]
    deep_metrics = deep_report["metrics"][deep_report["best_activation"]]

    comparison = {
        "tree_model": tree_best_name,
        "tree_metrics": tree_metrics,
        "deep_model": f"pytorch_mlp_{deep_report['best_activation']}",
        "deep_metrics": deep_metrics,
    }
    return comparison


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    df = pd.read_parquet(base / "data" / "processed" / "sag_mill_operation_with_kf.parquet")
    out_dir = base / "outputs" / "models"
    report = run_activation_comparison(df, out_dir)
    print(json.dumps(report, indent=2))

    mo_json = base / "outputs" / "reports" / "multioutput_model_comparison.json"
    if mo_json.exists():
        comparison = compare_against_tree_benchmark(report, mo_json)
        with open(base / "outputs" / "reports" / "deep_vs_tree_benchmark.json", "w") as f:
            json.dump(comparison, f, indent=2)
        print(json.dumps(comparison, indent=2))
