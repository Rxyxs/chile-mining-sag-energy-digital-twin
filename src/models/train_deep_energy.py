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

Entrenamiento: el tramo final del train (cronologico) se separa como
validacion. Cada red entrena al menos ``MIN_EPOCHS`` epocas y como maximo
``MAX_EPOCHS``; despues del minimo, early stopping corta si la loss de
validacion no mejora en ``PATIENCE`` epocas y se restauran los pesos de la
mejor epoca. La tasa de aprendizaje baja a la mitad cuando la validacion se
estanca. La activacion ganadora se elige en validacion: el test se mira una
sola vez, al final.

Checkpoints: cada epoca se guarda el estado completo (pesos, optimizador,
scheduler, historia, mejor modelo) en ``outputs/models/checkpoints/``. Si el
proceso se interrumpe, la siguiente corrida retoma desde la ultima epoca;
el checkpoint lleva una huella de los datos y la configuracion, asi que un
checkpoint de otro dataset no se reutiliza.
"""

from __future__ import annotations

import copy
import hashlib
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
MIN_EPOCHS = 100
MAX_EPOCHS = 400
PATIENCE = 25
VAL_FRACTION = 0.15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LR_FACTOR = 0.5
LR_PATIENCE = 10
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


def _fingerprint(*arrays: np.ndarray, **config) -> str:
    """Huella de los datos y la configuracion: un checkpoint solo se retoma si coincide."""
    h = hashlib.sha256()
    for a in arrays:
        if a is not None:
            h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    h.update(json.dumps(config, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, activation: str,
              X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
              n_epochs: int = MAX_EPOCHS, min_epochs: int = MIN_EPOCHS, patience: int = PATIENCE,
              batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE, seed: int = RANDOM_STATE,
              checkpoint_path: Path | None = None) -> tuple[EnergyThroughputMLP, dict]:
    """Entrena el MLP y devuelve (modelo, historia).

    Con datos de validacion: early stopping sobre la loss de validacion, que solo puede cortar despues de
    ``min_epochs``, y restauracion de los pesos de la mejor epoca. Sin validacion entrena ``n_epochs``
    completas. ``historia`` = {"train": [...], "val": [...], "best_epoch": int, "epochs_run": int,
    "stopped_early": bool, "resumed_from": int}.
    """
    torch.manual_seed(seed)
    model = EnergyThroughputMLP(n_features=X_train.shape[1], n_targets=y_train.shape[1], activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    has_val = X_val is not None and y_val is not None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=LR_FACTOR, patience=LR_PATIENCE) \
        if has_val else None

    target_std = _to_tensor(y_train.std(axis=0)).clamp_min(1e-6)
    dataset = TensorDataset(_to_tensor(X_train), _to_tensor(y_train))
    generator = torch.Generator()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    Xv, yv = (_to_tensor(X_val), _to_tensor(y_val)) if has_val else (None, None)

    fp = _fingerprint(X_train, y_train, X_val, y_val, activation=activation, n_epochs=n_epochs,
                      min_epochs=min_epochs, patience=patience, batch_size=batch_size, lr=lr, seed=seed,
                      hidden=HIDDEN_DIMS)
    state = {"epoch": 0, "train": [], "val": [], "best_val": float("inf"), "best_epoch": 0,
             "best_state": None, "bad_epochs": 0, "done": False}
    resumed_from = 0
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt.get("fingerprint") == fp:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            state = ckpt["state"]
            resumed_from = state["epoch"]

    def save_checkpoint():
        if checkpoint_path is None:
            return
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(checkpoint_path).with_suffix(".tmp")
        torch.save({"fingerprint": fp, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None, "state": state}, tmp)
        tmp.replace(checkpoint_path)  # escritura atomica: un corte no deja un checkpoint a medias

    while not state["done"] and state["epoch"] < n_epochs:
        epoch = state["epoch"]
        generator.manual_seed(seed + epoch)  # orden de batches reproducible tambien al retomar
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = weighted_huber_loss(model(xb), yb, target_std)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        state["train"].append(epoch_loss / max(n_batches, 1))
        state["epoch"] = epoch + 1

        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = float(weighted_huber_loss(model(Xv), yv, target_std).item())
            state["val"].append(val_loss)
            scheduler.step(val_loss)
            if val_loss < state["best_val"] - 1e-6:
                state.update(best_val=val_loss, best_epoch=state["epoch"], bad_epochs=0,
                             best_state=copy.deepcopy(model.state_dict()))
            else:
                state["bad_epochs"] += 1
            if state["epoch"] >= min_epochs and state["bad_epochs"] >= patience:
                state["done"] = True
        save_checkpoint()

    if has_val and state["best_state"] is not None:
        model.load_state_dict(state["best_state"])
    stopped_early = has_val and state["epoch"] < n_epochs
    state["done"] = True
    save_checkpoint()
    history = {"train": state["train"], "val": state["val"],
               "best_epoch": state["best_epoch"] if has_val else state["epoch"],
               "epochs_run": state["epoch"], "stopped_early": bool(stopped_early), "resumed_from": resumed_from}
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


def run_activation_comparison(df: pd.DataFrame, out_dir: Path, n_epochs: int = MAX_EPOCHS,
                               min_epochs: int = MIN_EPOCHS, patience: int = PATIENCE) -> dict:
    """Entrena un MLP identico (misma arquitectura, mismo loss, misma
    semilla) para cada activacion, elige la mejor en validacion y reporta
    test una sola vez, para aislar el efecto de la funcion de activacion."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"

    train_df, test_df = chronological_split(df)
    n_val = max(1, int(len(train_df) * VAL_FRACTION))
    fit_df, val_df = train_df.iloc[:-n_val], train_df.iloc[-n_val:]

    def xy(frame):
        return frame[FEATURE_COLS].to_numpy(dtype=np.float64), frame[TARGET_COLS].to_numpy(dtype=np.float64)

    (X_fit_raw, y_fit), (X_val_raw, y_val), (X_test_raw, y_test) = xy(fit_df), xy(val_df), xy(test_df)
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    results, val_results, histories, fitted = {}, {}, {}, {}
    for activation in ACTIVATIONS:
        model, history = train_mlp(X_fit, y_fit, activation, X_val=X_val, y_val=y_val, n_epochs=n_epochs,
                                   min_epochs=min_epochs, patience=patience,
                                   checkpoint_path=ckpt_dir / f"deep_energy_{activation}.ckpt")
        val_results[activation], _ = evaluate_mlp(model, X_val, y_val)
        histories[activation] = history
        fitted[activation] = model

    best_activation = max(val_results, key=lambda k: val_results[k]["overall_r2_uniform_average"])
    for activation in ACTIVATIONS:  # test: una sola evaluacion por modelo ya elegido
        results[activation], _ = evaluate_mlp(fitted[activation], X_test, y_test)
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

    # Curvas: una columna de validacion por activacion (nombre = activacion) y otra de train; las
    # activaciones paran en epocas distintas, asi que las columnas mas cortas quedan con NaN al final.
    longest = max(h["epochs_run"] for h in histories.values())
    curves = {"epoch": range(1, longest + 1)}
    for activation, h in histories.items():
        pad = [np.nan] * (longest - h["epochs_run"])
        curves[activation] = h["val"] + pad
        curves[f"{activation}_train"] = h["train"] + pad
    pd.DataFrame(curves).to_csv(reports_dir / "deep_energy_loss_curves.csv", index=False)

    report = {
        "best_activation": best_activation,
        "selected_on": "validation",
        "n_train": len(fit_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "max_epochs": n_epochs,
        "min_epochs": min_epochs,
        "patience": patience,
        "training": {a: {k: histories[a][k] for k in ("epochs_run", "best_epoch", "stopped_early", "resumed_from")}
                     | {"best_val_loss": float(min(histories[a]["val"]))} for a in ACTIVATIONS},
        "loss": "weighted_huber (per-target std normalized)",
        "val_metrics": val_results,
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
