"""Forecasting de demanda energetica (potencia del molino, MW) a 24h.

Predecir la potencia consumida por el SAG con 24 horas de anticipacion
permite planificar la compra de energia spot / la programacion de
mantenimientos en horas valle -- en Chile el costo de electricidad para
grandes mineras se negocia en gran parte via contratos y mercado spot, por
lo que anticipar la demanda del dia siguiente tiene valor economico directo.

Compara un baseline estadistico clasico (Holt-Winters, suavizamiento
exponencial con estacionalidad diaria) contra un enfoque de ML (LightGBM
sobre features de rezago/rolling), evaluados con validacion walk-forward
(TimeSeriesSplit) -- nunca K-Fold aleatorio, que romperia el orden temporal
y filtraria informacion futura.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing

HORIZON_HOURS = 24
LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
RANDOM_STATE = 42


def build_supervised_forecast_dataset(df: pd.DataFrame, horizon: int = HORIZON_HOURS) -> pd.DataFrame:
    """Construye un dataset supervisado: features conocidas en el origen t,
    target = potencia real en t + horizon."""
    s = df.set_index("timestamp")["mill_power_mw"].asfreq("h")
    feats = pd.DataFrame(index=s.index)

    for lag in LAGS:
        feats[f"lag_{lag}"] = s.shift(lag)
    feats["roll_mean_24"] = s.shift(1).rolling(24).mean()
    feats["roll_std_24"] = s.shift(1).rolling(24).std()
    feats["roll_mean_168"] = s.shift(1).rolling(168).mean()

    target_time = feats.index + pd.Timedelta(hours=horizon)
    feats["target_hour_sin"] = np.sin(2 * np.pi * target_time.hour / 24)
    feats["target_hour_cos"] = np.cos(2 * np.pi * target_time.hour / 24)
    feats["target_dow"] = target_time.dayofweek

    feats["target"] = s.shift(-horizon)
    return feats.dropna()


def _feature_cols(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c != "target"]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred_clipped = np.clip(y_pred, 0.0, None)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_clipped))),
        "mae": float(mean_absolute_error(y_true, y_pred_clipped)),
        "mape_pct": float(mean_absolute_percentage_error(y_true, y_pred_clipped) * 100),
    }


def evaluate_lightgbm_forecast(feats: pd.DataFrame, splits) -> tuple[list[dict], pd.DataFrame]:
    feature_cols = _feature_cols(feats)
    fold_metrics = []
    last_fold_preview = None

    for train_pos, test_pos in splits:
        X_train, y_train = feats.iloc[train_pos][feature_cols], feats.iloc[train_pos]["target"]
        X_test, y_test = feats.iloc[test_pos][feature_cols], feats.iloc[test_pos]["target"]

        model = LGBMRegressor(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            random_state=RANDOM_STATE, verbosity=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        fold_metrics.append(_compute_metrics(y_test.to_numpy(), preds))
        last_fold_preview = pd.DataFrame({
            "target_timestamp": feats.index[test_pos] + pd.Timedelta(hours=HORIZON_HOURS),
            "actual_mw": y_test.to_numpy(),
            "lightgbm_forecast_mw": np.clip(preds, 0.0, None),
        })

    return fold_metrics, last_fold_preview


def evaluate_holt_winters_forecast(df: pd.DataFrame, feats: pd.DataFrame, splits) -> tuple[list[dict], pd.DataFrame]:
    s = df.set_index("timestamp")["mill_power_mw"].asfreq("h")
    fold_metrics = []
    last_fold_preview = None

    for train_pos, test_pos in splits:
        train_end_time = feats.index[train_pos[-1]]
        target_times = feats.index[test_pos] + pd.Timedelta(hours=HORIZON_HOURS)

        history = s.loc[:train_end_time]
        steps_needed = int((target_times.max() - train_end_time) / pd.Timedelta(hours=1))

        hw_model = ExponentialSmoothing(
            history, trend="add", seasonal="add", seasonal_periods=24,
            initialization_method="estimated",
        ).fit()
        forecast_curve = hw_model.forecast(steps_needed)

        preds = forecast_curve.reindex(target_times).to_numpy()
        actual = feats.iloc[test_pos]["target"].to_numpy()

        fold_metrics.append(_compute_metrics(actual, preds))
        last_fold_preview = pd.DataFrame({
            "target_timestamp": target_times,
            "actual_mw": actual,
            "holt_winters_forecast_mw": np.clip(preds, 0.0, None),
        })

    return fold_metrics, last_fold_preview


def summarize_fold_metrics(fold_metrics: list[dict]) -> dict:
    keys = fold_metrics[0].keys()
    return {f"{k}_mean": float(np.mean([m[k] for m in fold_metrics])) for k in keys} | \
           {f"{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in keys}


def run_forecasting_pipeline(df: pd.DataFrame, out_dir: Path, n_splits: int = 5) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = build_supervised_forecast_dataset(df)
    splits = list(TimeSeriesSplit(n_splits=n_splits).split(feats))

    lgbm_folds, lgbm_preview = evaluate_lightgbm_forecast(feats, splits)
    hw_folds, hw_preview = evaluate_holt_winters_forecast(df, feats, splits)

    report = {
        "n_splits": n_splits,
        "horizon_hours": HORIZON_HOURS,
        "lightgbm": {"per_fold": lgbm_folds, "summary": summarize_fold_metrics(lgbm_folds)},
        "holt_winters": {"per_fold": hw_folds, "summary": summarize_fold_metrics(hw_folds)},
    }

    with open(out_dir / "forecasting_model_comparison.json", "w") as f:
        json.dump(report, f, indent=2)

    preview = lgbm_preview.merge(hw_preview, on=["target_timestamp", "actual_mw"], how="inner")
    preview.to_csv(out_dir / "forecast_last_fold_preview.csv", index=False)

    return report


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    df = pd.read_parquet(base / "data" / "processed" / "sag_mill_operation_with_kf.parquet")
    report = run_forecasting_pipeline(df, base / "outputs" / "reports")
    print(json.dumps(report["lightgbm"]["summary"], indent=2))
    print(json.dumps(report["holt_winters"]["summary"], indent=2))
