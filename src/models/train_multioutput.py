"""Modelo multi-output para energia especifica (kWh/t) y throughput (t/h).

Los dos objetivos estan fisicamente acoplados (Potencia = Energia_esp x
Throughput a potencia instalada aproximadamente fija), por lo que un modelo
multi-output que capture su covarianza es mas coherente que dos modelos
independientes -- y en la practica de planta, ambos numeros se necesitan
juntos para decidir un setpoint de operacion.

Compara: regresion lineal (baseline), Random Forest, Gradient Boosting
(sklearn) y LightGBM -- estos dos ultimos envueltos en MultiOutputRegressor
porque el boosting de sklearn/LightGBM no soporta multi-output nativo.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, learning_curve
from sklearn.multioutput import MultiOutputRegressor

TARGET_COLS = ["specific_energy_kwh_t", "throughput_tph"]

FEATURE_COLS = [
    "fresh_feed_tph", "mill_load_pct", "ball_charge_pct", "water_addition_m3h",
    "f80_um", "p80_um", "reduction_ratio",
    "load_deviation_from_optimum", "ball_charge_deviation",
    "wi_hat", "wi_hat_var", "hardness_proxy_wi",
    "fresh_feed_tph_roll_mean_6h", "fresh_feed_tph_roll_std_6h",
    "mill_load_pct_roll_mean_6h", "mill_load_pct_roll_std_6h",
    "hardness_proxy_wi_roll_mean_6h", "hardness_proxy_wi_roll_std_6h",
    "hour_sin", "hour_cos", "day_of_week",
]

TEST_FRACTION = 0.15
RANDOM_STATE = 42


def chronological_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_fraction))
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]


def build_candidate_models() -> dict:
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "gradient_boosting_multioutput": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=250, max_depth=3, learning_rate=0.05,
                subsample=0.85, random_state=RANDOM_STATE,
            )
        ),
        "lightgbm_multioutput": MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=400, num_leaves=31, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                random_state=RANDOM_STATE, verbosity=-1,
            )
        ),
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    preds = model.predict(X_test)
    metrics = {}
    for i, target in enumerate(TARGET_COLS):
        y_true = y_test.iloc[:, i].to_numpy()
        y_pred = preds[:, i]
        metrics[target] = {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    metrics["overall_r2_uniform_average"] = float(r2_score(y_test, preds, multioutput="uniform_average"))
    return metrics


def tune_lightgbm(X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int = 20) -> MultiOutputRegressor:
    """Busqueda aleatoria de hiperparametros con validacion TimeSeriesSplit
    (no K-Fold estandar: los datos son series de tiempo con autocorrelacion
    y features de rolling-window, por lo que un fold aleatorio filtraria
    informacion futura hacia el pasado)."""
    param_distributions = {
        "estimator__num_leaves": [15, 31, 63, 90],
        "estimator__max_depth": [-1, 4, 6, 8],
        "estimator__learning_rate": [0.02, 0.05, 0.08, 0.12],
        "estimator__n_estimators": [200, 300, 400, 600],
        "estimator__min_child_samples": [10, 20, 40],
        "estimator__subsample": [0.7, 0.85, 1.0],
        "estimator__colsample_bytree": [0.7, 0.85, 1.0],
    }
    base = MultiOutputRegressor(LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1))
    search = RandomizedSearchCV(
        base, param_distributions, n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=5), scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def compute_learning_curve(model, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=TimeSeriesSplit(n_splits=5),
        train_sizes=np.linspace(0.15, 1.0, 8), scoring="r2", n_jobs=-1,
    )
    return pd.DataFrame({
        "train_size": train_sizes,
        "train_r2_mean": train_scores.mean(axis=1),
        "train_r2_std": train_scores.std(axis=1),
        "val_r2_mean": test_scores.mean(axis=1),
        "val_r2_std": test_scores.std(axis=1),
    })


def run_training_pipeline(df: pd.DataFrame, out_dir: Path, tune: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df = chronological_split(df)

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COLS]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COLS]

    results = {}
    fitted_models = {}
    for name, model in build_candidate_models().items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        results[name] = evaluate_model(model, X_test, y_test)

    best_params = None
    if tune:
        tuned_model, best_params = tune_lightgbm(X_train, y_train)
        tuned_model.fit(X_train, y_train)
        fitted_models["lightgbm_multioutput_tuned"] = tuned_model
        results["lightgbm_multioutput_tuned"] = evaluate_model(tuned_model, X_test, y_test)

    best_name = min(results, key=lambda k: results[k]["overall_r2_uniform_average"] * -1)
    best_model = fitted_models[best_name]

    joblib.dump(best_model, out_dir / "best_multioutput_model.joblib")
    joblib.dump(FEATURE_COLS, out_dir / "feature_columns.joblib")

    learning_curve_df = compute_learning_curve(best_model, X_train, y_train)
    learning_curve_df.to_csv(reports_dir / "learning_curve.csv", index=False)

    residuals_df = test_df[["timestamp"] + TARGET_COLS].copy()
    preds = best_model.predict(X_test)
    for i, target in enumerate(TARGET_COLS):
        residuals_df[f"{target}_pred"] = preds[:, i]
        residuals_df[f"{target}_residual"] = residuals_df[target] - preds[:, i]
    residuals_df.to_csv(reports_dir / "test_residuals.csv", index=False)

    feature_importance = extract_feature_importance(best_model, FEATURE_COLS)
    feature_importance.to_csv(reports_dir / "feature_importance.csv", index=False)

    report = {
        "best_model": best_name,
        "best_params": best_params,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "metrics": results,
    }
    with open(reports_dir / "multioutput_model_comparison.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


def extract_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    if isinstance(model, MultiOutputRegressor):
        importances = np.mean(
            [est.feature_importances_ for est in model.estimators_], axis=0
        )
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(feature_cols))
    return pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2] / "data"
    df = pd.read_parquet(base / "processed" / "sag_mill_operation_with_kf.parquet")
    out_dir = Path(__file__).resolve().parents[2] / "outputs" / "models"
    report = run_training_pipeline(df, out_dir, tune=True)
    print(json.dumps(report, indent=2))
