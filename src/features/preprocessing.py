"""Limpieza, tratamiento de outliers, imputacion y feature engineering.

Pipeline aplicado sobre el crudo simulado por ``src.data.simulation`` antes
de la fusion Kalman y el entrenamiento de modelos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

SENSOR_COLS = ["fresh_feed_tph", "mill_load_pct", "water_addition_m3h", "hardness_proxy_wi"]
IQR_TREATED_COLS = ["specific_energy_kwh_t", "throughput_tph", "mill_power_mw"]


def clip_outliers_iqr(df: pd.DataFrame, cols: list[str], factor: float = 3.0) -> pd.DataFrame:
    """Recorta outliers univariados con la regla de Tukey (IQR robusto).

    Se usa un factor de 3.0 (en vez del clasico 1.5) porque el proceso es
    intrinsecamente ruidoso y variable por diseno; 1.5 recortaria variacion
    operacional legitima, no solo errores de sensor.
    """
    out = df.copy()
    for col in cols:
        q1, q3 = out[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        out[col] = out[col].clip(lower, upper)
    return out


def impute_sensor_dropouts(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Imputa caidas de sensor: forward-fill para huecos cortos (<=2h) y
    KNNImputer (sobre variables operacionales correlacionadas) para el resto.
    """
    cols = cols or SENSOR_COLS
    out = df.copy()

    for col in cols:
        out[col] = out[col].ffill(limit=2)

    remaining_na = out[cols].isna().any(axis=1)
    if remaining_na.any():
        imputer = KNNImputer(n_neighbors=5)
        out[cols] = imputer.fit_transform(out[cols])

    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features derivadas: razones de proceso, rolling stats y
    variables temporales ciclicas usadas por los modelos aguas abajo."""
    out = df.copy()
    out = out.sort_values("timestamp").reset_index(drop=True)

    out["reduction_ratio"] = out["f80_um"] / out["p80_um"]
    out["load_deviation_from_optimum"] = (out["mill_load_pct"] - 30.0).abs()
    out["ball_charge_deviation"] = (out["ball_charge_pct"] - 11.0).abs()

    for col in ["fresh_feed_tph", "mill_load_pct", "hardness_proxy_wi"]:
        out[f"{col}_roll_mean_6h"] = out[col].rolling(6, min_periods=1).mean()
        out[f"{col}_roll_std_6h"] = out[col].rolling(6, min_periods=1).std().fillna(0.0)

    out["hour_of_day"] = out["timestamp"].dt.hour
    out["hour_sin"] = np.sin(2 * np.pi * out["hour_of_day"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_of_day"] / 24)
    out["day_of_week"] = out["timestamp"].dt.dayofweek

    return out


def run_preprocessing_pipeline(raw: pd.DataFrame) -> pd.DataFrame:
    """Orquesta limpieza -> imputacion -> feature engineering."""
    cleaned = clip_outliers_iqr(raw, IQR_TREATED_COLS)
    imputed = impute_sensor_dropouts(cleaned)
    featured = engineer_features(imputed)
    return featured


if __name__ == "__main__":
    from pathlib import Path

    base = Path(__file__).resolve().parents[2] / "data"
    raw = pd.read_parquet(base / "raw" / "sag_mill_operation_raw.parquet")
    processed = run_preprocessing_pipeline(raw)
    out_path = base / "processed" / "sag_mill_operation_clean.parquet"
    processed.to_parquet(out_path, index=False)
    print(f"Preprocesados {len(processed)} registros -> {out_path}")
    print(f"NaNs restantes: {int(processed.isna().sum().sum())}")
