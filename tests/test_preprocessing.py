import numpy as np
import pandas as pd

from src.data.simulation import simulate_sag_mill_operation
from src.features.preprocessing import (
    clip_outliers_iqr,
    engineer_features,
    impute_sensor_dropouts,
    run_preprocessing_pipeline,
)


def _raw():
    return simulate_sag_mill_operation(n_hours=500, seed=2, sensor_dropout_rate=0.08)


def test_clip_outliers_bounds_extreme_values():
    df = _raw()
    df.loc[0, "specific_energy_kwh_t"] = 1_000.0
    clipped = clip_outliers_iqr(df, ["specific_energy_kwh_t"])
    assert clipped["specific_energy_kwh_t"].iloc[0] < 1_000.0


def test_impute_sensor_dropouts_removes_nans():
    df = _raw()
    assert df["hardness_proxy_wi"].isna().sum() > 0
    imputed = impute_sensor_dropouts(df)
    assert imputed[["fresh_feed_tph", "mill_load_pct", "water_addition_m3h", "hardness_proxy_wi"]].isna().sum().sum() == 0


def test_engineer_features_adds_expected_columns():
    df = engineer_features(_raw())
    for col in ["reduction_ratio", "hour_sin", "hour_cos", "fresh_feed_tph_roll_mean_6h"]:
        assert col in df.columns
    assert df["hour_sin"].between(-1.0, 1.0).all()


def test_run_preprocessing_pipeline_leaves_no_nans_in_sensor_cols():
    processed = run_preprocessing_pipeline(_raw())
    assert processed[["fresh_feed_tph", "mill_load_pct", "water_addition_m3h", "hardness_proxy_wi"]].isna().sum().sum() == 0
    assert len(processed) == 500
