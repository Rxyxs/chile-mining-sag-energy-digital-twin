import numpy as np
import pandas as pd

from src.features.kalman_filter import HardnessKalmanFilter, add_kalman_hardness_estimate


def _synthetic_hardness_series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    true_wi = np.full(n, 14.0)
    true_wi[150:] = 18.0  # salto de regimen
    proxy = pd.Series(true_wi + rng.normal(0, 2.5, n))
    lab = pd.Series(np.full(n, np.nan))
    lab.iloc[::10] = true_wi[::10] + rng.normal(0, 0.3, len(lab.iloc[::10]))
    return true_wi, proxy, lab


def test_kalman_filter_reduces_error_vs_raw_proxy():
    true_wi, proxy, lab = _synthetic_hardness_series()
    kf = HardnessKalmanFilter()
    estimate = kf.run(proxy, lab)

    rmse_proxy = np.sqrt(np.mean((proxy.to_numpy() - true_wi) ** 2))
    rmse_kf = np.sqrt(np.mean((estimate["wi_hat"].to_numpy() - true_wi) ** 2))

    assert rmse_kf < rmse_proxy


def test_kalman_filter_output_shape_matches_input():
    _, proxy, lab = _synthetic_hardness_series(n=120)
    kf = HardnessKalmanFilter()
    estimate = kf.run(proxy, lab)
    assert len(estimate) == 120
    assert (estimate["wi_hat_var"] > 0).all()


def test_from_data_estimates_reasonable_parameters():
    _, proxy, lab = _synthetic_hardness_series(n=400)
    kf = HardnessKalmanFilter.from_data(proxy, lab)
    assert kf.process_var > 0
    assert kf.proxy_var > 0
    assert kf.lab_var > 0


def test_add_kalman_hardness_estimate_adds_columns():
    from src.data.simulation import simulate_sag_mill_operation
    from src.features.preprocessing import run_preprocessing_pipeline

    raw = simulate_sag_mill_operation(n_hours=300, seed=4)
    processed = run_preprocessing_pipeline(raw)
    processed["lab_assay_wi"] = raw["lab_assay_wi"].to_numpy()

    out = add_kalman_hardness_estimate(processed)
    assert "wi_hat" in out.columns
    assert "wi_hat_var" in out.columns
    assert out["wi_hat"].notna().all()
