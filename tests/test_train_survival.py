import pandas as pd

from src.data.simulation import simulate_sag_mill_operation
from src.models.train_survival import (
    COVARIATE_COLS,
    build_cycle_covariates,
    build_survival_dataset,
    fit_coxph,
    predict_remaining_life,
)


def _history(n_hours=24 * 200, seed=7):
    return simulate_sag_mill_operation(n_hours=n_hours, seed=seed)


def test_build_cycle_covariates_produces_expected_columns():
    df = _history()
    cycles = build_cycle_covariates(df)
    assert set(COVARIATE_COLS).issubset(cycles.columns)
    assert len(cycles) > 0


def test_survival_dataset_has_valid_durations_and_events():
    df = _history()
    survival_df = build_survival_dataset(df)
    assert (survival_df["duration_hours"] > 0).all()
    assert set(survival_df["event_observed"].unique()).issubset({0, 1})


def test_fit_coxph_recovers_directionally_correct_significant_effects():
    df = _history(n_hours=24 * 400)
    survival_df = build_survival_dataset(df)
    cph, metrics = fit_coxph(survival_df)

    assert 0.0 <= metrics["concordance_index_test"] <= 1.0
    # El covariable dominante del generador sintetico (variabilidad de
    # energia especifica) debe recuperarse con el signo correcto y
    # significancia estadistica clara.
    energy_std_coef = metrics["coefficients"]["specific_energy_std"]
    assert energy_std_coef["estimated_log_hazard_ratio"] > 0
    assert energy_std_coef["p_value"] < 0.01


def test_predict_remaining_life_returns_positive_finite_values():
    df = _history(n_hours=24 * 400)
    survival_df = build_survival_dataset(df)
    cph, _ = fit_coxph(survival_df)

    row = survival_df[COVARIATE_COLS].iloc[[0]]
    result = predict_remaining_life(cph, row)

    assert result["median_remaining_life_hours"] > 0
    assert result["hazard_ratio_vs_reference"] > 0
