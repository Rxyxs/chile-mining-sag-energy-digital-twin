import numpy as np

from src.data.simulation import simulate_sag_mill_operation


def test_simulate_returns_expected_shape_and_columns():
    df = simulate_sag_mill_operation(n_hours=240, seed=1)
    assert len(df) == 240
    expected_cols = {
        "timestamp", "fresh_feed_tph", "mill_load_pct", "ball_charge_pct",
        "water_addition_m3h", "f80_um", "p80_um", "hardness_proxy_wi",
        "lab_assay_wi", "mill_power_mw", "specific_energy_kwh_t",
        "throughput_tph", "ore_hardness_wi_true",
    }
    assert expected_cols.issubset(df.columns)


def test_simulate_is_deterministic_given_seed():
    df1 = simulate_sag_mill_operation(n_hours=100, seed=7)
    df2 = simulate_sag_mill_operation(n_hours=100, seed=7)
    np.testing.assert_array_equal(df1["specific_energy_kwh_t"], df2["specific_energy_kwh_t"])


def test_physical_ranges_are_plausible():
    df = simulate_sag_mill_operation(n_hours=500, seed=3)
    assert df["specific_energy_kwh_t"].between(4.0, 45.0).all()
    assert df["throughput_tph"].between(800.0, 3000.0).all()
    assert df["mill_power_mw"].between(0.0, 30.0).all()


def test_specific_energy_does_not_pile_up_at_clip_floor():
    """La calibracion de Bond debe producir un rango realista (~8-20 kWh/t
    tipico) sin que una fraccion grande de los datos quede pegada en el
    piso de clipping -- eso indicaria una mala calibracion fisica."""
    df = simulate_sag_mill_operation(n_hours=2000, seed=6)
    floor = df["specific_energy_kwh_t"].min()
    pct_at_floor = (df["specific_energy_kwh_t"] <= floor + 0.05).mean()
    assert pct_at_floor < 0.05


def test_lab_assay_is_sparse_but_present():
    df = simulate_sag_mill_operation(n_hours=500, seed=5)
    n_available = df["lab_assay_wi"].notna().sum()
    assert 0 < n_available < len(df)


def test_sensor_dropout_produces_missing_values():
    df = simulate_sag_mill_operation(n_hours=2000, seed=11, sensor_dropout_rate=0.1)
    assert df["hardness_proxy_wi"].isna().sum() > 0
