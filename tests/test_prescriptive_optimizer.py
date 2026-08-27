import pandas as pd
import pytest

from prescriptive_optimizer import CONTROLLABLE_COLS, SETPOINT_BOUNDS, PrescriptiveOptimizer

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path("outputs/models/best_multioutput_model.joblib").exists(),
    reason="Requiere modelos entrenados en outputs/models/ (ejecutar run_pipeline.py y train_survival.py primero)",
)


@pytest.fixture(scope="module")
def optimizer():
    return PrescriptiveOptimizer()


@pytest.fixture
def sample_context():
    df = pd.read_parquet("data/processed/sag_mill_operation_with_kf.parquet")
    from src.models.train_multioutput import FEATURE_COLS

    row = df.iloc[-1]
    context = row[FEATURE_COLS].to_dict()
    context["specific_energy_std_recent"] = float(df["specific_energy_kwh_t"].tail(72).std())
    return context


def test_optimize_setpoints_respects_bounds(optimizer, sample_context):
    result = optimizer.optimize_setpoints(sample_context)
    for col in CONTROLLABLE_COLS:
        low, high = SETPOINT_BOUNDS[col]
        value = result["recommended"]["setpoints"][col]
        assert low - 1e-6 <= value <= high + 1e-6


def test_optimize_setpoints_respects_min_throughput_constraint(optimizer, sample_context):
    floor = sample_context["fresh_feed_tph"] * 0.95
    result = optimizer.optimize_setpoints(sample_context, min_throughput_tph=floor)
    assert result["recommended"]["throughput_tph"] >= floor - 1.0


def test_optimize_setpoints_converges(optimizer, sample_context):
    result = optimizer.optimize_setpoints(sample_context)
    assert result["converged"]
