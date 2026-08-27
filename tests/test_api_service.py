from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not Path("outputs/models/best_multioutput_model.joblib").exists(),
    reason="Requiere modelos entrenados en outputs/models/ (ejecutar run_pipeline.py y train_survival.py primero)",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.api.service import app

    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_endpoint_returns_prediction_and_shap(client):
    reading = {
        "timestamp": "2025-06-01T14:00:00",
        "fresh_feed_tph": 2400.0,
        "mill_load_pct": 30.0,
        "ball_charge_pct": 11.0,
        "water_addition_m3h": 850.0,
        "f80_um": 152_000.0,
        "p80_um": 150.0,
        "hardness_proxy_wi": 14.0,
        "lab_assay_wi": 14.1,
        "mill_power_mw": 25.0,
        "specific_energy_kwh_t": 12.0,
        "throughput_tph": 2200.0,
    }
    r = client.post("/predict", json=reading)
    assert r.status_code == 200
    body = r.json()
    assert "specific_energy_kwh_t" in body["prediction"]
    assert "throughput_tph" in body["prediction"]
    assert set(body["shap_explanation"].keys()) == {"specific_energy_kwh_t", "throughput_tph"}


def test_survival_predict_endpoint(client):
    context = {
        "load_deviation_from_optimum": 2.4,
        "ball_charge_deviation": 0.8,
        "specific_energy_std": 1.0,
        "hardness_proxy_wi_mean": 13.0,
    }
    r = client.post("/survival/predict", json=context)
    assert r.status_code == 200
    body = r.json()
    assert body["median_remaining_life_hours"] > 0
