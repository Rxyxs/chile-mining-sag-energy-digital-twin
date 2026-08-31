import json

from src.models.duckdb_store import persist_model_comparison, query_best_per_target


def _write(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f)


def test_persist_model_comparison_and_query_best(tmp_path):
    tree_report = {
        "best_model": "lightgbm_multioutput_tuned",
        "metrics": {
            "linear_regression": {
                "specific_energy_kwh_t": {"rmse": 2.0, "mae": 1.5, "r2": 0.3},
                "throughput_tph": {"rmse": 10.0, "mae": 8.0, "r2": 0.4},
                "overall_r2_uniform_average": 0.35,
            },
            "lightgbm_multioutput_tuned": {
                "specific_energy_kwh_t": {"rmse": 0.6, "mae": 0.4, "r2": 0.8},
                "throughput_tph": {"rmse": 3.0, "mae": 2.0, "r2": 0.79},
                "overall_r2_uniform_average": 0.795,
            },
        },
    }
    deep_report = {
        "best_activation": "gelu",
        "metrics": {
            "relu": {
                "specific_energy_kwh_t": {"rmse": 1.2, "mae": 0.9, "r2": 0.55},
                "throughput_tph": {"rmse": 6.0, "mae": 4.5, "r2": 0.5},
                "overall_r2_uniform_average": 0.52,
            },
            "gelu": {
                "specific_energy_kwh_t": {"rmse": 0.9, "mae": 0.7, "r2": 0.65},
                "throughput_tph": {"rmse": 4.5, "mae": 3.5, "r2": 0.6},
                "overall_r2_uniform_average": 0.62,
            },
        },
    }

    tree_json = tmp_path / "multioutput_model_comparison.json"
    deep_json = tmp_path / "deep_learning_activation_comparison.json"
    _write(tree_json, tree_report)
    _write(deep_json, deep_report)

    db_path = tmp_path / "model_metrics.duckdb"
    n_rows = persist_model_comparison(tree_json, deep_json, db_path)

    assert n_rows == 8  # 2 tree models x 2 targets + 2 activations x 2 targets
    assert db_path.exists()

    best_rows = query_best_per_target(db_path)
    best_by_target = {row[0]: row for row in best_rows}
    assert best_by_target["specific_energy_kwh_t"][2] == "lightgbm_multioutput_tuned"
    assert best_by_target["throughput_tph"][2] == "lightgbm_multioutput_tuned"


def test_persist_model_comparison_handles_missing_files(tmp_path):
    db_path = tmp_path / "empty.duckdb"
    n_rows = persist_model_comparison(
        tmp_path / "missing_tree.json", tmp_path / "missing_deep.json", db_path
    )
    assert n_rows == 0
    assert db_path.exists()
