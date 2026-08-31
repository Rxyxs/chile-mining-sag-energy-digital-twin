"""Persistencia de metricas comparativas de todos los modelos (arboles y
red neuronal) en una base DuckDB local, para poder consultarlas con SQL en
lugar de solo leer los JSON planos en outputs/reports/.

No reemplaza los reportes JSON existentes (multioutput_model_comparison.json,
deep_learning_activation_comparison.json) -- los consolida en una tabla
unica para analisis y para el README.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

DB_FILENAME = "model_metrics.duckdb"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS model_comparison (
    model_family VARCHAR,
    model_name VARCHAR,
    target VARCHAR,
    rmse DOUBLE,
    mae DOUBLE,
    r2 DOUBLE
)
"""


def _rows_from_tree_report(report: dict) -> list[tuple]:
    rows = []
    for model_name, metrics in report["metrics"].items():
        for target, vals in metrics.items():
            if target == "overall_r2_uniform_average":
                continue
            rows.append(("tree_ensemble", model_name, target, vals["rmse"], vals["mae"], vals["r2"]))
    return rows


def _rows_from_deep_report(report: dict) -> list[tuple]:
    rows = []
    for activation, metrics in report["metrics"].items():
        model_name = f"pytorch_mlp_{activation}"
        for target, vals in metrics.items():
            if target == "overall_r2_uniform_average":
                continue
            rows.append(("deep_learning", model_name, target, vals["rmse"], vals["mae"], vals["r2"]))
    return rows


def persist_model_comparison(
    multioutput_comparison_json: Path,
    deep_learning_comparison_json: Path,
    db_path: Path,
) -> int:
    """Reconstruye la tabla model_comparison desde cero con los reportes
    JSON mas recientes y devuelve el numero de filas insertadas."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple] = []

    if multioutput_comparison_json.exists():
        with open(multioutput_comparison_json) as f:
            rows.extend(_rows_from_tree_report(json.load(f)))

    if deep_learning_comparison_json.exists():
        with open(deep_learning_comparison_json) as f:
            rows.extend(_rows_from_deep_report(json.load(f)))

    con = duckdb.connect(str(db_path))
    try:
        con.execute("DROP TABLE IF EXISTS model_comparison")
        con.execute(CREATE_TABLE_SQL)
        if rows:
            con.executemany(
                "INSERT INTO model_comparison VALUES (?, ?, ?, ?, ?, ?)", rows
            )
        return len(rows)
    finally:
        con.close()


def query_best_per_target(db_path: Path):
    """Devuelve, por target, el modelo con menor RMSE -- util para
    verificar rapidamente cual enfoque gana en la base persistida."""
    con = duckdb.connect(str(db_path))
    try:
        return con.execute(
            """
            SELECT target, model_family, model_name, rmse, r2
            FROM model_comparison
            QUALIFY ROW_NUMBER() OVER (PARTITION BY target ORDER BY rmse ASC) = 1
            ORDER BY target
            """
        ).fetchall()
    finally:
        con.close()


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    reports_dir = base / "outputs" / "reports"
    db_path = reports_dir / DB_FILENAME
    n = persist_model_comparison(
        reports_dir / "multioutput_model_comparison.json",
        reports_dir / "deep_learning_activation_comparison.json",
        db_path,
    )
    print(f"{n} filas persistidas en {db_path}")
    for row in query_best_per_target(db_path):
        print(row)
