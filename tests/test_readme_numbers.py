"""Las cifras de datos reales que citan los README salen del pipeline (outputs/reports/energy_economics.json).

El JSON se regenera con ``python run_pipeline.py``; si no existe (checkout limpio sin correr el pipeline), el
test calcula lo mismo desde las series versionadas y el simulador, que es determinista.
"""
import json
from pathlib import Path

import pytest

from src.data.real_sources import cross_validation, load_cmg, load_cmg_observations, load_cochilco
from src.data.simulation import simulate_sag_mill_operation
from src.models.energy_economics import run_energy_economics

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def economics():
    path = ROOT / "outputs" / "reports" / "energy_economics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    out = run_energy_economics(simulate_sag_mill_operation(), load_cmg(), load_cochilco())
    out["price_extraction_check"] = cross_validation(load_cmg_observations())
    return out


def fmt(x: float, lang: str, d: int = 1) -> str:
    s = f"{x:,.{d}f}"
    return s if lang == "en" else s.replace(",", "X").replace(".", ",").replace("X", ".")


@pytest.mark.parametrize("lang,readme", [("en", "README.md"), ("es", "README.es.md")])
def test_readme_quotes_real_data_results(economics, lang, readme):
    text = (ROOT / readme).read_text(encoding="utf-8")
    p, k, v = economics["price"], economics["simulator_check"], economics["value_of_1pct_usd"]
    claims = [
        str(p["months"]),
        fmt(p["mean"], lang), fmt(p["median"], lang), fmt(p["max"], lang),
        fmt(v["p10"] / 1e3, lang, 0), fmt(v["p90"] / 1e3, lang, 0),
        fmt(k["sim_grinding_kwh_t"], lang), fmt(k["implied_grinding_share"] * 100, lang, 0),
        fmt(economics["price_extraction_check"]["max_rel_diff"] * 100, lang),
    ]
    missing = [c for c in claims if c not in text]
    assert not missing, f"{readme} no cita: {missing}"
