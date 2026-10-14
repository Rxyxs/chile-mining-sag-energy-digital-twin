"""Datos reales que anclan el gemelo digital: precio de la energia y consumo de las concentradoras.

El molino se sigue simulando (no hay datos SCADA publicos de un SAG), pero dos magnitudes que
definen el problema economico se toman de fuentes oficiales chilenas:

* **Costo marginal de la energia en la barra Crucero 220 kV**, el nodo de referencia del norte
  minero. Fuente: Reporte Mensual del Sector Energetico de la Comision Nacional de Energia (CNE),
  un PDF por mes en ``https://www.cne.cl/wp-content/uploads/AAAA/MM/RMensual_vAAAAMM.pdf``. El
  reporte de un mes informa el promedio del mes anterior. Hay tres formatos: una frase ("la barra
  Crucero registro un costo marginal promedio de X USD/MWh"), una tabla (valor, variacion mensual y
  anual) y reportes donde el texto es una imagen, que se leen con OCR. Las variaciones de la tabla
  permiten reconstruir el mes anterior y el mismo mes del ano previo cuando falta su reporte; esos
  valores quedan marcados como derivados.

* **Consumo electrico unitario de las concentradoras de cobre (MJ por tonelada procesada)**,
  2010-2024. Fuente: Cochilco, "Consumo de energia en la mineria del cobre, actualizacion ano 2024".
  El informe publica la serie solo como grafico; como el grafico es vectorial, los puntos se leen
  desde las coordenadas del PDF y se calibran con las lineas de grilla del eje (70 y 85 MJ/t).

Las series extraidas se versionan en ``data/real/`` para que el pipeline y los tests corran sin
red; ``refresh_real_data`` las regenera desde los PDF oficiales.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REAL_DIR = ROOT / "data" / "real"
CMG_CSV = REAL_DIR / "cne_crucero_cmg_monthly.csv"
CMG_OBS_CSV = REAL_DIR / "cne_crucero_observations.csv"
COCHILCO_CSV = REAL_DIR / "cochilco_concentradora_mj_t.csv"

CNE_URL = "https://www.cne.cl/wp-content/uploads/{y}/{m:02d}/RMensual_v{y}{m:02d}.pdf"
COCHILCO_URL = ("https://www.cochilco.cl/web/download/991/2025/15151/"
                "consumo-de-energia-en-la-mineria-del-cobre-actualizacion-ano-2024.pdf")
USER_AGENT = "Mozilla/5.0"
MJ_PER_KWH = 3.6

# Prioridad cuando un mismo mes aparece en mas de una fuente: lo publicado por el reporte de ese
# mes gana a lo reconstruido desde variaciones porcentuales de otro reporte.
SOURCE_PRIORITY = {"texto": 0, "tabla": 0, "ocr": 1, "derivado_mensual": 2, "derivado_anual": 3}


def _num(s: str) -> float:
    """'1.234,5' -> 1234.5 (formato chileno)."""
    return float(s.replace(".", "").replace(",", "."))


# ------------------------------------------------------------------------------- CNE: parseo
_SENTENCE = re.compile(
    r"Crucero\s*registr\S*\s*un\s*costo\s*marginal\s*promedio\s*de\s*([\d\.]+,\d)\s*(?:US\$|USD)"
    r"(?:.{0,160}?(?:incremento|aumento|disminuci\S*|decremento|variaci\S*)\s*de\s*\(?\s*(-?\s*[\d\.]+,\d)\s*%)?",
    re.IGNORECASE | re.DOTALL,
)
_TABLE = re.compile(r"Crucero\s+([\d\.]+,\d)\s+(-?[\d\.]+,\d)%\s+(-?[\d\.]+,\d)%")


def parse_crucero(text: str, report_month: str, source: str = "texto") -> list[dict]:
    """Observaciones del costo marginal de Crucero en el texto de un reporte.

    ``report_month`` es el mes de publicacion (AAAA-MM); el dato corresponde al mes anterior.
    Devuelve el valor directo y, si el reporte trae variaciones, los meses reconstruidos.
    """
    ref = pd.Period(report_month, "M") - 1
    flat = re.sub(r"\s+", " ", text)
    obs: list[dict] = []

    i = text.find("[USD/MWh]")
    block = text[i:text.find("PNE", i)] if i >= 0 else ""
    m = _TABLE.search(block)
    if m:
        value, monthly, annual = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
        obs.append({"month": str(ref), "cmg_usd_mwh": value, "source": "tabla" if source == "texto" else source,
                    "change_pct": monthly})
        obs.append({"month": str(ref - 1), "cmg_usd_mwh": round(value / (1 + monthly / 100), 1),
                    "source": "derivado_mensual"})
        obs.append({"month": str(ref - 12), "cmg_usd_mwh": round(value / (1 + annual / 100), 1),
                    "source": "derivado_anual"})
        return obs

    m = _SENTENCE.search(flat)
    if m:
        value = _num(m.group(1))
        change = None
        if m.group(2):
            change = _num(m.group(2).replace(" ", ""))
            word = m.group(0).lower()
            if ("disminu" in word or "decremento" in word) and change > 0:
                change = -change
        obs.append({"month": str(ref), "cmg_usd_mwh": value, "source": source, "change_pct": change})
        if change is not None:
            obs.append({"month": str(ref - 1), "cmg_usd_mwh": round(value / (1 + change / 100), 1),
                        "source": "derivado_mensual"})
    return obs


DIRECT = ("texto", "tabla", "ocr")


def flag_repeated_reports(obs: list[dict]) -> list[dict]:
    """Marca como 'duplicado' el dato de un reporte que repite exactamente el valor y la variacion
    del mes anterior. Ocurre en la fuente: el reporte de enero de 2024 copia la frase del de
    diciembre de 2023 y solo cambia el nombre del mes."""
    direct = sorted((o for o in obs if o["source"] in DIRECT), key=lambda o: o["month"])
    by_month = {o["month"]: o for o in direct}
    for o in direct:
        prev = by_month.get(str(pd.Period(o["month"], "M") - 1))
        if (prev and o.get("change_pct") is not None and prev.get("change_pct") is not None
                and o["cmg_usd_mwh"] == prev["cmg_usd_mwh"] and o["change_pct"] == prev["change_pct"]):
            o["source"] = "duplicado"
    # Lo reconstruido desde un reporte copiado hereda su variacion copiada: tambien se descarta.
    bad = {o.get("report") for o in obs if o["source"] == "duplicado" and o.get("report")}
    for o in obs:
        if o.get("report") in bad:
            o["source"] = "duplicado"
    return obs


def cross_validation(observations: pd.DataFrame) -> dict:
    """Concordancia entre el valor publicado de un mes y el reconstruido desde otro reporte.

    Es la validacion de la extraccion (incluido el OCR): dos reportes distintos deben implicar el
    mismo numero para el mismo mes.
    """
    obs = observations[observations["source"] != "duplicado"]
    published = obs[obs["source"].isin(DIRECT)].drop_duplicates("month").set_index("month")
    derived = obs[obs["source"].str.startswith("derivado")]
    pairs = derived.join(published[["cmg_usd_mwh", "source"]], on="month", rsuffix="_pub").dropna(
        subset=["cmg_usd_mwh_pub"])
    rel = (pairs["cmg_usd_mwh"] - pairs["cmg_usd_mwh_pub"]).abs() / pairs["cmg_usd_mwh_pub"]
    ocr = rel[pairs["source_pub"] == "ocr"]
    return {
        "pairs": int(len(rel)),
        "median_rel_diff": float(rel.median()),
        "p95_rel_diff": float(rel.quantile(0.95)),
        "max_rel_diff": float(rel.max()),
        "ocr_pairs": int(len(ocr)),
        "ocr_max_rel_diff": float(ocr.max()) if len(ocr) else None,
        "discarded_reports": sorted(observations.loc[observations["source"] == "duplicado", "report"].dropna().unique()),
    }


def combine_observations(obs: list[dict]) -> pd.DataFrame:
    """Una fila por mes, quedandose con la fuente de mayor prioridad (los duplicados se descartan)."""
    obs = [o for o in obs if o["source"] != "duplicado"]
    if not obs:
        return pd.DataFrame(columns=["month", "cmg_usd_mwh", "source"])
    df = pd.DataFrame(obs)
    df["rank"] = df["source"].map(SOURCE_PRIORITY)
    df = df.sort_values(["month", "rank"]).drop_duplicates("month").drop(columns="rank")
    keep = [c for c in ("month", "cmg_usd_mwh", "source", "report") if c in df.columns]
    return df[keep].sort_values("month").reset_index(drop=True)


# ------------------------------------------------------------------------------- CNE: descarga
def download_cne_reports(raw_dir: Path, start: str = "2019-01", end: str | None = None, delay: float = 0.3) -> list[Path]:
    """Baja los reportes mensuales disponibles (los meses sin reporte publicado se omiten)."""
    import requests

    raw_dir.mkdir(parents=True, exist_ok=True)
    end = end or str(pd.Period.now("M"))
    paths = []
    session = requests.Session()
    for p in pd.period_range(start, end, freq="M"):
        target = raw_dir / f"RMensual_v{p.year}{p.month:02d}.pdf"
        if not (target.exists() and target.read_bytes()[:4] == b"%PDF"):
            r = session.get(CNE_URL.format(y=p.year, m=p.month), headers={"User-Agent": USER_AGENT}, timeout=60)
            if r.status_code != 200 or r.content[:4] != b"%PDF":
                continue
            target.write_bytes(r.content)
            time.sleep(delay)
        paths.append(target)
    return paths


def report_text(pdf: Path, use_ocr: bool = True) -> tuple[str, str]:
    """Texto de un reporte y como se obtuvo ('texto' u 'ocr'). Los reportes armados como imagen
    no tienen capa de texto: se rasterizan las primeras paginas y se leen con OCR."""
    import pymupdf

    doc = pymupdf.open(pdf)
    text = "\n".join(page.get_text() for page in doc)
    if "Crucero" in text or not use_ocr:
        return text, "texto"
    cache = pdf.with_suffix(".ocr.txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8"), "ocr"
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return text, "texto"
    ocr = RapidOCR()
    lines = []
    for page in list(doc)[:6]:
        res, _ = ocr(page.get_pixmap(dpi=150).tobytes("png"))
        lines += [r[1] for r in (res or [])]
        if any("Crucero" in s for s in lines) and any("USD" in s for s in lines):
            break
    text = " ".join(lines)
    cache.write_text(text, encoding="utf-8")
    return text, "ocr"


def build_cmg_series(raw_dir: Path, use_ocr: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Serie mensual (una fila por mes) y todas las observaciones que la respaldan."""
    obs = []
    for pdf in sorted(raw_dir.glob("RMensual_v*.pdf")):
        text, how = report_text(pdf, use_ocr)
        pub = f"{pdf.stem[10:14]}-{pdf.stem[14:16]}"
        found = parse_crucero(text, pub, source=how)
        for o in found:
            o["report"] = pdf.name
        obs += found
    obs = flag_repeated_reports(obs)
    return combine_observations(obs), pd.DataFrame(obs).sort_values(["month", "source"]).reset_index(drop=True)


# ------------------------------------------------------------------------------- Cochilco
def concentrator_unit_consumption(pdf: Path) -> pd.DataFrame:
    """Lee la serie de consumo unitario de la concentradora (MJ/t procesada) del grafico vectorial.

    Busca la pagina con el panel "Concentradora" en "MJ/TM mineral procesado", calibra el eje Y con
    las etiquetas 70 y 85 y la linea de grilla de cada una, y convierte la polilinea de 15 vertices
    (2010-2024) a valores.
    """
    import pymupdf

    doc = pymupdf.open(pdf)
    for page in doc:
        text = page.get_text()
        if "MJ/TM mineral procesado" not in text or "Concentradora" not in text:
            continue
        labels = {}
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                s = "".join(sp["text"] for sp in line["spans"]).strip()
                if s in ("70", "85"):
                    y = (line["bbox"][1] + line["bbox"][3]) / 2
                    labels.setdefault(s, []).append(y)
        if "70" not in labels or "85" not in labels:
            continue
        for d in page.get_drawings():
            pts = [it[1] for it in d["items"] if it[0] == "l"] + ([d["items"][-1][2]] if d["items"] and d["items"][-1][0] == "l" else [])
            if len(d["items"]) != 14 or d.get("width", 0) < 1.5:
                continue
            ys = np.array([p.y for p in pts])
            y70 = min(labels["70"], key=lambda v: abs(v - ys.max()))
            y85 = min(labels["85"], key=lambda v: abs(v - ys.min()))
            if not (y85 < ys.min() and ys.max() < y70):
                continue
            mj = 70 + (y70 - ys) * (85 - 70) / (y70 - y85)
            out = pd.DataFrame({"year": range(2010, 2010 + len(mj)), "mj_per_t": np.round(mj, 1)})
            out["kwh_per_t"] = (out["mj_per_t"] / MJ_PER_KWH).round(2)
            return out
    raise ValueError("No se encontro el grafico de la concentradora en el informe de Cochilco")


def refresh_real_data(raw_root: Path | None = None, use_ocr: bool = True) -> dict[str, Path]:
    """Descarga los PDF oficiales y regenera las series versionadas en ``data/real``."""
    import requests

    raw_root = raw_root or ROOT / "data" / "raw" / "real"
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    cne_dir = raw_root / "cne"
    download_cne_reports(cne_dir)
    series, observations = build_cmg_series(cne_dir, use_ocr)
    series.to_csv(CMG_CSV, index=False)
    observations.to_csv(CMG_OBS_CSV, index=False)

    raw_root.mkdir(parents=True, exist_ok=True)
    pdf = raw_root / "cochilco_consumo_energia_2024.pdf"
    if not pdf.exists():
        r = requests.get(COCHILCO_URL, headers={"User-Agent": USER_AGENT}, timeout=120)
        r.raise_for_status()
        pdf.write_bytes(r.content)
    concentrator_unit_consumption(pdf).to_csv(COCHILCO_CSV, index=False)
    return {"cmg": CMG_CSV, "observations": CMG_OBS_CSV, "cochilco": COCHILCO_CSV}


def load_cmg() -> pd.DataFrame:
    return pd.read_csv(CMG_CSV)


def load_cmg_observations() -> pd.DataFrame:
    return pd.read_csv(CMG_OBS_CSV)


def load_cochilco() -> pd.DataFrame:
    return pd.read_csv(COCHILCO_CSV)


if __name__ == "__main__":
    out = refresh_real_data()
    cmg, coc = load_cmg(), load_cochilco()
    print(f"CNE Crucero: {len(cmg)} meses, {cmg['month'].min()} a {cmg['month'].max()}")
    print(cmg["source"].value_counts().to_string())
    print(cross_validation(load_cmg_observations()))
    print(f"Cochilco: {len(coc)} anos, {coc['mj_per_t'].min()}-{coc['mj_per_t'].max()} MJ/t")
