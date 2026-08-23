"""Simulador fisicamente fundado de un circuito de molienda SAG.

Genera series horarias sinteticas de operacion de un molino SAG a partir de
la Tercera Ley de Bond de la conminucion:

    W = 10 * Wi * (1/sqrt(P80) - 1/sqrt(F80))

donde W es la energia especifica de molienda (kWh/t), Wi es el Indice de
Trabajo de Bond del mineral (kWh/t, proxy de dureza), y P80/F80 son los
tamanos de particula (micrones) que pasan el 80% en producto y alimentacion.

No se usan datos SCADA propietarios: esta es una simulacion sintetica que
respeta la fisica de primer orden de la conminucion (Bond, 1952) mas
factores operacionales no ideales (carga de bolas, % de llenado, agua) que
introducen la no-linealidad que los modelos de ML deben aprender. Los
numeros de planta reales (Wi, P80, potencia instalada) varian por
yacimiento; los rangos aqui son representativos de una linea SAG grande de
cobre porfido chileno (28-32 MW), pero no representan ninguna operacion en
particular.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Constantes de planta (representativas, no de una faena especifica) ---
INSTALLED_POWER_MW = 28.0  # potencia instalada del motor gearless del SAG
F80_BASE_UM = 152_000.0  # alimentacion fresca ~150 mm (152,000 micrones)
P80_SETPOINT_UM = 150.0  # tamano de producto del circuito completo (~150 um,
# el estandar de alimentacion a flotacion en cobre porfido y el mismo P80 que
# define el ensayo de laboratorio del Indice de Trabajo de Bond)
BOND_CONSTANT = 10.0

RNG_SEED_DEFAULT = 42


def _ore_hardness_regime(n_hours: int, rng: np.random.Generator) -> np.ndarray:
    """Indice de Trabajo de Bond (Wi) real, no observado directamente.

    Modelado como un proceso de reversion a la media con saltos de regimen
    cada ~2-4 dias (cambios de fase / bloque de mina), que es como se
    comporta la dureza real: estable dentro de un bloque, con saltos al
    cambiar de zona geometalurgica.
    """
    wi = np.empty(n_hours)
    wi_current = rng.uniform(12.0, 16.0)
    block_target = wi_current
    hours_to_next_block = rng.integers(48, 96)

    for t in range(n_hours):
        if hours_to_next_block <= 0:
            block_target = np.clip(rng.normal(14.0, 2.5), 8.0, 22.0)
            hours_to_next_block = rng.integers(48, 96)

        wi_current += 0.05 * (block_target - wi_current) + rng.normal(0, 0.08)
        wi_current = np.clip(wi_current, 6.0, 24.0)
        wi[t] = wi_current
        hours_to_next_block -= 1

    return wi


def _operational_inefficiency(load_pct: np.ndarray, ball_charge_pct: np.ndarray,
                               water_m3h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Factor multiplicativo (>=1) que penaliza la energia especifica ideal
    de Bond cuando la operacion se aleja de las condiciones optimas.

    El optimo metalurgico tipico es ~ 28-32% de llenado de carga y
    ~10-12% de carga de bolas; alejarse en cualquier direccion incrementa
    el consumo especifico de energia (curva en U), lo mismo que un
    caudal de agua fuera de rango dispersa/densifica mal la pulpa.
    """
    load_penalty = 1.0 + 0.012 * (load_pct - 30.0) ** 2 / 10.0
    ball_penalty = 1.0 + 0.02 * (ball_charge_pct - 11.0) ** 2 / 5.0
    water_penalty = 1.0 + 0.0006 * (water_m3h - 850.0) ** 2 / 100.0
    noise = rng.normal(1.0, 0.02, size=load_pct.shape)
    return np.clip(load_penalty * ball_penalty * water_penalty * noise, 1.0, 2.2)


def simulate_sag_mill_operation(
    n_hours: int = 24 * 180,
    start: str = "2025-01-01",
    seed: int = RNG_SEED_DEFAULT,
    sensor_dropout_rate: float = 0.02,
) -> pd.DataFrame:
    """Simula ``n_hours`` de operacion horaria de un molino SAG.

    Devuelve un DataFrame con variables operacionales controlables,
    proxies de sensor ruidosos, ensayos de laboratorio esparcidos, y las
    variables objetivo (energia especifica, throughput) derivadas de la
    Ley de Bond mas ineficiencia operacional y ruido de medicion.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=n_hours, freq="h")

    wi_true = _ore_hardness_regime(n_hours, rng)

    # --- Variables controlables (setpoints de operador + variacion) ---
    feed_setpoint = 2_450.0 + 150.0 * np.sin(np.arange(n_hours) / (24 * 7) * 2 * np.pi)
    fresh_feed_tph = np.clip(feed_setpoint + rng.normal(0, 40, n_hours), 1_600, 2_900)
    mill_load_pct = np.clip(30.0 + rng.normal(0, 3.0, n_hours), 18.0, 42.0)
    ball_charge_pct = np.clip(11.0 + rng.normal(0, 1.0, n_hours), 7.0, 16.0)
    water_addition_m3h = np.clip(850.0 + rng.normal(0, 60.0, n_hours), 600.0, 1_150.0)
    f80_um = np.clip(F80_BASE_UM + rng.normal(0, 8_000, n_hours), 120_000, 190_000)
    p80_um = np.clip(P80_SETPOINT_UM + rng.normal(0, 12, n_hours), 110.0, 200.0)

    # --- Energia especifica ideal (Bond) + ineficiencia operacional ---
    bond_specific_energy = BOND_CONSTANT * wi_true * (
        1.0 / np.sqrt(p80_um) - 1.0 / np.sqrt(f80_um)
    )
    inefficiency = _operational_inefficiency(mill_load_pct, ball_charge_pct, water_addition_m3h, rng)
    specific_energy_kwh_t = np.clip(
        bond_specific_energy * inefficiency + rng.normal(0, 0.4, n_hours), 4.0, 45.0
    )

    # --- Throughput acoplado fisicamente: Potencia = Energia_esp * Throughput ---
    # A potencia instalada fija, un mayor consumo especifico obliga a bajar
    # el throughput (o viceversa): es la restriccion real de planta.
    throughput_from_power_tph = (INSTALLED_POWER_MW * 1_000.0) / specific_energy_kwh_t
    throughput_tph = np.clip(
        np.minimum(fresh_feed_tph, throughput_from_power_tph) + rng.normal(0, 25, n_hours),
        800.0, 3_000.0,
    )
    mill_power_mw = np.clip(
        specific_energy_kwh_t * throughput_tph / 1_000.0 + rng.normal(0, 0.3, n_hours),
        4.0, INSTALLED_POWER_MW * 1.03,
    )

    # --- Proxy de dureza en tiempo real: ruidoso, disponible cada hora ---
    hardness_proxy_noisy = wi_true + rng.normal(0, 2.6, n_hours)

    # --- Ensayo de laboratorio: preciso pero esparcido (cada 8-12 h) ---
    lab_assay_wi = np.full(n_hours, np.nan)
    t = 0
    while t < n_hours:
        lab_assay_wi[t] = wi_true[t] + rng.normal(0, 0.35)
        t += int(rng.integers(8, 13))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "fresh_feed_tph": fresh_feed_tph,
        "mill_load_pct": mill_load_pct,
        "ball_charge_pct": ball_charge_pct,
        "water_addition_m3h": water_addition_m3h,
        "f80_um": f80_um,
        "p80_um": p80_um,
        "hardness_proxy_wi": hardness_proxy_noisy,
        "lab_assay_wi": lab_assay_wi,
        "mill_power_mw": mill_power_mw,
        "specific_energy_kwh_t": specific_energy_kwh_t,
        "throughput_tph": throughput_tph,
        "ore_hardness_wi_true": wi_true,  # oculto en produccion; solo para validar el KF
    })

    # --- Dropout de sensores: simula caidas de comunicacion / mantenimiento ---
    dropout_mask = rng.random(n_hours) < sensor_dropout_rate
    sensor_cols = ["fresh_feed_tph", "mill_load_pct", "water_addition_m3h", "hardness_proxy_wi"]
    for col in sensor_cols:
        col_mask = dropout_mask & (rng.random(n_hours) < 0.6)
        df.loc[col_mask, col] = np.nan

    return df


if __name__ == "__main__":
    from pathlib import Path

    out_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = simulate_sag_mill_operation()
    data.to_parquet(out_dir / "sag_mill_operation_raw.parquet", index=False)
    print(f"Simulados {len(data)} registros horarios -> {out_dir / 'sag_mill_operation_raw.parquet'}")
    print(data.describe().T)
