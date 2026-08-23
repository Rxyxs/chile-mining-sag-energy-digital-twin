"""Filtro de Kalman escalar para estimacion de dureza de mineral (soft sensor).

Problema: el Indice de Trabajo de Bond (Wi) real del mineral que entra al
molino no se mide en linea. En planta se dispone de dos fuentes indirectas:

1. Un proxy de dureza inferido cada hora a partir de senales de potencia /
   vibracion -- disponible siempre, pero ruidoso (R grande).
2. Un ensayo metalurgico de laboratorio sobre muestra compuesta -- preciso
   (R pequeno), pero disponible solo cada 8-12 horas (turnaround de lab).

Un Filtro de Kalman fusiona ambas fuentes con un modelo de caminata
aleatoria para la evolucion de Wi, ponderando cada observacion por su
incertidumbre relativa (ganancia de Kalman), y produce una estimacion
suavizada ``wi_hat`` que se usa como feature de entrada aguas abajo en el
modelo multi-output. Esto es el patron clasico de "soft sensor" en control
de procesos mineros.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class HardnessKalmanFilter:
    """Filtro de Kalman 1D (caminata aleatoria) con dos sensores fusionados
    secuencialmente en cada paso de tiempo.

    Parameters
    ----------
    process_var : float
        Varianza del ruido de proceso Q (cuanto puede variar Wi hora a hora).
    proxy_var : float
        Varianza del ruido de medicion del proxy en linea (R_proxy).
    lab_var : float
        Varianza del ruido de medicion del ensayo de laboratorio (R_lab).
    """

    def __init__(self, process_var: float = 0.05 ** 2, proxy_var: float = 2.6 ** 2,
                 lab_var: float = 0.35 ** 2):
        self.process_var = process_var
        self.proxy_var = proxy_var
        self.lab_var = lab_var

    @classmethod
    def from_data(cls, proxy: pd.Series, lab_assay: pd.Series) -> "HardnessKalmanFilter":
        """Estima Q y R empiricamente a partir de los datos disponibles,
        en vez de fijarlos a mano: una eleccion de diseno mas defendible
        cuando no se conoce el proceso generador real.

        - R_proxy: varianza de alta frecuencia del proxy respecto a su
          media movil de 24h (aproxima el ruido de medicion).
        - R_lab: varianza de alta frecuencia analoga sobre las lecturas
          de laboratorio disponibles.
        - Q: varianza de la primera diferencia de la media movil del
          proxy (aproxima cuanto se mueve el estado real hora a hora).
        """
        proxy_smooth = proxy.rolling(24, min_periods=6, center=True).mean()
        proxy_resid = (proxy - proxy_smooth).dropna()
        proxy_var = float(proxy_resid.var()) if len(proxy_resid) > 5 else 2.6 ** 2

        lab_values = lab_assay.dropna()
        if len(lab_values) > 5:
            lab_var = float(lab_values.diff().dropna().var() / 2.0)
            lab_var = max(lab_var, 0.05 ** 2)
        else:
            lab_var = 0.35 ** 2

        process_var = float(proxy_smooth.diff().dropna().var())
        process_var = max(process_var, 1e-4)

        return cls(process_var=process_var, proxy_var=proxy_var, lab_var=lab_var)

    def run(self, proxy: pd.Series, lab_assay: pd.Series) -> pd.DataFrame:
        """Corre el filtro hacia adelante sobre toda la serie.

        Returns
        -------
        DataFrame con columnas ``wi_hat`` (estimacion filtrada) y
        ``wi_hat_var`` (varianza posterior, util como medida de confianza).
        """
        n = len(proxy)
        proxy_arr = proxy.to_numpy(dtype=float)
        lab_arr = lab_assay.to_numpy(dtype=float)

        x_hat = np.empty(n)
        p_hat = np.empty(n)

        # Inicializacion: primera observacion de proxy disponible.
        x = proxy_arr[0] if not np.isnan(proxy_arr[0]) else np.nanmean(proxy_arr)
        p = self.proxy_var

        for t in range(n):
            # --- Prediccion (caminata aleatoria: transicion identidad) ---
            x_pred = x
            p_pred = p + self.process_var

            # --- Actualizacion con proxy en linea (siempre disponible) ---
            z_proxy = proxy_arr[t]
            if not np.isnan(z_proxy):
                k = p_pred / (p_pred + self.proxy_var)
                x_pred = x_pred + k * (z_proxy - x_pred)
                p_pred = (1 - k) * p_pred

            # --- Actualizacion con ensayo de laboratorio (esparcido) ---
            z_lab = lab_arr[t]
            if not np.isnan(z_lab):
                k = p_pred / (p_pred + self.lab_var)
                x_pred = x_pred + k * (z_lab - x_pred)
                p_pred = (1 - k) * p_pred

            x, p = x_pred, p_pred
            x_hat[t] = x
            p_hat[t] = p

        return pd.DataFrame({"wi_hat": x_hat, "wi_hat_var": p_hat}, index=proxy.index)


def add_kalman_hardness_estimate(df: pd.DataFrame, fit_params: bool = True) -> pd.DataFrame:
    """Agrega la estimacion Kalman de dureza (``wi_hat``) como feature."""
    out = df.copy()
    if fit_params:
        kf = HardnessKalmanFilter.from_data(out["hardness_proxy_wi"], out["lab_assay_wi"])
    else:
        kf = HardnessKalmanFilter()
    estimate = kf.run(out["hardness_proxy_wi"], out["lab_assay_wi"])
    out["wi_hat"] = estimate["wi_hat"].to_numpy()
    out["wi_hat_var"] = estimate["wi_hat_var"].to_numpy()
    return out


if __name__ == "__main__":
    from pathlib import Path

    base = Path(__file__).resolve().parents[2] / "data"
    df = pd.read_parquet(base / "processed" / "sag_mill_operation_clean.parquet")

    raw = pd.read_parquet(base / "raw" / "sag_mill_operation_raw.parquet")
    df["lab_assay_wi"] = raw["lab_assay_wi"].to_numpy()

    df = add_kalman_hardness_estimate(df)

    if "ore_hardness_wi_true" in raw.columns:
        true_wi = raw["ore_hardness_wi_true"].to_numpy()
        rmse_proxy = float(np.sqrt(np.mean((df["hardness_proxy_wi"].to_numpy() - true_wi) ** 2)))
        rmse_kf = float(np.sqrt(np.mean((df["wi_hat"].to_numpy() - true_wi) ** 2)))
        print(f"RMSE proxy crudo vs Wi real:      {rmse_proxy:.3f}")
        print(f"RMSE estimacion Kalman vs Wi real: {rmse_kf:.3f}")
        print(f"Reduccion de error: {(1 - rmse_kf / rmse_proxy) * 100:.1f}%")

    out_path = base / "processed" / "sag_mill_operation_with_kf.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Guardado -> {out_path}")
