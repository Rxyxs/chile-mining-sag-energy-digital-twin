"""Genera versiones animadas (GIF) de los graficos de series de tiempo genuinos.

Solo se animan los graficos que son series de tiempo reales (linea vs.
tiempo): filtro de Kalman, forecast de energia y la serie operacional
completa. Usa exactamente los mismos datos reales que `plots.py` — no se
fabrica ningun valor. Estilo "racing line chart" sobre fondo oscuro,
guardado con Pillow (sin ffmpeg).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

CAT = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "violet": "#4a3aa7",
}

N_FRAMES_MAX = 50


def _subsample_index(n_points: int, n_frames: int = N_FRAMES_MAX) -> np.ndarray:
    """Indices (crecientes) de los puntos reales a revelar en cada frame."""
    n_frames = min(n_frames, n_points)
    # al menos 2 puntos en el primer frame para que se vea una linea
    return np.unique(np.linspace(2, n_points, n_frames, dtype=int))


def _annotate_tip(ax, x, y, label, color):
    return ax.annotate(
        f"{label}\n{y:,.2f}",
        xy=(x, y),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="none", alpha=0.9),
        zorder=6,
    )


def animate_kalman_trace(raw_parquet: Path, kf_parquet: Path, out_path: Path, window_hours: int = 240):
    raw = pd.read_parquet(raw_parquet)
    kf = pd.read_parquet(kf_parquet)
    df = kf.iloc[:window_hours].copy()
    df["ore_hardness_wi_true"] = raw["ore_hardness_wi_true"].iloc[:window_hours].to_numpy()

    x = df["timestamp"].to_numpy()
    y_true = df["ore_hardness_wi_true"].to_numpy()
    y_hat = df["wi_hat"].to_numpy()

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    line_true, = ax.plot([], [], color="#999999", linewidth=1.5, linestyle="--", label="Wi real (no observable)")
    line_hat, = ax.plot([], [], color=CAT["blue"], linewidth=2.4, label="Estimacion Kalman (fusion)")
    label_true = _annotate_tip(ax, x[0], y_true[0], "Wi real", "#666666")
    label_hat = _annotate_tip(ax, x[0], y_hat[0], "Kalman", CAT["blue"])

    ax.set_xlim(x[0], x[-1])
    pad = 0.08 * (max(y_true.max(), y_hat.max()) - min(y_true.min(), y_hat.min()) + 1e-9)
    ax.set_ylim(min(y_true.min(), y_hat.min()) - pad, max(y_true.max(), y_hat.max()) + pad)
    ax.set_ylabel("Indice de Trabajo de Bond, Wi (kWh/t)")
    ax.set_title("Filtro de Kalman — fusion de sensores de dureza (animado)", fontweight="bold", loc="left")
    ax.legend(loc="upper left", frameon=True, framealpha=0.85)
    fig.autofmt_xdate()
    fig.tight_layout()

    frame_indices = _subsample_index(len(x))

    def update(i):
        k = frame_indices[i]
        line_true.set_data(x[:k], y_true[:k])
        line_hat.set_data(x[:k], y_hat[:k])
        label_true.xy = (x[k - 1], y_true[k - 1])
        label_true.set_text(f"Wi real\n{y_true[k - 1]:,.2f}")
        label_hat.xy = (x[k - 1], y_hat[k - 1])
        label_hat.set_text(f"Kalman\n{y_hat[k - 1]:,.2f}")
        return line_true, line_hat, label_true, label_hat

    ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=120, blit=False)
    ani.save(out_path, writer="pillow")
    plt.close(fig)
    plt.style.use("default")


def animate_forecast_comparison(preview_csv: Path, out_path: Path):
    df = pd.read_csv(preview_csv, parse_dates=["target_timestamp"])
    x = df["target_timestamp"].to_numpy()
    series = {
        "Real": (df["actual_mw"].to_numpy(), "white"),
        "LightGBM": (df["lightgbm_forecast_mw"].to_numpy(), CAT["blue"]),
        "Holt-Winters": (df["holt_winters_forecast_mw"].to_numpy(), CAT["orange"]),
    }

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    lines = {}
    labels = {}
    for name, (y, color) in series.items():
        lines[name], = ax.plot([], [], color=color, linewidth=2, label=name)
        labels[name] = _annotate_tip(ax, x[0], y[0], name, color if color != "white" else "#555555")

    all_y = np.concatenate([y for y, _ in series.values()])
    pad = 0.08 * (all_y.max() - all_y.min() + 1e-9)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax.set_ylabel("Potencia del molino (MW)")
    ax.set_title("Forecast de demanda energetica a 24h (animado)", fontweight="bold", loc="left")
    ax.legend(loc="upper left", frameon=True, framealpha=0.85)
    fig.autofmt_xdate()
    fig.tight_layout()

    frame_indices = _subsample_index(len(x))

    def update(i):
        k = frame_indices[i]
        artists = []
        for name, (y, color) in series.items():
            lines[name].set_data(x[:k], y[:k])
            labels[name].xy = (x[k - 1], y[k - 1])
            labels[name].set_text(f"{name}\n{y[k - 1]:,.1f}")
            artists += [lines[name], labels[name]]
        return artists

    ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=150, blit=False)
    ani.save(out_path, writer="pillow")
    plt.close(fig)
    plt.style.use("default")


def animate_operational_overview(clean_parquet: Path, out_path: Path):
    df = pd.read_parquet(clean_parquet)
    x = df["timestamp"].to_numpy()
    y_power = df["mill_power_mw"].to_numpy()
    y_wi = df["wi_hat"].to_numpy()

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    line_power, = axes[0].plot([], [], color=CAT["blue"], linewidth=0.9)
    label_power = _annotate_tip(axes[0], x[0], y_power[0], "Potencia", CAT["blue"])
    axes[0].set_xlim(x[0], x[-1])
    axes[0].set_ylim(y_power.min() - 1, y_power.max() + 1)
    axes[0].set_ylabel("Potencia (MW)")
    axes[0].set_title("Serie operacional completa (180 dias, animado)", fontweight="bold", loc="left")

    line_wi, = axes[1].plot([], [], color=CAT["violet"], linewidth=1.0)
    label_wi = _annotate_tip(axes[1], x[0], y_wi[0], "Wi estimado", CAT["violet"])
    axes[1].set_ylim(y_wi.min() - 0.3, y_wi.max() + 0.3)
    axes[1].set_ylabel("Wi estimado (kWh/t)")
    axes[1].set_xlabel("Fecha")

    fig.autofmt_xdate()
    fig.tight_layout()

    frame_indices = _subsample_index(len(x))

    def update(i):
        k = frame_indices[i]
        line_power.set_data(x[:k], y_power[:k])
        label_power.xy = (x[k - 1], y_power[k - 1])
        label_power.set_text(f"Potencia\n{y_power[k - 1]:,.1f}")
        line_wi.set_data(x[:k], y_wi[:k])
        label_wi.xy = (x[k - 1], y_wi[k - 1])
        label_wi.set_text(f"Wi estimado\n{y_wi[k - 1]:,.2f}")
        return line_power, line_wi, label_power, label_wi

    ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=100, blit=False)
    ani.save(out_path, writer="pillow")
    plt.close(fig)
    plt.style.use("default")


def generate_all_animations(base_dir: Path):
    data_dir = base_dir / "data"
    plots_dir = base_dir / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    animate_kalman_trace(
        data_dir / "raw" / "sag_mill_operation_raw.parquet",
        data_dir / "processed" / "sag_mill_operation_with_kf.parquet",
        plots_dir / "kalman_filter_trace_animated.gif",
    )
    animate_forecast_comparison(
        base_dir / "outputs" / "reports" / "forecast_last_fold_preview.csv",
        plots_dir / "forecast_comparison_animated.gif",
    )
    animate_operational_overview(
        data_dir / "processed" / "sag_mill_operation_with_kf.parquet",
        plots_dir / "operational_overview_animated.gif",
    )

    print(f"Animaciones guardadas en {plots_dir}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    generate_all_animations(base_dir)
