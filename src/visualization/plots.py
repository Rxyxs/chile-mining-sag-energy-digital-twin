"""Generacion de graficos para el reporte de resultados (EDA + modelos).

Paleta y reglas de diseno: colores categoricos en orden fijo (nunca por
rango), un solo hue para magnitud secuencial, azul<->rojo con punto medio
gris para variables divergentes (correlacion), sin ejes duales, etiquetas
directas selectivas y leyenda siempre presente con >=2 series.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# --- Paleta validada (ver dataviz skill / references/palette.md) ---
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

CAT = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}

MODEL_COLORS = {
    "linear_regression": CAT["magenta"],
    "random_forest": CAT["aqua"],
    "gradient_boosting_multioutput": CAT["yellow"],
    "lightgbm_multioutput": CAT["violet"],
    "lightgbm_multioutput_tuned": CAT["blue"],
}

MODEL_LABELS = {
    "linear_regression": "Regresion Lineal",
    "random_forest": "Random Forest",
    "gradient_boosting_multioutput": "Gradient Boosting",
    "lightgbm_multioutput": "LightGBM",
    "lightgbm_multioutput_tuned": "LightGBM (tuned)",
}

TARGET_LABELS = {
    "specific_energy_kwh_t": "Energia especifica (kWh/t)",
    "throughput_tph": "Throughput (t/h)",
}

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_red_diverging", [CAT["blue"], "#f0efec", CAT["red"]]
)


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_feature_importance(importance_csv: Path, out_path: Path, top_n: int = 15):
    df = pd.read_csv(importance_csv).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    bars = ax.barh(df["feature"], df["importance"], color=CAT["blue"], zorder=3)
    max_importance = df["importance"].max()
    for bar, val in zip(bars, df["importance"]):
        ax.text(bar.get_width() + max_importance * 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                 va="center", fontsize=8, color=INK_SECONDARY)
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Importancia (promedio de ganancia, LightGBM)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Importancia de variables — mejor modelo multi-output", color=INK_PRIMARY,
                 fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_learning_curve(learning_curve_csv: Path, out_path: Path):
    df = pd.read_csv(learning_curve_csv)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)

    ax.plot(df["train_size"], df["train_r2_mean"], color=CAT["blue"], linewidth=2, label="Entrenamiento")
    ax.fill_between(df["train_size"], df["train_r2_mean"] - df["train_r2_std"],
                     df["train_r2_mean"] + df["train_r2_std"], color=CAT["blue"], alpha=0.15)

    ax.plot(df["train_size"], df["val_r2_mean"], color=CAT["orange"], linewidth=2, label="Validacion (TimeSeriesSplit)")
    ax.fill_between(df["train_size"], df["val_r2_mean"] - df["val_r2_std"],
                     df["val_r2_mean"] + df["val_r2_std"], color=CAT["orange"], alpha=0.15)

    _style_axes(ax)
    ax.set_xlabel("Tamano del set de entrenamiento (n registros)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("R² (promedio uniforme, 2 salidas)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Curva de aprendizaje — mejor modelo multi-output", color=INK_PRIMARY,
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.92,
              fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_residuals(residuals_csv: Path, out_path: Path):
    df = pd.read_csv(residuals_csv)
    targets = ["specific_energy_kwh_t", "throughput_tph"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)

    for row, target in enumerate(targets):
        pred_col, resid_col = f"{target}_pred", f"{target}_residual"

        ax = axes[row, 0]
        ax.scatter(df[pred_col], df[resid_col], s=10, color=CAT["blue"], alpha=0.5, zorder=3)
        ax.axhline(0, color=INK_MUTED, linewidth=1, linestyle="--", zorder=2)
        _style_axes(ax)
        ax.set_xlabel(f"Prediccion — {TARGET_LABELS[target]}", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel("Residuo", color=INK_SECONDARY, fontsize=9)
        if row == 0:
            ax.set_title("Residuo vs prediccion", color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")

        ax = axes[row, 1]
        ax.hist(df[resid_col], bins=40, color=CAT["blue"], zorder=3)
        ax.axvline(0, color=INK_MUTED, linewidth=1, linestyle="--", zorder=2)
        _style_axes(ax)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel(f"Residuo — {TARGET_LABELS[target]}", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel("Frecuencia", color=INK_SECONDARY, fontsize=9)
        if row == 0:
            ax.set_title("Distribucion del residuo", color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")

    fig.suptitle("Diagnostico de residuos — set de test (holdout cronologico)", color=INK_PRIMARY,
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_model_comparison(comparison_json: Path, out_path: Path):
    with open(comparison_json) as f:
        report = json.load(f)
    metrics = report["metrics"]
    model_names = [m for m in metrics if m in MODEL_COLORS]
    targets = ["specific_energy_kwh_t", "throughput_tph"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)
    for ax, target in zip(axes, targets):
        values = [metrics[m][target]["rmse"] for m in model_names]
        colors = [MODEL_COLORS[m] for m in model_names]
        bars = ax.bar(range(len(model_names)), values, color=colors, zorder=3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01, f"{val:.2f}",
                     ha="center", fontsize=8, color=INK_SECONDARY)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in model_names], rotation=30, ha="right", fontsize=8)
        _style_axes(ax)
        ax.set_ylabel("RMSE (test holdout)", color=INK_SECONDARY, fontsize=9)
        ax.set_title(TARGET_LABELS[target], color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")

    fig.suptitle("Comparacion de modelos — RMSE en test cronologico", color=INK_PRIMARY,
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_kalman_trace(raw_parquet: Path, kf_parquet: Path, out_path: Path, window_hours: int = 240):
    raw = pd.read_parquet(raw_parquet)
    kf = pd.read_parquet(kf_parquet)
    df = kf.iloc[:window_hours].copy()
    df["ore_hardness_wi_true"] = raw["ore_hardness_wi_true"].iloc[:window_hours].to_numpy()
    df["lab_assay_wi"] = raw["lab_assay_wi"].iloc[:window_hours].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
    ax.plot(df["timestamp"], df["ore_hardness_wi_true"], color=INK_MUTED, linewidth=1.5,
            linestyle="--", label="Wi real (no observable)", zorder=2)
    ax.scatter(df["timestamp"], df["hardness_proxy_wi"], color=CAT["orange"], s=10, alpha=0.5,
               label="Proxy en linea (ruidoso)", zorder=3)
    lab_points = df.dropna(subset=["lab_assay_wi"])
    ax.scatter(lab_points["timestamp"], lab_points["lab_assay_wi"], color=CAT["aqua"], s=45,
               marker="D", label="Ensayo de laboratorio (esparcido)", zorder=4)
    ax.plot(df["timestamp"], df["wi_hat"], color=CAT["blue"], linewidth=2.2,
            label="Estimacion Kalman (fusion)", zorder=5)

    _style_axes(ax)
    ax.set_ylabel("Indice de Trabajo de Bond, Wi (kWh/t)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Filtro de Kalman — fusion de sensores de dureza (ventana de 10 dias)",
                 color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.92,
              fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_forecast_comparison(preview_csv: Path, out_path: Path):
    df = pd.read_csv(preview_csv, parse_dates=["target_timestamp"])
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)

    ax.plot(df["target_timestamp"], df["actual_mw"], color=INK_PRIMARY, linewidth=2, label="Real")
    ax.plot(df["target_timestamp"], df["lightgbm_forecast_mw"], color=CAT["blue"], linewidth=1.8,
            label="LightGBM (24h ahead)")
    ax.plot(df["target_timestamp"], df["holt_winters_forecast_mw"], color=CAT["orange"], linewidth=1.8,
            label="Holt-Winters (24h ahead)")

    _style_axes(ax)
    ax.set_ylabel("Potencia del molino (MW)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Forecast de demanda energetica a 24h — ultimo fold de validacion",
                 color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.92,
              fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_operational_overview(clean_parquet: Path, out_path: Path):
    df = pd.read_parquet(clean_parquet)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), facecolor=SURFACE, sharex=True)

    axes[0].plot(df["timestamp"], df["mill_power_mw"], color=CAT["blue"], linewidth=0.8)
    _style_axes(axes[0])
    axes[0].set_ylabel("Potencia (MW)", color=INK_SECONDARY, fontsize=10)
    axes[0].set_title("Serie operacional completa (180 dias) — potencia y dureza estimada",
                       color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left")

    axes[1].plot(df["timestamp"], df["wi_hat"], color=CAT["violet"], linewidth=0.9)
    _style_axes(axes[1])
    axes[1].set_ylabel("Wi estimado (kWh/t)", color=INK_SECONDARY, fontsize=10)
    axes[1].set_xlabel("Fecha", color=INK_SECONDARY, fontsize=10)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_correlation_heatmap(clean_parquet: Path, out_path: Path):
    df = pd.read_parquet(clean_parquet)
    cols = ["fresh_feed_tph", "mill_load_pct", "ball_charge_pct", "water_addition_m3h",
            "f80_um", "p80_um", "wi_hat", "specific_energy_kwh_t", "throughput_tph", "mill_power_mw"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8.5, 7), facecolor=SURFACE)
    im = ax.imshow(corr, cmap=DIVERGING_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8, color=INK_SECONDARY)
    ax.set_yticklabels(cols, fontsize=8, color=INK_SECONDARY)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            text_color = SURFACE if abs(val) > 0.55 else INK_PRIMARY
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=text_color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.set_title("Matriz de correlacion — variables operacionales y objetivos",
                 color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


ACTIVATION_COLORS = {
    "relu": CAT["orange"],
    "gelu": CAT["aqua"],
    "swish": CAT["blue"],
}

ACTIVATION_LABELS = {
    "relu": "ReLU",
    "gelu": "GELU",
    "swish": "Swish (SiLU)",
}


def plot_activation_loss_curves(loss_curves_csv: Path, out_path: Path):
    df = pd.read_csv(loss_curves_csv)
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    for activation in ACTIVATION_COLORS:
        if activation not in df.columns:
            continue
        ax.plot(df["epoch"], df[activation], color=ACTIVATION_COLORS[activation], linewidth=2,
                label=ACTIVATION_LABELS[activation])
    _style_axes(ax)
    ax.set_xlabel("Epoca", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Huber loss ponderado (train)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("MLP PyTorch — convergencia por funcion de activacion",
                 color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.92,
              fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_deep_vs_tree_benchmark(benchmark_json: Path, out_path: Path):
    with open(benchmark_json) as f:
        report = json.load(f)
    targets = ["specific_energy_kwh_t", "throughput_tph"]
    tree_label = MODEL_LABELS.get(report["tree_model"], report["tree_model"])
    deep_label = report["deep_model"].replace("pytorch_mlp_", "MLP PyTorch (") + ")"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=SURFACE)
    for ax, target in zip(axes, targets):
        values = [report["tree_metrics"][target]["rmse"], report["deep_metrics"][target]["rmse"]]
        colors = [CAT["violet"], CAT["blue"]]
        bars = ax.bar([tree_label, deep_label], values, color=colors, zorder=3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01, f"{val:.2f}",
                     ha="center", fontsize=9, color=INK_SECONDARY)
        ax.tick_params(axis="x", labelrotation=15)
        _style_axes(ax)
        ax.set_ylabel("RMSE (test holdout)", color=INK_SECONDARY, fontsize=9)
        ax.set_title(TARGET_LABELS[target], color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")

    fig.suptitle("Benchmark: arbol ganador vs. red neuronal (surrogate model)", color=INK_PRIMARY,
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def generate_all_plots(base_dir: Path):
    data_dir = base_dir / "data"
    reports_dir = base_dir / "outputs" / "reports"
    plots_dir = base_dir / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_feature_importance(reports_dir / "feature_importance.csv", plots_dir / "feature_importance.png")
    plot_learning_curve(reports_dir / "learning_curve.csv", plots_dir / "learning_curve.png")
    plot_residuals(reports_dir / "test_residuals.csv", plots_dir / "residuals.png")
    plot_model_comparison(reports_dir / "multioutput_model_comparison.json", plots_dir / "model_comparison.png")
    plot_kalman_trace(data_dir / "raw" / "sag_mill_operation_raw.parquet",
                       data_dir / "processed" / "sag_mill_operation_with_kf.parquet",
                       plots_dir / "kalman_filter_trace.png")
    plot_forecast_comparison(reports_dir / "forecast_last_fold_preview.csv", plots_dir / "forecast_comparison.png")
    plot_operational_overview(data_dir / "processed" / "sag_mill_operation_with_kf.parquet",
                               plots_dir / "operational_overview.png")
    plot_correlation_heatmap(data_dir / "processed" / "sag_mill_operation_with_kf.parquet",
                              plots_dir / "correlation_heatmap.png")

    loss_curves_csv = reports_dir / "deep_energy_loss_curves.csv"
    if loss_curves_csv.exists():
        plot_activation_loss_curves(loss_curves_csv, plots_dir / "deep_energy_activation_curves.png")

    deep_vs_tree_json = reports_dir / "deep_vs_tree_benchmark.json"
    if deep_vs_tree_json.exists():
        plot_deep_vs_tree_benchmark(deep_vs_tree_json, plots_dir / "deep_vs_tree_benchmark.png")

    print(f"Graficos guardados en {plots_dir}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    generate_all_plots(base_dir)
