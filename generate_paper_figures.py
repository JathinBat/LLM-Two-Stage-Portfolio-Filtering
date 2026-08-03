#!/usr/bin/env python3
"""Generate paper figures from paper_analysis_tables.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLES_JSON = RESULTS_DIR / "paper_analysis_tables.json"
TABLES_XLSX = RESULTS_DIR / "paper_analysis_tables.xlsx"

COLORS = {
    "primary": "#2563EB",
    "secondary": "#059669",
    "accent": "#DC2626",
    "neutral": "#475569",
    "muted": "#CBD5E1",
}

CONFIG_ORDER = [
    "10->5",
    "20->5",
    "unfiltered",
    "10->3",
    "20->10",
    "ranked_final",
    "30->15",
    "5->3",
    "30->10",
    "30->5",
]

DISPLAY = {
    "10->5": "10→5",
    "20->5": "20→5",
    "unfiltered": "Unfiltered",
    "10->3": "10→3",
    "20->10": "20→10",
    "ranked_final": "Ranked Final",
    "30->15": "30→15",
    "5->3": "5→3",
    "30->10": "30→10",
    "30->5": "30→5",
    "gpt-5.1": "GPT-5.1",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
}


def load_tables() -> dict[str, pd.DataFrame]:
    payload = json.loads(TABLES_JSON.read_text(encoding="utf-8"))
    return {name: pd.DataFrame(rows) for name, rows in payload["tables"].items()}


def setup_ax(ax, title: str, xlabel: str | None = None, ylabel: str | None = None):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#94A3B8")
    ax.tick_params(axis="both", labelsize=9)


def save(fig, filename: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(path)


def annotate_hbars(ax, values, fmt="{:.1f}%"):
    xmax = max(values) if len(values) else 0
    pad = xmax * 0.012 if xmax else 0.5
    for i, value in enumerate(values):
        ax.text(value + pad, i, fmt.format(value), va="center", fontsize=9, color="#334155")


def fig_421_config_mean_roi(tables):
    df = tables["Overall Configuration Performance"].copy()
    df["order"] = df["Filter"].map({v: i for i, v in enumerate(CONFIG_ORDER)})
    df = df.sort_values("Mean ROI", ascending=True)
    labels = [DISPLAY.get(v, v) for v in df["Filter"]]
    values = df["Mean ROI"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.barh(labels, values, color=COLORS["primary"])
    setup_ax(
        ax,
        "Figure 4.2.1 – Mean ROI by Filtering Configuration",
        "Mean return on investment (%)",
    )
    annotate_hbars(ax, values)
    ax.set_xlim(0, max(values) * 1.16)
    save(fig, "figure_4_2_1_mean_roi_by_filter.png")


def fig_422_win_rate_vs_unfiltered(tables):
    df = tables["Configuration vs Unfiltered"].copy()
    df["Win Rate %"] = df["Win Rate vs Unfiltered"].astype(float) * 100
    df = df.sort_values("Win Rate %", ascending=True)
    labels = [DISPLAY.get(v, v) for v in df["Filter"]]
    values = df["Win Rate %"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = [COLORS["secondary"] if v >= 50 else COLORS["neutral"] for v in values]
    ax.barh(labels, values, color=colors)
    ax.axvline(50, color="#0F172A", linewidth=1, linestyle="--", alpha=0.75)
    setup_ax(
        ax,
        "Figure 4.2.2 – Filtering Win Rate vs Unfiltered Baseline",
        "Matched evaluation windows beating unfiltered (%)",
    )
    annotate_hbars(ax, values)
    ax.set_xlim(0, max(values) * 1.18)
    save(fig, "figure_4_2_2_win_rate_vs_unfiltered.png")


def fig_431_model_mean_roi(tables):
    df = tables["Model Performance"].copy()
    model_order = ["gpt-5.1", "gpt-4o", "gpt-4o-mini"]
    df["order"] = df["Model"].map({v: i for i, v in enumerate(model_order)})
    df = df.sort_values("order")
    labels = [DISPLAY.get(v, v) for v in df["Model"]]
    values = df["Mean ROI"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.bar(labels, values, color=[COLORS["primary"], COLORS["neutral"], COLORS["secondary"]])
    setup_ax(ax, "Figure 4.3.1 – Mean ROI by Language Model", "Language model", "Mean ROI (%)")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(values) * 1.2)
    for i, value in enumerate(values):
        ax.text(i, value + max(values) * 0.025, f"{value:.1f}%", ha="center", fontsize=9)
    save(fig, "figure_4_3_1_mean_roi_by_model.png")


def fig_432_gpt51_by_config(tables):
    run_df = pd.read_excel(TABLES_XLSX, sheet_name="Run Level Data")
    df = (
        run_df[run_df["Model"] == "gpt-5.1"]
        .groupby("Filter", dropna=False)["ROI"]
        .mean()
        .reset_index(name="Mean ROI")
        .sort_values("Mean ROI", ascending=True)
    )
    labels = [DISPLAY.get(v, v) for v in df["Filter"]]
    values = df["Mean ROI"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.barh(labels, values, color=COLORS["secondary"])
    setup_ax(
        ax,
        "Figure 4.3.2 – GPT-5.1 ROI Across Filtering Configurations",
        "Mean return on investment (%)",
    )
    annotate_hbars(ax, values)
    ax.set_xlim(0, max(values) * 1.16)
    save(fig, "figure_4_3_2_gpt51_by_filter.png")


def fig_441_roi_by_window(tables):
    run_df = pd.read_excel(TABLES_XLSX, sheet_name="Run Level Data")
    grouped = (
        run_df.groupby(["Period (mo)", "Start", "End"], dropna=False)["ROI"]
        .mean()
        .reset_index(name="Mean ROI")
        .sort_values(["Period (mo)", "Start"])
    )

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for period, grp in grouped.groupby("Period (mo)"):
        labels = [f"{s}–{e}" for s, e in zip(grp["Start"], grp["End"])]
        ax.plot(labels, grp["Mean ROI"], marker="o", linewidth=2, label=f"{int(period)} mo")
    setup_ax(ax, "Figure 4.4.1 – ROI by Evaluation Window", "Evaluation window", "Mean ROI (%)")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=45)
    save(fig, "figure_4_4_1_roi_by_evaluation_window.png")


def fig_461_cv_by_config(tables):
    df = tables["Reliability Metrics"].copy()
    df = df.sort_values("CV", ascending=False)
    labels = [DISPLAY.get(v, v) for v in df["Filter"]]
    values = (df["CV"].astype(float) * 100).tolist()

    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.barh(labels, values, color=COLORS["accent"])
    setup_ax(
        ax,
        "Figure 4.6.1 – Coefficient of Variation by Filtering Configuration",
        "Coefficient of variation (%)",
    )
    annotate_hbars(ax, values)
    ax.set_xlim(0, max(values) * 1.16)
    save(fig, "figure_4_6_1_cv_by_filter.png")


def main() -> int:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    tables = load_tables()
    fig_421_config_mean_roi(tables)
    fig_422_win_rate_vs_unfiltered(tables)
    fig_431_model_mean_roi(tables)
    fig_432_gpt51_by_config(tables)
    fig_441_roi_by_window(tables)
    fig_461_cv_by_config(tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
