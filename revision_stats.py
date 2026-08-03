#!/usr/bin/env python3
"""
revision_stats.py — Workstream A statistics for the NHSJS revision.

Run from the project root on the machine that holds results/perm_*.json:

    python revision_stats.py

Scans EVERY saved run (repeats included) exactly like build_paper_analysis_tables.py,
then produces the reviewer-requested statistics on the pooled run set:

  1. Run-breakdown table (Major #5): per model x temperature x window x configuration —
     runs found, runs valid, runs analyzed (and how many repeats each cell carries).
  2. Exact binomial 95% CIs beside every win rate (Major #5).
  3. Paired per-window tests (Major #3, Rec #2): sign test + Wilcoxon signed-rank on the
     per-cell ROI difference between each configuration and its matched unfiltered
     baseline, plus a bootstrap CI on the median difference.
     Cell values are the MEAN over repeat runs in that cell (pooling basis, matching
     how the submitted paper computed win rates).
  4. GPT-5.1 replication analysis (Major #3): temperature labels collapsed as replicates
     (the API ignored custom temperature for gpt-5.1), within-window spread across all
     repeat runs, and window-level (n = windows) paired tests on replicate means.

Output: results/revision_stats.xlsx and results/revision_stats.json
Requires: pandas, numpy, scipy (same as build_paper_analysis_tables.py).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from permutation_runner import (
    RESULTS_DIR,
    ST_SKIPPED,
    build_permutations,
    _filter_label,
    _is_valid_result,
    _iter_saved_result_json_files,
    _match_perm_by_metadata,
)

RESULTS_PATH = Path(RESULTS_DIR)
OUT_XLSX = RESULTS_PATH / "revision_stats.xlsx"
OUT_JSON = RESULTS_PATH / "revision_stats.json"

PERIOD_FOR_PAPER = 12  # the NHSJS paper analyzes 12-month windows only


def _safe_float(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def load_all_runs() -> pd.DataFrame:
    """Every valid run (repeats included), matched to its permutation cell."""
    perms = build_permutations()
    perm_by_id = {p["id"]: p for p in perms}
    rows = []
    for fpath in _iter_saved_result_json_files():
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            continue
        if not _is_valid_result(result):
            continue
        p = _match_perm_by_metadata(result, perm_by_id, perms)
        if p is None or p["status"] == ST_SKIPPED:
            continue
        ret = result.get("return_analysis", {}) or {}
        bench = ret.get("benchmark_comparison", {}) or {}
        roi = _safe_float(ret.get("total_roi"))
        market = _safe_float(bench.get("market_return"))
        alpha = _safe_float(bench.get("alpha"))
        if alpha is None and roi is not None and market is not None:
            alpha = roi - market
        rows.append({
            "model": p["model"],
            "temp": p["temp"],
            "period": p["period"],
            "start": p["start"].strftime("%Y-%m"),
            "filter": _filter_label(p["init_n"], p["fin_n"]),
            "roi": roi,
            "alpha": alpha,
            "file": str(fpath),
            "timestamp": (result.get("metadata") or {}).get("timestamp"),
        })
    df = pd.DataFrame(rows)
    return df[df["roi"].notna()].reset_index(drop=True)


def run_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["model", "temp", "start", "filter"])
           .agg(runs_analyzed=("roi", "size"))
           .reset_index())
    return g


def _hl_median_ci(x: np.ndarray, n_boot: int = 20000, seed: int = 42):
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, (n_boot, len(x))), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def paired_tests(df: pd.DataFrame, collapse_temp_for: set[str] = frozenset()) -> pd.DataFrame:
    """Cell = model x temp x window (or model x window when temp collapsed).
    Cell value = mean ROI over repeats. Config vs matched unfiltered cell."""
    out = []
    for model, dm in df.groupby("model"):
        keys = ["start"] if model in collapse_temp_for else ["temp", "start"]
        cell = dm.groupby(keys + ["filter"])["roi"].mean().reset_index()
        base = cell[cell["filter"] == "unfiltered"][keys + ["roi"]].rename(columns={"roi": "base"})
        comp = cell[cell["filter"] != "unfiltered"].merge(base, on=keys, how="inner")
        comp["diff"] = comp["roi"] - comp["base"]
        for f, gc in comp.groupby("filter"):
            x = gc["diff"].to_numpy(float)
            n = len(x)
            if n == 0:
                continue
            wins = int((x > 0).sum())
            ci = st.binomtest(wins, n).proportion_ci(0.95, "exact")
            sign_p = st.binomtest(wins, n, 0.5).pvalue
            try:
                w_stat, w_p = st.wilcoxon(x)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
            lo, hi = _hl_median_ci(x)
            out.append({
                "model": model, "filter": f,
                "unit": "window" if model in collapse_temp_for else "temp x window",
                "n_pairs": n, "wins": wins,
                "win_rate_%": 100 * wins / n,
                "winrate_CI_low_%": 100 * ci.low, "winrate_CI_high_%": 100 * ci.high,
                "mean_diff_pp": float(x.mean()), "median_diff_pp": float(np.median(x)),
                "median_CI_low": lo, "median_CI_high": hi,
                "sign_test_p": sign_p, "wilcoxon_W": w_stat, "wilcoxon_p": w_p,
            })
    return pd.DataFrame(out).sort_values(["model", "win_rate_%"], ascending=[True, False])


def gpt51_replication(df: pd.DataFrame) -> pd.DataFrame:
    """All gpt-5.1 runs in a window x config cell are replicates (temperature was
    ignored by the API). Report the spread."""
    g = df[df["model"] == "gpt-5.1"]
    rep = (g.groupby(["start", "filter"])["roi"]
             .agg(["count", "mean", "std", "min", "max"])
             .reset_index())
    rep["range"] = rep["max"] - rep["min"]
    return rep


def main():
    df = load_all_runs()
    d12 = df[df["period"] == PERIOD_FOR_PAPER].copy()
    print(f"valid runs loaded: {len(df)} (12mo: {len(d12)})")

    tables = {
        "Run breakdown": run_breakdown(d12),
        "Paired tests (temp x window)": paired_tests(d12),
        "Paired tests (gpt-5.1 window)": paired_tests(
            d12[d12["model"] == "gpt-5.1"], collapse_temp_for={"gpt-5.1"}),
        "GPT-5.1 replication": gpt51_replication(d12),
        "Config means (pooled)": (
            d12.groupby(["model", "filter"])["roi"]
               .agg(["count", "mean", "median", "std", "min", "max"]).reset_index()),
        "Beat S&P": (
            d12.assign(beat=d12["alpha"] > 0)
               .groupby(["model", "filter"])["beat"]
               .agg(["sum", "count", "mean"]).reset_index()
               .rename(columns={"sum": "beat_n", "count": "n", "mean": "beat_share"})),
    }

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        for name, t in tables.items():
            t.round(4).to_excel(xw, sheet_name=name[:31], index=False)
    OUT_JSON.write_text(json.dumps(
        {k: json.loads(t.round(6).to_json(orient="records")) for k, t in tables.items()},
        indent=1))
    print(f"wrote {OUT_XLSX}\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
