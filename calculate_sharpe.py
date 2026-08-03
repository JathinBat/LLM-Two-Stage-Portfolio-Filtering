#!/usr/bin/env python3
"""
calculate_sharpe.py
===================
Reads every perm_*.json in results/, fetches daily closing prices from
Yahoo Finance, and computes accurate portfolio risk metrics:

  Sharpe Ratio      – annualised  (Mean excess return / Vol × √252)
  Sortino Ratio     – annualised  (Mean excess return / Downside-vol × √252)
  Volatility %      – annualised  (Std-dev of daily returns × √252)
  Max Drawdown %    – peak-to-trough of the cumulative return curve
  Calmar Ratio      – annualised return / |Max Drawdown|
  Benchmark Sharpe  – S&P 500 (^GSPC) Sharpe for the same holding period
  Alpha vs Bench %  – portfolio annualised return minus S&P 500 annualised return

Risk-free rate: 13-week T-bill yield (^IRX) averaged over the holding period,
falling back to 5 % per annum if yfinance cannot supply the data.

Outputs
-------
  results/sharpe_analysis.xlsx
      Standalone flat summary — one row per JSON file, all metrics.

  results/permutation_results_T{0.3,0.4,0.5}.xlsx
      The existing permutation Excels are updated IN PLACE — Sharpe columns are
      appended to every data row.  (Re-run compile_results_to_excel.py first if
      the Excels are stale or missing.)

Run from the project root:
    python calculate_sharpe.py
"""

import glob
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)

from permutation_runner import (
    EXCEL_PATH_FMT,
    TEMPERATURES,
    MODELS,
    _NO_FILTER,
    _is_valid_result,
    _match_perm_by_metadata,
    build_permutations,
)

TRADING_DAYS = 252

# Columns appended to every permutation Excel data row (and their widths)
SHARPE_COLS = [
    "Sharpe Ratio",
    "Sortino Ratio",
    "Volatility %",
    "Max Drawdown %",
    "Calmar Ratio",
    "Benchmark Sharpe",
    "Alpha vs Bench %",
]

# ── Module-level yfinance result caches ───────────────────────────────────────
# Key: (ticker, start_str, end_str_exclusive) → pd.Series of daily returns or None
_PRICE_CACHE: Dict[Tuple, Optional[pd.Series]] = {}

# Key: (start_str, end_str_exclusive) → float (annual rate as decimal, e.g. 0.052)
_RF_CACHE: Dict[Tuple, float] = {}

# Key: (start_str, end_str_exclusive) → metrics dict
_BENCH_CACHE: Dict[Tuple, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
#  yfinance helpers
# ─────────────────────────────────────────────────────────────────────────────

def _yf_end(end_date_str: str) -> str:
    """yfinance end is exclusive — add one day so the analysis end date IS included."""
    return (datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


def _fetch_returns(ticker: str, start: str, end_inclusive: str) -> Optional[pd.Series]:
    """
    Return daily simple-return series for *ticker* over [start, end_inclusive].
    Uses an in-process cache so the same ticker/period is only downloaded once.
    """
    end_ex = _yf_end(end_inclusive)
    key = (ticker, start, end_ex)
    if key in _PRICE_CACHE:
        return _PRICE_CACHE[key]

    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end_ex,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if raw.empty:
            _PRICE_CACHE[key] = None
            return None

        # yfinance ≥ 0.2 may return MultiIndex columns: ("Close", ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            if ("Close", ticker) in raw.columns:
                close = raw[("Close", ticker)]
            else:
                close = raw["Close"].iloc[:, 0]
        else:
            close = raw["Close"]

        close = close.squeeze().dropna()
        if len(close) < 2:
            _PRICE_CACHE[key] = None
            return None

        returns = close.pct_change().dropna()
        _PRICE_CACHE[key] = returns
        return returns
    except Exception as exc:
        print(f"  [yf warn] {ticker} ({start} – {end_inclusive}): {exc}")
        _PRICE_CACHE[key] = None
        return None


def _risk_free_rate(start: str, end_inclusive: str) -> float:
    """
    Annualised risk-free rate as a decimal (e.g. 0.0525) for the given period.

    Uses ^IRX (CBOE 13-week T-bill index), which yfinance returns as an
    annualised percentage (e.g. 5.25).  Falls back to 5 % if unavailable.
    """
    end_ex = _yf_end(end_inclusive)
    key = (start, end_ex)
    if key in _RF_CACHE:
        return _RF_CACHE[key]

    try:
        irx = yf.download(
            "^IRX",
            start=start,
            end=end_ex,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if not irx.empty:
            if isinstance(irx.columns, pd.MultiIndex):
                c = irx["Close"].iloc[:, 0]
            else:
                c = irx["Close"]
            rf_pct = float(c.squeeze().dropna().mean())
            rf_dec = rf_pct / 100.0
            _RF_CACHE[key] = rf_dec
            return rf_dec
    except Exception:
        pass

    _RF_CACHE[key] = 0.05   # fallback 5 % per annum
    return 0.05


# ─────────────────────────────────────────────────────────────────────────────
#  Core Sharpe / risk metrics
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_metrics(
    tickers: List[str],
    weights_pct: List[float],
    start_date: str,
    end_date: str,
) -> dict:
    """
    Compute annualised Sharpe, Sortino, volatility, max-drawdown, Calmar for a
    portfolio over [start_date, end_date] (both inclusive).

    Parameters
    ----------
    tickers     : ticker symbols
    weights_pct : allocation percentages (renormalised internally; need not sum
                  to exactly 100)
    start_date  : 'YYYY-MM-DD'
    end_date    : 'YYYY-MM-DD'

    Returns
    -------
    dict with keys: sharpe, sortino, ann_vol_pct, ann_ret_pct, max_dd_pct,
                    calmar, n_days, rf_pct
    Empty dict if the period is too short or no prices are available.
    """
    if not tickers:
        return {}

    # --- Download returns for every ticker ----------------------------------
    series_list, good_tickers, good_weights = [], [], []
    for t, w in zip(tickers, weights_pct):
        r = _fetch_returns(t, start_date, end_date)
        if r is not None and len(r) >= 5:
            series_list.append(r)
            good_tickers.append(t)
            good_weights.append(w)

    if not series_list:
        return {}

    # --- Align on common trading days, build portfolio return series --------
    total_w = sum(good_weights)
    norm_w  = [w / total_w for w in good_weights]

    df = pd.concat(series_list, axis=1, keys=good_tickers)
    df = df.ffill().dropna()        # forward-fill small gaps, remove unaligned days
    n  = len(df)
    if n < 10:
        return {}

    port_ret = (df * norm_w).sum(axis=1)    # daily portfolio returns (decimal)

    # --- Risk-free rate ------------------------------------------------------
    rf_ann  = _risk_free_rate(start_date, end_date)
    rf_d    = (1 + rf_ann) ** (1 / TRADING_DAYS) - 1   # daily equivalent

    excess  = port_ret - rf_d

    # --- Annualised metrics --------------------------------------------------
    ann_ret = float((1 + port_ret.mean()) ** TRADING_DAYS - 1)
    ann_vol = float(port_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))

    sharpe  = (ann_ret - rf_ann) / ann_vol if ann_vol > 1e-8 else 0.0

    # Sortino: downside deviation of excess returns
    neg_ex  = excess[excess < 0]
    if len(neg_ex) >= 2:
        ds_vol = float(neg_ex.std(ddof=1) * np.sqrt(TRADING_DAYS))
    else:
        ds_vol = ann_vol
    sortino = (ann_ret - rf_ann) / ds_vol if ds_vol > 1e-8 else 0.0

    # Max drawdown
    cum       = (1 + port_ret).cumprod()
    drawdown  = (cum - cum.cummax()) / cum.cummax()
    max_dd    = float(drawdown.min())   # negative or zero

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd < -1e-6 else None

    return {
        "sharpe":      round(sharpe,  4),
        "sortino":     round(sortino, 4),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "ann_ret_pct": round(ann_ret * 100, 2),
        "max_dd_pct":  round(max_dd  * 100, 2),
        "calmar":      round(calmar,  4) if calmar is not None else None,
        "n_days":      n,
        "rf_pct":      round(rf_ann  * 100, 2),
    }


def _benchmark(start_date: str, end_date: str) -> dict:
    """S&P 500 metrics for the same period (cached)."""
    key = (start_date, end_date)
    if key not in _BENCH_CACHE:
        _BENCH_CACHE[key] = portfolio_metrics(["^GSPC"], [100.0], start_date, end_date)
    return _BENCH_CACHE[key]


# ─────────────────────────────────────────────────────────────────────────────
#  Build the (perm_id, run_num) → metrics lookup
# ─────────────────────────────────────────────────────────────────────────────

def build_sharpe_lookup(perms: list) -> Tuple[dict, list]:
    """
    Scan all perm_*.json files in RESULTS_DIR, match them to permutations
    (in the same order and using the same logic as compile_results_to_excel.py),
    compute Sharpe metrics for each, and return:

        sharpe_lookup  : { (perm_id, run_num): metrics_dict }
        flat_rows      : list of dicts suitable for a standalone summary Excel
    """
    files      = sorted(glob.glob(os.path.join(RESULTS_DIR, "perm_*.json")))
    perm_by_id = {p["id"]: p for p in perms}
    run_counts: Dict[int, int] = {}   # perm_id -> number of runs seen so far

    sharpe_lookup: dict = {}
    flat_rows: list = []

    total = len(files)
    for i, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        print(f"\r  [{i:4d}/{total}]  {fname[:60]:<60}", end="", flush=True)

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception as exc:
            continue

        if not _is_valid_result(result):
            continue

        p = _match_perm_by_metadata(result, perm_by_id, perms)
        if p is None:
            continue

        pid     = p["id"]
        run_num = run_counts.get(pid, 0) + 1
        run_counts[pid] = run_num

        meta  = result.get("metadata", {})
        start = meta.get("analysis_start_date", "")
        end   = meta.get("analysis_end_date",   "")
        recs  = result.get("final_recommendations", [])

        if not start or not end or not recs:
            continue

        tickers  = [r["ticker"]               for r in recs if "ticker" in r]
        wts      = [r.get("final_allocation", 0) for r in recs]

        m = portfolio_metrics(tickers, wts, start, end)
        if not m:
            continue

        bench = _benchmark(start, end)
        bench_ann_ret = bench.get("ann_ret_pct")
        alpha = (m["ann_ret_pct"] - bench_ann_ret) if bench_ann_ret is not None else None

        row = {
            "Perm ID":          pid,
            "Run #":            run_num,
            "Model":            meta.get("model", ""),
            "Temperature":      meta.get("temperature"),
            "Period (mo)":      meta.get("period_months"),
            "Start":            start[:7],
            "End":              end[:7],
            "Filter":           ("unfiltered"
                                 if (p["init_n"], p["fin_n"]) == _NO_FILTER
                                 else f"{p['init_n']}->{p['fin_n']}"),
            "ROI %":            result.get("return_analysis", {}).get("total_roi"),
            "Ann Return %":     m["ann_ret_pct"],
            "Sharpe Ratio":     m["sharpe"],
            "Sortino Ratio":    m["sortino"],
            "Volatility %":     m["ann_vol_pct"],
            "Max Drawdown %":   m["max_dd_pct"],
            "Calmar Ratio":     m["calmar"],
            "Benchmark Sharpe": bench.get("sharpe"),
            "Alpha vs Bench %": round(alpha, 2) if alpha is not None else None,
            "Risk-Free Rate %": m["rf_pct"],
            "N Trading Days":   m["n_days"],
            "Source File":      fname,
        }
        flat_rows.append(row)

        lookup_metrics = {
            "Sharpe Ratio":     m["sharpe"],
            "Sortino Ratio":    m["sortino"],
            "Volatility %":     m["ann_vol_pct"],
            "Max Drawdown %":   m["max_dd_pct"],
            "Calmar Ratio":     m["calmar"],
            "Benchmark Sharpe": bench.get("sharpe"),
            "Alpha vs Bench %": round(alpha, 2) if alpha is not None else None,
        }
        sharpe_lookup[(pid, run_num)] = lookup_metrics

    print()   # newline after progress bar
    return sharpe_lookup, flat_rows


# ─────────────────────────────────────────────────────────────────────────────
#  Write standalone sharpe_analysis.xlsx
# ─────────────────────────────────────────────────────────────────────────────

def write_sharpe_analysis(flat_rows: list) -> str:
    """Write a standalone summary Excel sorted by Sharpe Ratio descending."""
    if not flat_rows:
        return ""

    out_path = os.path.join(RESULTS_DIR, "sharpe_analysis.xlsx")
    df = pd.DataFrame(flat_rows)

    # Sort: best Sharpe first within each temperature group
    df = df.sort_values(
        ["Temperature", "Sharpe Ratio"],
        ascending=[True, False],
        na_position="last",
    )

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "All Runs"

    thin   = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    hdr_fill = PatternFill("solid", fgColor="FF2E4057")
    hdr_font = Font(bold=True, color="FFFFFFFF", size=10)

    cols = list(df.columns)
    for ci, name in enumerate(cols, 1):
        c = ws.cell(1, ci, name)
        c.fill, c.font = hdr_fill, hdr_font
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = border
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"

    # Temperature fill colours (matching permutation runner)
    temp_fills = {
        0.3: "FFD6E4F0",
        0.4: "FFD4EDDA",
        0.5: "FFFFF3CD",
    }

    for ri, (_, row) in enumerate(df.iterrows(), 2):
        t    = row.get("Temperature")
        fill = PatternFill("solid", fgColor=temp_fills.get(t, "FFFFFFFF"))
        for ci, name in enumerate(cols, 1):
            val = row[name]
            if pd.isna(val):
                val = ""
            cell = ws.cell(ri, ci, val)
            cell.fill   = fill
            cell.border = border
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if name in ("Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                        "Benchmark Sharpe") and isinstance(val, float):
                cell.number_format = "0.0000"
            if name in ("Volatility %", "Max Drawdown %", "Ann Return %",
                        "ROI %", "Alpha vs Bench %", "Risk-Free Rate %") and isinstance(val, float):
                cell.number_format = '0.00"%"'

    # Auto-size columns
    for ci, name in enumerate(cols, 1):
        width = max(len(name) + 2, 8)
        for ri in range(2, ws.max_row + 1):
            v = ws.cell(ri, ci).value
            if v:
                width = max(width, min(len(str(v)) + 2, 40))
        ws.column_dimensions[get_column_letter(ci)].width = width

    try:
        wb.save(out_path)
        return out_path
    except PermissionError:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Patch Sharpe columns into existing permutation_results_T*.xlsx
# ─────────────────────────────────────────────────────────────────────────────

def patch_permutation_excels(sharpe_lookup: dict) -> None:
    """
    Open each permutation_results_T{temp}.xlsx and append the SHARPE_COLS to
    every data row, matched by (Perm ID, Run #).

    Rows that have no Sharpe data (e.g. JSON was unreadable or ticker prices
    were unavailable) receive empty cells.
    """
    thin   = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    hdr_fill = PatternFill("solid", fgColor="FF2E4057")
    hdr_font = Font(bold=True, color="FFFFFFFF", size=10)

    for temp in TEMPERATURES:
        path = EXCEL_PATH_FMT.format(temp=temp)
        if not os.path.exists(path):
            print(f"  T={temp}: {os.path.basename(path)} not found — skip")
            continue

        try:
            wb = openpyxl.load_workbook(path)
        except Exception as exc:
            print(f"  T={temp}: cannot open {os.path.basename(path)}: {exc}")
            continue

        for ws in wb.worksheets:
            if ws.title in ("Summary (95% CI)", "Summary"):
                continue   # skip summary sheet

            # --- Locate "Perm ID" and "Run #" column indices -----------------
            header = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1)}
            pid_col = header.get("Perm ID")
            run_col = header.get("Run #")
            if pid_col is None or run_col is None:
                continue   # not a data sheet we recognise

            # --- Remove any pre-existing Sharpe columns ----------------------
            # (in case this script was run before — avoid duplicates)
            cols_to_remove = []
            for c in range(1, ws.max_column + 1):
                if ws.cell(1, c).value in SHARPE_COLS:
                    cols_to_remove.append(c)
            for c in sorted(cols_to_remove, reverse=True):
                ws.delete_cols(c)

            # --- Append Sharpe column headers --------------------------------
            start_col = ws.max_column + 1
            for ci, name in enumerate(SHARPE_COLS, start_col):
                cell = ws.cell(1, ci, name)
                cell.fill, cell.font = hdr_fill, hdr_font
                cell.alignment = Alignment(horizontal="center", vertical="center",
                                           wrap_text=True)
                cell.border = border
                ws.column_dimensions[get_column_letter(ci)].width = 14

            ws.row_dimensions[1].height = 30

            # --- Write Sharpe data for each data row -------------------------
            for row in range(2, ws.max_row + 1):
                pid_val = ws.cell(row, pid_col).value
                run_val = ws.cell(row, run_col).value
                if pid_val is None:
                    continue    # separator or blank row
                try:
                    pid     = int(pid_val)
                    run_num = int(run_val)
                except (TypeError, ValueError):
                    continue

                m = sharpe_lookup.get((pid, run_num), {})

                # Preserve the row's existing fill colour (copied from col 1)
                try:
                    src = ws.cell(row, 1).fill
                    row_fill = PatternFill("solid", fgColor=src.fgColor.rgb) if src.fill_type == "solid" else None
                except Exception:
                    row_fill = None

                for ci, col_name in enumerate(SHARPE_COLS, start_col):
                    val  = m.get(col_name, "")
                    if val is None:
                        val = ""
                    cell = ws.cell(row, ci, val)
                    if row_fill:
                        cell.fill = row_fill
                    cell.border = border
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    if col_name in ("Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                                    "Benchmark Sharpe") and isinstance(val, float):
                        cell.number_format = "0.0000"
                    if col_name in ("Volatility %", "Max Drawdown %",
                                    "Alpha vs Bench %") and isinstance(val, float):
                        cell.number_format = '0.00"%"'

        try:
            wb.save(path)
            print(f"  T={temp}: {os.path.basename(path)} updated OK")
        except PermissionError:
            print(f"  T={temp}: FAILED — close the file in Excel and re-run")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Portfolio Sharpe Ratio Calculator")
    print("=" * 65)

    print("\nBuilding permutation parameter space ...")
    perms = build_permutations()
    print(f"  {len(perms)} permutations defined")

    print("\nScanning JSON files and computing Sharpe metrics ...")
    print("  (first run downloads prices from Yahoo Finance — may take a while)")
    sharpe_lookup, flat_rows = build_sharpe_lookup(perms)

    computed = len(flat_rows)
    print(f"\n  {computed} runs computed")

    if computed == 0:
        print("\nNo valid results found — nothing to write.")
        return

    # ── Standalone summary Excel ──────────────────────────────────────────────
    print("\nWriting sharpe_analysis.xlsx ...", end=" ", flush=True)
    out = write_sharpe_analysis(flat_rows)
    if out:
        print(f"OK  ({os.path.basename(out)})")
    else:
        print("FAILED (file may be open in Excel)")

    # ── Patch permutation Excels ──────────────────────────────────────────────
    print("\nPatching permutation_results_T*.xlsx ...")
    patch_permutation_excels(sharpe_lookup)

    # ── Quick stats summary ───────────────────────────────────────────────────
    sharpe_vals = [v["Sharpe Ratio"] for v in sharpe_lookup.values()
                   if v.get("Sharpe Ratio") is not None]
    if sharpe_vals:
        print(f"\nSharpe Ratio summary across {len(sharpe_vals)} runs:")
        print(f"  Mean    : {np.mean(sharpe_vals):.4f}")
        print(f"  Median  : {np.median(sharpe_vals):.4f}")
        print(f"  Best    : {max(sharpe_vals):.4f}")
        print(f"  Worst   : {min(sharpe_vals):.4f}")
        print(f"  > 1.0   : {sum(1 for s in sharpe_vals if s > 1.0)} runs")
        print(f"  > 0.0   : {sum(1 for s in sharpe_vals if s > 0.0)} runs")

    print("\nDone.")


if __name__ == "__main__":
    main()
