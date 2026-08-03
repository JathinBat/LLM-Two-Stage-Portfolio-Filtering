#!/usr/bin/env python3
"""
Permutation Runner
==================
Sweeps every combination of analysis parameters through the
InvestmentStrategyGenerator and tracks progress in a live GUI.

Parameter space
---------------
  Period length  : 12 mo | 24 mo
  Start month    : Jun 2024 -> Jun 2025  (13 dates, monthly steps)
  Temperature    : 0.3 | 0.4 | 0.5
  Model          : gpt-4o | gpt-4o-mini | gpt-5.1
  Filter config  : ranked_final | 30->15 | 30->10 | 30->5 | 20->10 | 20->5 | 10->5 | 10->3 | 5->3
                   (news-companies -> final-portfolio)

Output
------
  results/permutation_results.xlsx  -- one sheet per model,
                                       rows grouped by temperature,
                                       updated after every completed run.
  results/<perm_id>_*.json          -- raw JSON per run (compatible with
                                       results_viewer.py).
"""

import sys
import os
import glob
import json
import re
import subprocess
import tempfile
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from dateutil.relativedelta import relativedelta
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# --- load API keys from .env (kept out of git); safe no-op if .env is absent ---
import os as _os, pathlib as _pl
for _d in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents]:
    _env = _d / ".env"
    if _env.exists():
        for _line in _env.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _os.environ.setdefault(_k.strip(), _v.strip().strip(chr(34)).strip(chr(39)))
        break
# --- end .env loader ---

# ── Locate the generator ───────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, ROOT)            # make sector_config importable
import sector_config                 # sector registry (SECTOR env var; default technology)
if "--save-excel-from-json" not in sys.argv:
    from investment_strategy_generator import InvestmentStrategyGenerator  # noqa: E402

# ── Fixed config ───────────────────────────────────────────────────────────
NYT_API_KEY    = os.environ.get("NYT_API_KEY", "")
NEWS_DATA_PATH = sector_config.news_file(ROOT)   # active-sector news file (default: merged_news_data.csv)
RESULTS_DIR    = os.path.join(ROOT, "results")
EXCEL_PATH_FMT = os.path.join(RESULTS_DIR, "permutation_results_T{temp}.xlsx")
CONFIG_PATH    = os.path.join(ROOT, ".permutation_runner_config.json")
KEYWORD        = sector_config.keyword()         # active sector (SECTOR env var; default "technology")
INVESTMENT_AMT = 10_000.0
CONSOLIDATED_JSON_DIR = os.path.join(RESULTS_DIR, "consolidated_json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Parameter space ────────────────────────────────────────────────────────
PERIOD_MONTHS  = [12, 24]
TEMPERATURES   = [0.3, 0.4, 0.5]
MODELS         = ["gpt-4o", "gpt-4o-mini", "gpt-5.1"]
FILTER_CONFIGS = [
    (-1, -1),  # rank companies at each step, select final portfolio only at the end
    (30, 15),  # identify 30 companies from news, keep 15 after financial validation
    (30, 10),  # identify 30 companies from news, keep 10 after financial validation
    (30, 5),   # identify 30 companies from news, keep 5 after financial validation
    (20, 5),   # identify 20 companies from news, keep 5 after financial validation
    (20, 10),  # identify 20 companies from news, keep 10 after financial validation
    (10, 5),   # identify 10 companies from news, keep 5 after financial validation
    (10, 3),   # identify 10 companies from news, keep 3 after financial validation
    (5,  3),   # identify  5 companies from news, keep 3 after financial validation
    (0,  0),   # no filtering: all news + all financial data, LLM decides freely
]

# Sentinel for the "no filter" config
_NO_FILTER = (0, 0)
_RANKED_FINAL_FILTER = (-1, -1)


def _is_ranked_final_filter(init_n: int, fin_n: int) -> bool:
    return (init_n, fin_n) == _RANKED_FINAL_FILTER


def _filter_label(init_n: int, fin_n: int, arrow: str = "->") -> str:
    if (init_n, fin_n) == _NO_FILTER:
        return "unfiltered"
    if _is_ranked_final_filter(init_n, fin_n):
        return "ranked_final"
    return f"{init_n}{arrow}{fin_n}"


def _load_app_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _save_app_config(config: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _safe_results_subdir(value: str) -> str:
    value = (value or "").strip().replace("\\", "/")
    parts = []
    for part in value.split("/"):
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", part.strip())
        clean = clean.strip("._")
        if clean:
            parts.append(clean)
    return os.path.join(*parts) if parts else ""


def _is_relative_to(path: str, parent: str) -> bool:
    try:
        return (
            os.path.commonpath([os.path.abspath(path), os.path.abspath(parent)])
            == os.path.abspath(parent)
        )
    except ValueError:
        return False


def _iter_saved_result_json_files() -> list[str]:
    """Return source result JSONs, excluding generated consolidated copies."""
    files = []
    for fpath in glob.glob(os.path.join(RESULTS_DIR, "**", "perm_*.json"), recursive=True):
        if _is_relative_to(fpath, CONSOLIDATED_JSON_DIR):
            continue
        files.append(fpath)
    return sorted(files)

_base       = datetime(2024, 6, 1)
START_DATES = [_base + relativedelta(months=i) for i in range(13)]  # Jun 2024 - Jun 2025

# Training-data cutoffs: skip any period whose START falls before the cutoff.
TRAINING_CUTOFFS = {
    "gpt-4o":      datetime(2023, 10, 1),
    "gpt-4o-mini": datetime(2023, 10, 1),
    "gpt-5":       datetime(2024, 10, 1),   # estimated
    "gpt-5.1":     datetime(2024,  9, 1),
}

TODAY = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

# ── Row status labels ──────────────────────────────────────────────────────
ST_PENDING = "Pending"
ST_SKIPPED = "Skipped"
ST_RUNNING = "Running"
ST_DONE    = "Done"
ST_ERROR   = "Error"

ROW_BG = {
    ST_PENDING: "",
    ST_SKIPPED: "#fff3cd",
    ST_RUNNING: "#cce5ff",
    ST_DONE:    "#d4edda",
    ST_ERROR:   "#f8d7da",
}

# Temperature -> Excel fill colour (ARGB hex, no '#')
TEMP_FILLS = {
    0.3: "FFD6E4F0",   # soft blue
    0.4: "FFD6F0D6",   # soft green
    0.5: "FFFFF0D6",   # soft yellow
}


# ──────────────────────────────────────────────────────────────────────────
# Permutation builder
# ──────────────────────────────────────────────────────────────────────────

def _skip_reason(start: datetime, months: int, model: str) -> str | None:
    end    = start + relativedelta(months=months)
    cutoff = TRAINING_CUTOFFS.get(model)
    if cutoff and start < cutoff:
        return f"Training overlap ({model} cutoff {cutoff:%Y-%m})"
    if end > TODAY:
        return f"Period end {end:%Y-%m} > today ({TODAY:%Y-%m})"
    return None


def build_permutations() -> list[dict]:
    rows, idx = [], 0
    # Temperature is the outermost loop so T=0.3 runs finish before T=0.4, etc.
    for temp in TEMPERATURES:
        for period in PERIOD_MONTHS:
            for start in START_DATES:
                for model in MODELS:
                    for (init_n, fin_n) in FILTER_CONFIGS:
                        idx += 1
                        end  = start + relativedelta(months=period)
                        skip = _skip_reason(start, period, model)
                        rows.append({
                            "id":     idx,
                            "period": period,
                            "start":  start,
                            "end":    end,
                            "model":  model,
                            "temp":   temp,
                            "init_n": init_n,
                            "fin_n":  fin_n,
                            "skip":      skip,
                            "status":    ST_SKIPPED if skip else ST_PENDING,
                            "note":      skip or "",
                            "result":    None,    # latest result (for table display)
                            "results":   [],      # all completed runs (for Excel)
                            "elapsed_s": 0,
                        })
    return rows


# ──────────────────────────────────────────────────────────────────────────
# Excel export helper
# ──────────────────────────────────────────────────────────────────────────

EXCEL_COLS = [
    "Perm ID", "Run #", "Period (mo)", "Start", "End",
    "Temperature", "Filter",
    "Return %", "ROI %", "Final Value ($)",
    "Risk Level", "Expected Return", "Strategy Focus",
    "Companies", "Notes",
]


def _extract_row(p: dict, result: dict, run_num: int) -> dict:
    """Flatten one run of a permutation into a dict matching EXCEL_COLS."""
    r    = result or {}
    ret  = r.get("return_analysis", {})
    port = r.get("portfolio_summary", {})
    recs = r.get("final_recommendations", [])

    companies = "  |  ".join(
        f"{c.get('ticker', '?')} ({c.get('final_allocation', 0):.0f}%)"
        for c in recs
    )

    def _num(val):
        try:
            return round(float(val), 2)
        except (TypeError, ValueError):
            return val

    roi_val   = _num(ret.get("total_roi"))
    elapsed_s = r.get("metadata", {}).get("elapsed_s", 0)
    notes_parts = []
    if isinstance(roi_val, (int, float)):
        notes_parts.append(f"ROI: {roi_val:.2f}%")
    notes_parts.append(f"Time: {elapsed_s}s")

    return {
        "Perm ID":         p["id"],
        "Run #":           run_num,
        "Period (mo)":     p["period"],
        "Start":           p["start"].strftime("%Y-%m"),
        "End":             p["end"].strftime("%Y-%m"),
        "Temperature":     p["temp"],
        "Filter":          _filter_label(p["init_n"], p["fin_n"]),
        "Return %":        _num(ret.get("main_return")),
        "ROI %":           roi_val,
        "Final Value ($)": _num(ret.get("total_final_value")),
        "Risk Level":      port.get("risk_level", ""),
        "Expected Return": port.get("expected_return", ""),
        "Strategy Focus":  port.get("strategy_focus", ""),
        "Companies":       companies,
        "Notes":           "  |  ".join(notes_parts),
    }


def save_excel(perms: list[dict], path: str, temp_filter: float | None = None) -> str | None:
    """
    Write completed/errored permutations for one temperature to an Excel workbook.
    One sheet per model, rows sorted by Start ASC then Filter ASC.
    temp_filter: if given, only include perms with that temperature.
    Returns None on success, or an error string.
    """
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        return "openpyxl not installed -- run: pip install openpyxl"

    wb = openpyxl.Workbook()
    if wb.active is not None:
        wb.remove(wb.active)   # remove default blank sheet

    thin = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for model in MODELS:
        ws = wb.create_sheet(title=model)

        # ── Header row ────────────────────────────────────────────────────
        hdr_fill = PatternFill("solid", fgColor="FF2E4057")   # dark blue-grey
        hdr_font = Font(bold=True, color="FFFFFFFF", size=10)
        for col_idx, col_name in enumerate(EXCEL_COLS, start=1):
            cell = ws.cell(row=1, column=col_idx, value=col_name)
            cell.fill = hdr_fill
            cell.font = hdr_font
            cell.alignment = Alignment(horizontal="center", vertical="center",
                                       wrap_text=True)
            cell.border = border

        ws.row_dimensions[1].height = 30
        ws.freeze_panes = "A2"

        # ── Collect and sort rows for this model (one row per run) ──────
        model_runs = []   # list of (perm, result, run_num)
        for p in perms:
            if p["model"] != model:
                continue
            if temp_filter is not None and p["temp"] != temp_filter:
                continue
            for run_num, result in enumerate(p.get("results", []), start=1):
                model_runs.append((p, result, run_num))
        model_runs.sort(key=lambda t: (t[0]["start"], t[0]["init_n"], t[2]))

        # ── Data rows ─────────────────────────────────────────────────────
        for row_idx, (p, result, run_num) in enumerate(model_runs, start=2):
            row_data = _extract_row(p, result, run_num)
            fill_hex = TEMP_FILLS.get(p["temp"], "FFFFFFFF")
            fill     = PatternFill("solid", fgColor=fill_hex)

            for col_idx, col_name in enumerate(EXCEL_COLS, start=1):
                val  = row_data.get(col_name, "")
                cell = ws.cell(row=row_idx, column=col_idx, value=val)
                cell.fill   = fill
                cell.border = border
                cell.alignment = Alignment(
                    horizontal="left" if col_name in ("Companies", "Strategy Focus") else "center",
                    vertical="center",
                    wrap_text=(col_name in ("Companies", "Strategy Focus")),
                )
                if col_name in ("Return %", "ROI %") and isinstance(val, (int, float)):
                    cell.number_format = '0.00"%"'
                if col_name == "Final Value ($)" and isinstance(val, (int, float)):
                    cell.number_format = '"$"#,##0.00'

        # ── Temperature group separator rows ──────────────────────────────
        prev_temp = None
        insert_offset = 0
        for idx, (p, *_) in enumerate(model_runs):
            if p["temp"] != prev_temp and prev_temp is not None:
                sep_row = idx + 2 + insert_offset
                ws.insert_rows(sep_row)
                sep_cell = ws.cell(row=sep_row, column=1,
                                   value=f"-- Temperature {p['temp']} --")
                sep_cell.font = Font(italic=True, color="FF666666", size=9)
                insert_offset += 1
            prev_temp = p["temp"]

        # ── Column widths ─────────────────────────────────────────────────
        col_widths = {
            "Perm ID": 8, "Run #": 6, "Period (mo)": 10, "Start": 10, "End": 10,
            "Temperature": 11, "Filter": 14,
            "Return %": 10, "ROI %": 10, "Final Value ($)": 14,
            "Risk Level": 12, "Expected Return": 14, "Strategy Focus": 30,
            "Companies": 50, "Notes": 28,
        }
        for col_idx, col_name in enumerate(EXCEL_COLS, start=1):
            ws.column_dimensions[
                get_column_letter(col_idx)
            ].width = col_widths.get(col_name, 12)

    # ── Summary sheet: 95% CI across runs per permutation ────────────────
    import math
    SUM_COLS = ["Perm ID", "Model", "Temp", "Period (mo)", "Start", "End",
                "Filter", "N Runs", "Mean ROI %", "Std Dev", "95% CI ±", "CI Low", "CI High"]
    ws_sum = wb.create_sheet(title="Summary (95% CI)")

    hdr_fill = PatternFill("solid", fgColor="FF2E4057")
    hdr_font = Font(bold=True, color="FFFFFFFF", size=10)
    thin = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for col_idx, col_name in enumerate(SUM_COLS, start=1):
        cell = ws_sum.cell(row=1, column=col_idx, value=col_name)
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border
    ws_sum.row_dimensions[1].height = 30
    ws_sum.freeze_panes = "A2"

    # Collect perms that have at least 1 valid run, respecting temp_filter
    sum_perms = [
        p for p in perms
        if p.get("results")
        and (temp_filter is None or p["temp"] == temp_filter)
    ]
    sum_perms.sort(key=lambda p: (p["model"], p["temp"], p["start"], p["init_n"]))

    for row_idx, p in enumerate(sum_perms, start=2):
        rois = []
        for r in p["results"]:
            try:
                rois.append(float(r["return_analysis"]["total_roi"]))
            except (KeyError, TypeError, ValueError):
                pass
        if not rois:
            continue
        n    = len(rois)
        mean = sum(rois) / n
        std  = math.sqrt(sum((x - mean) ** 2 for x in rois) / max(n - 1, 1))
        # t critical value: use 1.96 for large n, else simple lookup
        t_crit = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57,
                  6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}.get(n - 1, 1.96)
        se    = std / math.sqrt(n)
        ci    = t_crit * se

        row_vals = [
            p["id"], p["model"], p["temp"], p["period"],
            p["start"].strftime("%Y-%m"), p["end"].strftime("%Y-%m"),
            _filter_label(p["init_n"], p["fin_n"]),
            n, round(mean, 2), round(std, 2), round(ci, 2),
            round(mean - ci, 2), round(mean + ci, 2),
        ]
        fill = PatternFill("solid", fgColor=TEMP_FILLS.get(p["temp"], "FFFFFFFF"))
        for col_idx, val in enumerate(row_vals, start=1):
            cell = ws_sum.cell(row=row_idx, column=col_idx, value=val)
            cell.fill   = fill
            cell.border = border
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if col_idx >= 9:   # numeric columns
                cell.number_format = '0.00'

    sum_widths = {"Perm ID": 8, "Model": 12, "Temp": 7, "Period (mo)": 10,
                  "Start": 9, "End": 9, "Filter": 14, "N Runs": 7,
                  "Mean ROI %": 12, "Std Dev": 10, "95% CI ±": 10,
                  "CI Low": 10, "CI High": 10}
    for col_idx, col_name in enumerate(SUM_COLS, start=1):
        ws_sum.column_dimensions[get_column_letter(col_idx)].width = sum_widths.get(col_name, 10)

    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp.xlsx",
            dir=out_dir,
        )
        os.close(fd)
        wb.save(tmp_path)
        os.replace(tmp_path, path)
    except PermissionError:
        return f"Permission denied writing {path} - close the file in Excel and try again"
    except OSError as exc:
        return f"Could not write {path}: {exc}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return None   # success


# ──────────────────────────────────────────────────────────────────────────
# Filter helper (module-level so both RunnerThread and App can use it)
# ──────────────────────────────────────────────────────────────────────────

def _invalid_result_reason(result) -> str | None:
    """Return None for a completed result, otherwise explain why it is not usable."""
    if not isinstance(result, dict):
        return "Result was not a JSON object"
    if result.get("error"):
        return str(result.get("error"))[:500]
    if not result.get("final_recommendations"):
        return "Missing final_recommendations"
    try:
        float(result.get("return_analysis", {}).get("total_roi"))
    except (TypeError, ValueError):
        return "Missing numeric return_analysis.total_roi"
    return None


def _is_valid_result(result) -> bool:
    """Return True only if the result has the minimum data needed for a meaningful Excel row."""
    return _invalid_result_reason(result) is None


def _match_perm_by_metadata(result: dict, perm_by_id: dict, perms: list[dict]):
    """
    Return the permutation dict that matches *result* by content, not by perm_id.

    Matching key: (model, temperature, period_months, analysis_start_date YYYY-MM,
                   filter_init, filter_final).

    Falls back to the perm_id stored in the JSON only when a content match fails,
    so that correctly-ID'd legacy files still load.  Returns None on total miss.
    """
    meta = result.get("metadata", {})
    try:
        model       = str(meta.get("model", "")).strip()
        temp        = float(meta.get("temperature", -1))
        period      = int(meta.get("period_months", -1))
        start_str   = str(meta.get("analysis_start_date", ""))[:7]  # "YYYY-MM"
        filter_init = int(meta.get("filter_init", -99))
        filter_fin  = int(meta.get("filter_final", -99))
    except (TypeError, ValueError):
        return None

    if not (model and temp > 0 and period > 0 and start_str):
        return None

    for p in perms:
        if (p["model"]  == model
                and abs(p["temp"] - temp) < 1e-6
                and p["period"] == period
                and p["start"].strftime("%Y-%m") == start_str
                and p["init_n"] == filter_init
                and p["fin_n"]  == filter_fin):
            return p

    # Content match failed — fall back to the stored perm_id
    if all(k in meta for k in (
        "model", "temperature", "period_months", "analysis_start_date",
        "filter_init", "filter_final",
    )):
        return None

    pid = meta.get("perm_id")
    if pid is not None:
        return perm_by_id.get(int(pid))
    return None


def _perm_passes_filter(p: dict, filters: dict) -> bool:
    """Return True if permutation p satisfies all active filter criteria."""
    if not (filters["id_from"] <= p["id"] <= filters["id_to"]):
        return False
    if p["model"] not in filters["models"]:
        return False
    if p["temp"] not in filters["temps"]:
        return False
    if p["period"] not in filters["periods"]:
        return False
    if (p["init_n"], p["fin_n"]) not in filters["filter_cfgs"]:
        return False
    if not (filters["date_from"] <= p["start"] <= filters["date_to"]):
        return False
    return True


def load_json_results_into_perms(perms: list[dict]) -> tuple[int, int]:
    """Load all valid saved JSON results into the matching permutation rows."""
    files = _iter_saved_result_json_files()
    perm_by_id = {p["id"]: p for p in perms}
    files_read = runs_loaded = 0

    for fpath in files:
        files_read += 1
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
        p["results"].append(result)
        runs_loaded += 1

    for p in perms:
        if not p["results"]:
            continue
        latest = p["results"][-1]
        p["status"] = ST_DONE
        p["result"] = latest
        p["elapsed_s"] = latest.get("metadata", {}).get("elapsed_s", 0)
    return files_read, runs_loaded


def save_excel_from_json_files() -> int:
    """CLI-safe Excel rebuild. Runs in a child process to isolate openpyxl crashes."""
    perms = build_permutations()
    files_read, runs_loaded = load_json_results_into_perms(perms)
    print(f"Loaded {runs_loaded} valid run(s) from {files_read} JSON file(s).")
    if runs_loaded == 0:
        return 0

    failed = False
    for temp in TEMPERATURES:
        count = sum(len(p["results"]) for p in perms if p["temp"] == temp)
        if count == 0:
            continue
        path = EXCEL_PATH_FMT.format(temp=temp)
        err = save_excel(perms, path, temp_filter=temp)
        if err:
            print(f"T={temp}: {err}", file=sys.stderr)
            failed = True
        else:
            print(f"T={temp}: saved {os.path.basename(path)} ({count} run(s))")
    return 1 if failed else 0


# ──────────────────────────────────────────────────────────────────────────
# Background worker thread
# ──────────────────────────────────────────────────────────────────────────

class RunnerThread(threading.Thread):
    """Coordinator thread: distributes permutations across a ThreadPoolExecutor."""

    def __init__(self, perms, q, api_key, stop_evt, pause_evt, n_threads, filters,
                 target_runs, retry_errors_only=False, results_subdir=""):
        super().__init__(daemon=True)
        self.perms             = perms
        self.q                 = q
        self.api_key           = api_key
        self.stop_evt          = stop_evt
        self.pause_evt         = pause_evt
        self.n_threads         = n_threads
        self.filters           = filters
        self.target_runs       = target_runs
        self.retry_errors_only = retry_errors_only
        self.results_subdir    = _safe_results_subdir(results_subdir)
        self.output_dir        = os.path.join(RESULTS_DIR, self.results_subdir) if self.results_subdir else RESULTS_DIR

    def _post(self, kind, **kw):
        self.q.put({"kind": kind, **kw})

    def _perm_label(self, p):
        return (
            f"#{p['id']:04d}  {p['period']}mo  "
            f"{p['start']:%Y-%m}->{p['end']:%Y-%m}  "
            f"{p['model']}  T={p['temp']}  "
            + _filter_label(p["init_n"], p["fin_n"])
        )

    def _run_one_perm(self, p):
        """Execute a single permutation. Runs inside a ThreadPoolExecutor worker."""
        run_token = f"{p['id']}-{threading.get_ident()}-{time.time_ns()}"
        self._post("status", id=p["id"], run_token=run_token, status=ST_RUNNING, note="")
        self._post("log", text=f"[{datetime.now():%H:%M:%S}] ▶ {self._perm_label(p)}")

        t0 = time.time()
        try:
            is_no_filter = (p["init_n"], p["fin_n"]) == _NO_FILTER
            is_ranked_final = _is_ranked_final_filter(p["init_n"], p["fin_n"])
            gen = InvestmentStrategyGenerator(
                api_key_openai=self.api_key,
                nyt_api_key=NYT_API_KEY,
                temperature=p["temp"],
                model=p["model"],
                initial_candidates=50 if is_no_filter else (0 if is_ranked_final else p["init_n"]),
                final_portfolio=15 if is_no_filter else (0 if is_ranked_final else p["fin_n"]),
            )

            t0 = time.time()  # reset after construction (only time the strategy call)

            # The LLM decides AT analysis_start_date — only show news before that moment.
            news_end   = p["start"]
            news_start = p["start"] - relativedelta(years=2)

            result = gen.generate_complete_strategy(
                user_input_keyword=KEYWORD,
                investment_amount=INVESTMENT_AMT,
                news_start_date=news_start.strftime("%Y-%m-%d"),
                news_end_date=news_end.strftime("%Y-%m-%d"),
                analysis_start_date=p["start"].strftime("%Y-%m-%d"),
                analysis_end_date=p["end"].strftime("%Y-%m-%d"),
                news_data_path=NEWS_DATA_PATH,
                include_financial_validation=True,
                single_pass=is_no_filter,
                ranked_final_selection=is_ranked_final,
            )
            elapsed_s = round(time.time() - t0)

            if not isinstance(result, dict):
                result = {"error": "Generator returned a non-JSON result", "raw_result": str(result)}

            result.setdefault("metadata", {})
            result["metadata"].update({
                    "perm_id":       p["id"],
                    "period_months": p["period"],
                    "filter_init":   p["init_n"],
                    "filter_final":  p["fin_n"],
                    "filter_mode":   "ranked_final" if is_ranked_final else ("unfiltered" if is_no_filter else "standard"),
                    "elapsed_s":     elapsed_s,
                })

            try:
                roi_raw = result.get("return_analysis", {}).get("total_roi")
                roi_str = f"ROI: {float(roi_raw):.2f}%  |  " if roi_raw is not None else ""
            except Exception:
                roi_str = ""
            time_str = f"Time: {elapsed_s}s"

            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = (
                f"perm_{p['id']:04d}_"
                f"{p['model'].replace('.', '')}_"
                f"T{p['temp']}_"
                f"{p['period']}mo_"
                f"{p['start']:%Y%m}_{ts}.json"
            )
            os.makedirs(self.output_dir, exist_ok=True)
            fpath = os.path.join(self.output_dir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

            rel_fname = os.path.join(self.results_subdir, fname) if self.results_subdir else fname
            invalid_reason = _invalid_result_reason(result)
            if invalid_reason:
                note = f"{time_str}  |  {rel_fname}  |  {invalid_reason[:120]}"
                self._post("status", id=p["id"], run_token=run_token, status=ST_ERROR,
                           note=note, result=None, elapsed_s=elapsed_s)
                self._post("log",
                           text=f"[{datetime.now():%H:%M:%S}]  x #{p['id']:04d} INVALID RESULT ({elapsed_s}s): {invalid_reason} -> {rel_fname}")
                return

            note = f"{roi_str}{time_str}  |  {rel_fname}"
            self._post("status", id=p["id"], run_token=run_token, status=ST_DONE,
                       note=note, result=result, elapsed_s=elapsed_s)
            self._post("log",
                       text=f"[{datetime.now():%H:%M:%S}]  v #{p['id']:04d}  {roi_str}{time_str}  -> {rel_fname}")

        except Exception as exc:
            import traceback
            elapsed_s = round(time.time() - t0)
            note = f"Time: {elapsed_s}s  |  {str(exc)[:120]}"
            self._post("status", id=p["id"], run_token=run_token, status=ST_ERROR,
                       note=note, result=None, elapsed_s=elapsed_s)
            self._post("log",
                       text=f"[{datetime.now():%H:%M:%S}]  x #{p['id']:04d} ERROR ({elapsed_s}s): {exc}")
            # ── Write human-readable error log ────────────────────────────
            try:
                logs_dir = os.path.join(self.output_dir, "logs")
                os.makedirs(logs_dir, exist_ok=True)
                ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_name = f"perm_{p['id']:04d}_error_{ts}.txt"
                log_path = os.path.join(logs_dir, log_name)
                with open(log_path, "w", encoding="utf-8") as lf:
                    lf.write(f"{'='*60}\n")
                    lf.write(f"PERMUTATION ERROR REPORT\n")
                    lf.write(f"{'='*60}\n\n")
                    lf.write(f"Timestamp  : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                    lf.write(f"Perm ID    : {p['id']}\n")
                    lf.write(f"Model      : {p['model']}\n")
                    lf.write(f"Temperature: {p['temp']}\n")
                    lf.write(f"Period     : {p['period']} months\n")
                    lf.write(f"Start date : {p['start']:%Y-%m-%d}\n")
                    lf.write(f"End date   : {p['end']:%Y-%m-%d}\n")
                    _filt_label = _filter_label(p["init_n"], p["fin_n"], arrow=" -> ")
                    lf.write(f"Filter     : {_filt_label}\n")
                    lf.write(f"Elapsed    : {elapsed_s}s\n\n")
                    lf.write(f"{'─'*60}\n")
                    lf.write(f"ERROR\n")
                    lf.write(f"{'─'*60}\n")
                    lf.write(f"{type(exc).__name__}: {exc}\n\n")
                    lf.write(f"{'─'*60}\n")
                    lf.write(f"TRACEBACK\n")
                    lf.write(f"{'─'*60}\n")
                    lf.write(traceback.format_exc())
            except Exception:
                pass  # never let log writing crash the runner

    def run(self):
        # Build work list.
        work = []
        if self.retry_errors_only:
            # Retry mode: one attempt for every perm currently marked ST_ERROR
            # that passes the active filters.
            for p in self.perms:
                if p["status"] != ST_ERROR:
                    continue
                if not _perm_passes_filter(p, self.filters):
                    continue
                work.append(p)
        else:
            # Normal mode: each perm repeated (target_runs - existing_runs) times.
            for p in self.perms:
                if p["status"] == ST_SKIPPED:
                    continue
                if not _perm_passes_filter(p, self.filters):
                    continue
                remaining = self.target_runs - len(p.get("results", []))
                for _ in range(remaining):
                    work.append(p)

        scheduled_by_pid = {}
        for p in work:
            scheduled_by_pid[p["id"]] = scheduled_by_pid.get(p["id"], 0) + 1
        self._post("plan", scheduled_by_pid=scheduled_by_pid, total=len(work))

        sem = threading.Semaphore(self.n_threads)

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for p in work:
                if self.stop_evt.is_set():
                    break

                # Honour pause before acquiring a worker slot
                while self.pause_evt.is_set() and not self.stop_evt.is_set():
                    time.sleep(0.3)
                if self.stop_evt.is_set():
                    break

                sem.acquire()
                if self.stop_evt.is_set():
                    sem.release()
                    break

                def _task(p=p):           # default-arg captures current p
                    try:
                        self._run_one_perm(p)
                    finally:
                        sem.release()     # free slot for next submission

                futures.append(executor.submit(_task))

            # Wait for every submitted task to complete before posting "done"
            for f in futures:
                try:
                    f.result()
                except Exception:
                    pass

        self._post("done")


# ──────────────────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────────────────

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Investment Strategy Permutation Runner")
        self.geometry("1300x820")
        self.minsize(900, 600)

        self.perms       = build_permutations()
        self._q          = queue.Queue()
        self._stop_evt   = threading.Event()
        self._pause_evt  = threading.Event()
        self._worker     = None
        self._iid: dict[int, str] = {}
        self._sort_rev: dict[str, bool] = {}
        self._done_count = 0
        self._app_config = _load_app_config()
        self._active_runs: dict[str, dict] = {}
        self._run_total = 0
        self._run_started = 0
        self._run_finished = 0
        self._run_succeeded = 0
        self._run_failed = 0
        self._recent_runs: list[str] = []
        self._scheduled_by_pid: dict[int, int] = {}
        self._finished_by_pid: dict[int, int] = {}
        self._excel_save_running = False
        self._excel_save_requested = False

        # ── Filter state (snapshot at Start time, passed to RunnerThread) ──
        max_id = max(p["id"] for p in self.perms)
        self._id_from_var  = tk.IntVar(value=1)
        self._id_to_var    = tk.IntVar(value=max_id)
        self._model_vars   = {m: tk.BooleanVar(value=True) for m in MODELS}
        self._temp_vars    = {t: tk.BooleanVar(value=True) for t in TEMPERATURES}
        self._period_vars  = {p: tk.BooleanVar(value=True) for p in PERIOD_MONTHS}
        self._filt_vars    = {(i, f): tk.BooleanVar(value=True)
                              for (i, f) in FILTER_CONFIGS}
        _date_strs = [d.strftime("%Y-%m") for d in START_DATES]
        self._date_from_var = tk.StringVar(value=_date_strs[0])
        self._date_to_var   = tk.StringVar(value=_date_strs[-1])
        self._filter_widgets: list = []   # populated in _build_ui for lock/unlock

        self._build_ui()
        self._load_existing_results()   # restore ST_DONE from previous sessions
        self._fill_table()
        self._refresh_stats()
        self.after(250, self._drain_queue)

    # ── Restore previous session results ───────────────────────────────────

    def _load_existing_results(self):
        """Scan results/perm_*.json and restore all runs from prior sessions.
        Matches each file to its permutation by metadata content (model, temp,
        period, start, filter) so that perm ID shifts from adding new filter
        configs do not corrupt the restore."""
        _, total_runs = load_json_results_into_perms(self.perms)
        for p in self.perms:
            if not p["results"]:
                continue
            latest    = p["results"][-1]
            elapsed_s = latest.get("metadata", {}).get("elapsed_s", 0)
            n         = len(p["results"])
            try:
                roi_raw = latest.get("return_analysis", {}).get("total_roi")
                roi_str = f"ROI: {float(roi_raw):.2f}%  |  " if roi_raw is not None else ""
            except Exception:
                roi_str = ""
            p["status"]    = ST_DONE
            p["result"]    = latest
            p["elapsed_s"] = elapsed_s
            p["note"]      = f"{n} run(s)  |  {roi_str}Time: {elapsed_s}s"

        if total_runs:
            n_perms = sum(1 for p in self.perms if p["results"])
            self._log_append(
                f"[Startup] Restored {total_runs} run(s) across "
                f"{n_perms} perm(s) from existing JSON files."
            )
            self._on_save_excel()   # rebuild Excel with all restored runs

    # ── Layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        bar = ttk.Frame(self, padding=(8, 6))
        bar.pack(fill="x", side="top")

        self._btn_start  = ttk.Button(bar, text="▶  Start",       command=self._on_start)
        self._btn_pause  = ttk.Button(bar, text="⏸  Pause",       command=self._on_pause,        state="disabled")
        self._btn_stop   = ttk.Button(bar, text="⏹  Stop",        command=self._on_stop,         state="disabled")
        self._btn_retry  = ttk.Button(bar, text="Retry Errors", command=self._on_retry_errors)
        self._btn_excel  = ttk.Button(bar, text="Save Excel now",  command=self._on_save_excel)
        for btn in (self._btn_start, self._btn_pause, self._btn_stop, self._btn_retry, self._btn_excel):
            btn.pack(side="left", padx=3)

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(bar, text="Threads:").pack(side="left", padx=(0, 2))
        self._threads_var = tk.IntVar(value=1)
        self._spin_threads = ttk.Spinbox(
            bar, from_=1, to=16, textvariable=self._threads_var, width=4)
        self._spin_threads.pack(side="left", padx=(0, 10))

        ttk.Label(bar, text="Runs/perm:").pack(side="left", padx=(0, 2))
        self._runs_var = tk.IntVar(value=1)
        self._spin_runs = ttk.Spinbox(
            bar, from_=1, to=50, textvariable=self._runs_var, width=4)
        self._spin_runs.pack(side="left", padx=(0, 10))

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)

        self._stats_var = tk.StringVar()
        ttk.Label(bar, textvariable=self._stats_var,
                  font=("Courier", 10)).pack(side="left")

        ttk.Label(bar, text="OpenAI API key:").pack(side="right", padx=(10, 2))
        saved_key = self._app_config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
        self._key_var = tk.StringVar(value=saved_key)
        ttk.Entry(bar, textvariable=self._key_var, width=52,
                  show="*").pack(side="right")

        ttk.Label(bar, text="Folder:").pack(side="right", padx=(10, 2))
        self._folder_var = tk.StringVar(value=self._app_config.get("results_subdir", ""))
        self._folder_entry = ttk.Entry(bar, textvariable=self._folder_var, width=22)
        self._folder_entry.pack(side="right")

        pb_frame = ttk.Frame(self, padding=(8, 2))
        pb_frame.pack(fill="x", side="top")
        self._pb_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(pb_frame, variable=self._pb_var,
                        maximum=100, mode="determinate").pack(fill="x", expand=True)
        self._cur_var = tk.StringVar(value="Not started")
        ttk.Label(pb_frame, textvariable=self._cur_var,
                  font=("Courier", 9), foreground="#444").pack(anchor="w")

        leg = ttk.Frame(self, padding=(8, 0))
        leg.pack(fill="x", side="top")
        for label, color in [
            ("Pending", "#e0e0e0"), ("Skipped", "#fff3cd"),
            ("Running", "#cce5ff"), ("Done", "#d4edda"), ("Error", "#f8d7da"),
        ]:
            tk.Label(leg, text=f"  {label}  ", background=color,
                     relief="solid", bd=1, font=("Courier", 8)).pack(side="left", padx=4, pady=2)

        # ── Excel colour legend ───────────────────────────────────────────
        ttk.Separator(leg, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Label(leg, text="Excel rows:", font=("Courier", 8)).pack(side="left", padx=4)
        for _, fgcolor, lbl in [
            (0.3, "#D6E4F0", "T=0.3"), (0.4, "#D6F0D6", "T=0.4"), (0.5, "#FFF0D6", "T=0.5"),
        ]:
            tk.Label(leg, text=f"  {lbl}  ", background=fgcolor,
                     relief="solid", bd=1, font=("Courier", 8)).pack(side="left", padx=2)

        # ── Run filter bar ──────────────────────────────────────────────────
        fbar = ttk.LabelFrame(self, text="Run Filters  (applied at Start)", padding=(8, 4))
        fbar.pack(fill="x", side="top", padx=6, pady=(0, 2))

        # Row 1 — ID range + model checkboxes
        r1 = ttk.Frame(fbar)
        r1.pack(fill="x", anchor="w")

        ttk.Label(r1, text="ID range:").pack(side="left")
        spin_from = ttk.Spinbox(r1, from_=1, to=9999,
                                textvariable=self._id_from_var, width=6)
        spin_from.pack(side="left", padx=(3, 1))
        ttk.Label(r1, text="to").pack(side="left", padx=2)
        spin_to = ttk.Spinbox(r1, from_=1, to=9999,
                              textvariable=self._id_to_var, width=6)
        spin_to.pack(side="left", padx=(1, 8))
        self._filter_widgets += [spin_from, spin_to]

        ttk.Separator(r1, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(r1, text="Models:").pack(side="left", padx=(0, 4))
        for m in MODELS:
            cb = ttk.Checkbutton(r1, text=m, variable=self._model_vars[m])
            cb.pack(side="left", padx=2)
            self._filter_widgets.append(cb)

        # Row 2 — Temperature + period + filter-config checkboxes
        r2 = ttk.Frame(fbar)
        r2.pack(fill="x", anchor="w", pady=(3, 0))

        ttk.Label(r2, text="Temps:").pack(side="left", padx=(0, 4))
        for t in TEMPERATURES:
            cb = ttk.Checkbutton(r2, text=str(t), variable=self._temp_vars[t])
            cb.pack(side="left", padx=2)
            self._filter_widgets.append(cb)

        ttk.Separator(r2, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(r2, text="Periods:").pack(side="left", padx=(0, 4))
        for p in PERIOD_MONTHS:
            cb = ttk.Checkbutton(r2, text=f"{p}mo", variable=self._period_vars[p])
            cb.pack(side="left", padx=2)
            self._filter_widgets.append(cb)

        ttk.Separator(r2, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(r2, text="Filter configs:").pack(side="left", padx=(0, 4))
        for (i, f) in FILTER_CONFIGS:
            lbl_text = _filter_label(i, f)
            cb = ttk.Checkbutton(r2, text=lbl_text,
                                 variable=self._filt_vars[(i, f)])
            cb.pack(side="left", padx=2)
            self._filter_widgets.append(cb)

        # Row 3 — Analysis start date range
        r3 = ttk.Frame(fbar)
        r3.pack(fill="x", anchor="w", pady=(3, 0))

        _date_strs = [d.strftime("%Y-%m") for d in START_DATES]
        ttk.Label(r3, text="Start date:").pack(side="left")
        cb_from = ttk.Combobox(r3, textvariable=self._date_from_var,
                               values=_date_strs, state="readonly", width=9)
        cb_from.pack(side="left", padx=(3, 1))
        ttk.Label(r3, text="to").pack(side="left", padx=2)
        cb_to = ttk.Combobox(r3, textvariable=self._date_to_var,
                             values=_date_strs, state="readonly", width=9)
        cb_to.pack(side="left", padx=(1, 8))
        self._filter_widgets += [cb_from, cb_to]

        pw = ttk.PanedWindow(self, orient="vertical")
        pw.pack(fill="both", expand=True, padx=6, pady=4)

        tbl = ttk.Frame(pw)
        pw.add(tbl, weight=4)

        COLS = ("#", "Period", "Start", "End", "Model", "Temp", "Filter", "Status", "Note")
        WIDTHS = {"#": 48, "Period": 64, "Start": 72, "End": 72,
                  "Model": 90, "Temp": 50, "Filter": 100, "Status": 72, "Note": 340}

        self._tree = ttk.Treeview(tbl, columns=COLS, show="headings", selectmode="browse")
        for col in COLS:
            self._tree.heading(col, text=col, command=lambda c=col: self._sort(c))
            self._tree.column(col, width=WIDTHS.get(col, 80),
                              anchor="w" if col == "Note" else "center",
                              stretch=(col == "Note"))

        ys = ttk.Scrollbar(tbl, orient="vertical",   command=self._tree.yview)
        xs = ttk.Scrollbar(tbl, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y")
        xs.pack(side="bottom", fill="x")
        self._tree.pack(fill="both", expand=True)

        for st, bg in ROW_BG.items():
            if bg:
                self._tree.tag_configure(st, background=bg)

        log_lf = ttk.LabelFrame(pw, text="Log", padding=4)
        pw.add(log_lf, weight=1)
        self._log_box = scrolledtext.ScrolledText(
            log_lf, height=7, state="disabled",
            font=("Courier", 9), wrap="word")
        self._log_box.pack(fill="both", expand=True)

    # ── Table ──────────────────────────────────────────────────────────────

    def _row_vals(self, p):
        return (
            p["id"], f"{p['period']} mo",
            p["start"].strftime("%Y-%m"), p["end"].strftime("%Y-%m"),
            p["model"], p["temp"],
            _filter_label(p["init_n"], p["fin_n"], arrow=" -> "),
            p["status"], p["note"],
        )

    def _fill_table(self):
        for p in self.perms:
            iid = self._tree.insert("", "end", values=self._row_vals(p),
                                    tags=(p["status"],))
            self._iid[p["id"]] = iid

    def _update_row(self, pid):
        p = self.perms[pid - 1]
        self._tree.item(self._iid[pid], values=self._row_vals(p), tags=(p["status"],))
        self._tree.see(self._iid[pid])

    # ── Stats ──────────────────────────────────────────────────────────────

    def _refresh_stats(self):
        if self._run_total and self._run_finished < self._run_total:
            running_attempts = len(self._active_runs)
            pending_attempts = max(0, self._run_total - self._run_started)
            self._stats_var.set(
                f"Attempts {self._run_finished}/{self._run_total} finished  |  "
                f"OK {self._run_succeeded}  Error {self._run_failed}  "
                f"Running {running_attempts}  Pending {pending_attempts}"
            )
            self._pb_var.set(self._run_finished / self._run_total * 100)
            return

        total   = len(self.perms)
        skipped = sum(1 for p in self.perms if p["status"] == ST_SKIPPED)
        done    = sum(1 for p in self.perms if p["status"] == ST_DONE)
        error   = sum(1 for p in self.perms if p["status"] == ST_ERROR)
        running = sum(1 for p in self.perms if p["status"] == ST_RUNNING)
        pending = sum(1 for p in self.perms if p["status"] == ST_PENDING)
        runnable  = total - skipped
        pct = ((done + error) / runnable * 100) if runnable else 0.0
        self._stats_var.set(
            f"Total {total}  |  Runnable {runnable}  |  "
            f"Done {done}  Error {error}  Running {running}  "
            f"Pending {pending}  Skipped {skipped}"
        )
        self._pb_var.set(pct)

    def _reset_run_tracking(self, scheduled_by_pid: dict[int, int]):
        self._active_runs.clear()
        self._run_total = sum(scheduled_by_pid.values())
        self._run_started = 0
        self._run_finished = 0
        self._run_succeeded = 0
        self._run_failed = 0
        self._recent_runs = []
        self._scheduled_by_pid = dict(scheduled_by_pid)
        self._finished_by_pid = {pid: 0 for pid in scheduled_by_pid}
        for pid, count in self._scheduled_by_pid.items():
            p = self.perms[pid - 1]
            p["status"] = ST_PENDING
            p["note"] = f"{count} queued run attempt(s)"
            if pid in self._iid:
                self._update_row(pid)
        self._refresh_run_display()

    def _scheduled_for_start(self, filters: dict, target_runs: int) -> dict[int, int]:
        scheduled = {}
        for p in self.perms:
            if p["status"] == ST_SKIPPED:
                continue
            if not _perm_passes_filter(p, filters):
                continue
            remaining = max(0, target_runs - len(p.get("results", [])))
            if remaining:
                scheduled[p["id"]] = remaining
        return scheduled

    def _scheduled_for_retry(self, filters: dict) -> dict[int, int]:
        return {
            p["id"]: 1
            for p in self.perms
            if p["status"] == ST_ERROR and _perm_passes_filter(p, filters)
        }

    def _refresh_run_display(self):
        if not self._run_total:
            self._stats_var.set("Attempts 0/0 finished  |  OK 0  Error 0  Running 0  Pending 0")
            self._cur_var.set("No active run plan.")
            self.update_idletasks()
            return

        running = len(self._active_runs)
        pending = max(0, self._run_total - self._run_started)
        finished = self._run_finished
        pct = finished / self._run_total * 100 if self._run_total else 0.0
        self._pb_var.set(pct)
        self._stats_var.set(
            f"Attempts {finished}/{self._run_total} finished  |  "
            f"OK {self._run_succeeded}  Error {self._run_failed}  "
            f"Running {running}  Pending {pending}"
        )

        active_items = list(self._active_runs.values())
        active_labels = [
            f"#{p['id']:04d} {p['period']}mo {p['start']:%Y-%m} {p['model']} {_filter_label(p['init_n'], p['fin_n'])}"
            for p in active_items[:4]
        ]
        active_suffix = " | ".join(active_labels) if active_labels else "none"
        recent = " ; ".join(self._recent_runs[:3]) if self._recent_runs else "none"
        self._cur_var.set(
            f"Runs: {finished}/{self._run_total} done, {running} running, {pending} pending "
            f"(OK {self._run_succeeded}, Error {self._run_failed}) | Active: {active_suffix} | Recent: {recent}"
        )
        self.update_idletasks()

    def _refresh_active_display(self):
        self._refresh_run_display()

    # ── Controls ───────────────────────────────────────────────────────────

    def _on_start(self):
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            messagebox.showerror("API Key Missing",
                                 "Enter your OpenAI API key in the toolbar before starting.")
            return
        self._persist_settings()
        self._stop_evt.clear()
        self._pause_evt.clear()
        n_threads = max(1, self._threads_var.get())

        # Snapshot filter settings at Start time
        filters = {
            "id_from":     self._id_from_var.get(),
            "id_to":       self._id_to_var.get(),
            "models":      {m for m, v in self._model_vars.items()  if v.get()},
            "temps":       {t for t, v in self._temp_vars.items()   if v.get()},
            "periods":     {p for p, v in self._period_vars.items() if v.get()},
            "filter_cfgs": {k for k, v in self._filt_vars.items()   if v.get()},
            "date_from":   datetime.strptime(self._date_from_var.get(), "%Y-%m"),
            "date_to":     datetime.strptime(self._date_to_var.get(),   "%Y-%m"),
        }

        target_runs = max(1, self._runs_var.get())
        scheduled_by_pid = self._scheduled_for_start(filters, target_runs)
        self._reset_run_tracking(scheduled_by_pid)

        self._btn_start.config(state="disabled")
        self._btn_retry.config(state="disabled")
        self._spin_threads.config(state="disabled")
        self._spin_runs.config(state="disabled")
        self._folder_entry.config(state="disabled")
        for w in self._filter_widgets:
            w.config(state="disabled")
        self._btn_pause.config(state="normal")
        self._btn_stop.config(state="normal")
        self._worker = RunnerThread(self.perms, self._q, key,
                                    self._stop_evt, self._pause_evt,
                                    n_threads=n_threads, filters=filters,
                                    target_runs=target_runs,
                                    results_subdir=self._folder_var.get())
        self._worker.start()
        skipped_by_filter = sum(1 for p in self.perms if not _perm_passes_filter(p, filters))
        self._log_append(
            f"[{datetime.now():%H:%M:%S}] Run started "
            f"({n_threads} thread(s), {self._run_total} queued run attempt(s), {target_runs} run(s)/perm, "
            f"{skipped_by_filter} perms excluded by filter)."
        )

    def _on_retry_errors(self):
        key = self._key_var.get().strip()
        if not key:
            messagebox.showerror("API Key Missing",
                                 "Enter your OpenAI API key in the toolbar before starting.")
            return
        self._persist_settings()

        # Snapshot current filters
        filters = {
            "id_from":     self._id_from_var.get(),
            "id_to":       self._id_to_var.get(),
            "models":      {m for m, v in self._model_vars.items()  if v.get()},
            "temps":       {t for t, v in self._temp_vars.items()   if v.get()},
            "periods":     {p for p, v in self._period_vars.items() if v.get()},
            "filter_cfgs": {k for k, v in self._filt_vars.items()   if v.get()},
            "date_from":   datetime.strptime(self._date_from_var.get(), "%Y-%m"),
            "date_to":     datetime.strptime(self._date_to_var.get(),   "%Y-%m"),
        }

        error_perms = [
            p for p in self.perms
            if p["status"] == ST_ERROR and _perm_passes_filter(p, filters)
        ]
        n_errors = len(error_perms)

        if n_errors == 0:
            messagebox.showinfo("No Errors", "No errored permutations match the current filters.")
            return

        if not messagebox.askyesno(
            "Retry Errors",
            f"Re-run {n_errors} errored permutation(s) matching the current filters?\n\n"
            "Each will get one fresh attempt."
        ):
            return

        self._reset_run_tracking(self._scheduled_for_retry(filters))

        self._stop_evt.clear()
        self._pause_evt.clear()
        n_threads = max(1, self._threads_var.get())

        self._btn_start.config(state="disabled")
        self._btn_retry.config(state="disabled")
        self._spin_threads.config(state="disabled")
        self._spin_runs.config(state="disabled")
        self._folder_entry.config(state="disabled")
        for w in self._filter_widgets:
            w.config(state="disabled")
        self._btn_pause.config(state="normal")
        self._btn_stop.config(state="normal")

        self._worker = RunnerThread(
            self.perms, self._q, key,
            self._stop_evt, self._pause_evt,
            n_threads=n_threads, filters=filters,
            target_runs=1, retry_errors_only=True,
            results_subdir=self._folder_var.get(),
        )
        self._worker.start()
        self._log_append(
            f"[{datetime.now():%H:%M:%S}] Retrying {n_errors} errored perm(s) "
            f"({n_threads} thread(s))."
        )

    def _on_pause(self):
        if self._pause_evt.is_set():
            self._pause_evt.clear()
            self._btn_pause.config(text="⏸  Pause")
            self._log_append(f"[{datetime.now():%H:%M:%S}] Resumed.")
        else:
            self._pause_evt.set()
            self._btn_pause.config(text="▶  Resume")
            self._log_append(f"[{datetime.now():%H:%M:%S}] Paused.")

    def _on_stop(self):
        self._stop_evt.set()
        self._pause_evt.clear()
        self._active_runs.clear()
        self._refresh_active_display()
        self._btn_start.config(state="normal")
        self._btn_retry.config(state="normal")
        self._spin_threads.config(state="normal")
        self._spin_runs.config(state="normal")
        self._folder_entry.config(state="normal")
        for w in self._filter_widgets:
            w.config(state="normal")
        self._btn_pause.config(state="disabled", text="⏸  Pause")
        self._btn_stop.config(state="disabled")
        self._log_append(f"[{datetime.now():%H:%M:%S}] Stopped.")
        self._on_save_excel()   # always save on stop

    def _persist_settings(self):
        config = dict(self._app_config)
        config["openai_api_key"] = self._key_var.get().strip()
        config["results_subdir"] = _safe_results_subdir(self._folder_var.get())
        self._folder_var.set(config["results_subdir"])
        try:
            _save_app_config(config)
            self._app_config = config
        except Exception as exc:
            self._log_append(f"  Settings save error: {exc}")

    def _on_save_excel(self):
        if self._excel_save_running:
            self._excel_save_requested = True
            return

        self._excel_save_running = True
        self._excel_save_requested = False
        threading.Thread(target=self._save_excel_worker, daemon=True).start()

    def _excel_python_executable(self) -> str:
        candidates = [
            os.environ.get("PERM_RUNNER_EXCEL_PYTHON"),
            r"C:\Users\jathi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
            sys.executable,
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return sys.executable

    def _save_excel_worker(self):
        started = datetime.now()
        py_exe = self._excel_python_executable()
        cmd = [py_exe, os.path.abspath(__file__), "--save-excel-from-json"]
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=300,
            )
            ok = proc.returncode == 0
            output = (proc.stdout or "").strip()
            error = (proc.stderr or "").strip()
            self._q.put({
                "kind": "excel_done",
                "ok": ok,
                "elapsed_s": round((datetime.now() - started).total_seconds(), 1),
                "output": output,
                "error": error,
                "python": py_exe,
            })
        except Exception as exc:
            self._q.put({
                "kind": "excel_done",
                "ok": False,
                "elapsed_s": round((datetime.now() - started).total_seconds(), 1),
                "output": "",
                "error": str(exc),
                "python": py_exe,
            })

    # ── Queue drain ────────────────────────────────────────────────────────

    def _drain_queue(self):
        while True:
            try:
                msg = self._q.get_nowait()
            except queue.Empty:
                break

            if msg["kind"] == "status":
                pid = msg["id"]
                p = self.perms[pid - 1]
                run_token = msg.get("run_token") or f"{pid}-legacy"
                if msg["status"] == ST_DONE:
                    invalid_reason = _invalid_result_reason(msg.get("result"))
                    if invalid_reason:
                        msg = dict(msg)
                        msg["status"] = ST_ERROR
                        msg["result"] = None
                        msg["note"] = msg.get("note", "") + f"  |  Invalid result: {invalid_reason[:120]}"

                if msg["status"] == ST_RUNNING:
                    if run_token not in self._active_runs:
                        self._run_started += 1
                    self._active_runs[run_token] = p
                elif msg["status"] in (ST_DONE, ST_ERROR):
                    self._active_runs.pop(run_token, None)
                    self._run_finished += 1
                    self._finished_by_pid[pid] = self._finished_by_pid.get(pid, 0) + 1
                    if msg["status"] == ST_DONE:
                        self._run_succeeded += 1
                    else:
                        self._run_failed += 1
                    label = (
                        f"#{pid:04d} {msg['status']} "
                        f"{p['period']}mo {p['start']:%Y-%m} {p['model']} "
                        f"{_filter_label(p['init_n'], p['fin_n'])}"
                    )
                    self._recent_runs.insert(0, label)
                    self._recent_runs = self._recent_runs[:8]

                p["status"]    = msg["status"]
                p["elapsed_s"] = msg.get("elapsed_s", 0)
                if msg.get("result") is not None and _is_valid_result(msg["result"]):
                    p["result"] = msg["result"]
                    p["results"].append(msg["result"])
                    n = len(p["results"])
                    p["note"] = f"{n} run(s)  |  " + msg.get("note", "")
                else:
                    p["note"] = msg.get("note", "")

                active_same_perm = sum(1 for active in self._active_runs.values() if active["id"] == pid)
                scheduled = self._scheduled_by_pid.get(pid, 0)
                finished = self._finished_by_pid.get(pid, 0)
                queued_same_perm = max(0, scheduled - finished - active_same_perm)
                if active_same_perm:
                    p["status"] = ST_RUNNING
                    p["note"] = f"{active_same_perm} active, {queued_same_perm} queued  |  {p['note']}"
                elif queued_same_perm:
                    p["status"] = ST_PENDING
                    p["note"] = f"{queued_same_perm} queued run attempt(s)  |  {p['note']}"
                self._update_row(pid)
                self._refresh_stats()
                self._refresh_active_display()

                if msg["status"] == ST_RUNNING:
                    pass

                elif msg["status"] in (ST_DONE, ST_ERROR):
                    self._done_count += 1
                    self._on_save_excel()

            elif msg["kind"] == "log":
                self._log_append(msg["text"])

            elif msg["kind"] == "plan":
                scheduled_by_pid = msg.get("scheduled_by_pid", {})
                if scheduled_by_pid != self._scheduled_by_pid:
                    self._reset_run_tracking(scheduled_by_pid)
                self._log_append(
                    f"[{datetime.now():%H:%M:%S}] Worker queued {msg.get('total', 0)} run attempt(s)."
                )

            elif msg["kind"] == "excel_done":
                self._excel_save_running = False
                if msg.get("ok"):
                    self._log_append(
                        f"[{datetime.now():%H:%M:%S}] Excel saved in child process "
                        f"({msg.get('elapsed_s')}s)."
                    )
                else:
                    self._log_append(
                        f"[{datetime.now():%H:%M:%S}] Excel save failed in child process "
                        f"({msg.get('elapsed_s')}s): {msg.get('error') or msg.get('output')}"
                    )
                if self._excel_save_requested:
                    self._on_save_excel()

            elif msg["kind"] == "done":
                self._active_runs.clear()
                self._on_stop()   # saves Excel via _on_stop
                self._cur_var.set("All runnable permutations complete.")
                self._log_append(f"[{datetime.now():%H:%M:%S}] All done!")
                messagebox.showinfo(
                    "Complete",
                    f"All runnable permutations have finished!\n\n"
                    f"Excel files: {RESULTS_DIR}/permutation_results_T*.xlsx\n"
                    f"JSONs: {RESULTS_DIR}/")

        self.after(250, self._drain_queue)

    # ── Log ────────────────────────────────────────────────────────────────

    def _log_append(self, text):
        self._log_box.configure(state="normal")
        self._log_box.insert("end", text + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    # ── Column sort ────────────────────────────────────────────────────────

    def _sort(self, col):
        rev  = self._sort_rev.get(col, False)
        data = [(self._tree.set(iid, col), iid)
                for iid in self._tree.get_children("")]

        def _key(val: str) -> str:
            clean = val.replace(" mo", "").split(" -> ")[0].split("->")[0].strip()
            try:
                return f"{float(clean):020.6f}"   # zero-padded so numeric sort works lexicographically
            except ValueError:
                return clean

        data.sort(key=lambda x: _key(str(x[0])), reverse=rev)
        for i, (_, iid) in enumerate(data):
            self._tree.move(iid, "", i)
        self._sort_rev[col] = not rev


# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__" and "--save-excel-from-json" in sys.argv:
    raise SystemExit(save_excel_from_json_files())

if __name__ == "__main__":
    app = App()
    app.mainloop()
