#!/usr/bin/env python3
"""
Investment Strategy Chart Visualizer
=====================================
Three chart types powered by the permutation_results_T0.3/0.4/0.5 Excel files:
  • Box Plots  – distribution of ROI (or other metrics) across dimensions
  • Bar Graphs – aggregated metric across dimensions with up to 4 dimensions
  • Pie Charts – sector allocation proportions by dimension
"""

import os
import re
import sys
import threading

import pandas as pd
import matplotlib

# ──────────────────────────────────────────────────────────────────────────────
#  Headless / GUI mode detection
# ──────────────────────────────────────────────────────────────────────────────
# The GUI needs tkinter and a Tk-capable matplotlib backend.  When the program
# is launched with a CLI export/headless flag — or when tkinter simply is not
# available (e.g. a server or CI box) — fall back to the non-interactive "Agg"
# backend and install lightweight stand-ins for the tkinter names so that the
# module (including the GUI class definitions) still imports cleanly.
_HEADLESS_FLAGS = ("--export-paper-figures", "--headless", "--chart")
_WANT_HEADLESS = any(f in sys.argv for f in _HEADLESS_FLAGS)

try:
    if _WANT_HEADLESS:
        raise ImportError("headless mode requested")
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False
    matplotlib.use("Agg")

    class _TkStub:
        """Minimal stand-in so tk/ttk-referencing class definitions import.

        Real attribute access (e.g. ``tk.BooleanVar``) only ever happens inside
        GUI methods, which are never called in headless mode — so raising there
        gives a clear error instead of silently misbehaving.
        """
        Tk = object       # base class for App(tk.Tk)
        Frame = object    # base class for BaseTab(ttk.Frame)

        def __getattr__(self, name):
            raise RuntimeError(
                "tkinter is unavailable — GUI features are disabled in "
                "headless mode (use --headless / --chart / --export-paper-figures)"
            )

    tk = ttk = messagebox = filedialog = _TkStub()
    FigureCanvasTkAgg = NavigationToolbar2Tk = None

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  Domain constants
# ──────────────────────────────────────────────────────────────────────────────

TICKER_INFO = {
    "GOOGL": ("Mega-Cap Tech",    "Alphabet"),
    "MSFT":  ("Mega-Cap Tech",    "Microsoft"),
    "NVDA":  ("Semiconductors",   "NVIDIA"),
    "AAPL":  ("Mega-Cap Tech",    "Apple"),
    "TSM":   ("Semiconductors",   "TSMC"),
    "CSCO":  ("Networking/Infra", "Cisco"),
    "AMD":   ("Semiconductors",   "AMD"),
    "TXN":   ("Semiconductors",   "Texas Instruments"),
    "MCHP":  ("Semiconductors",   "Microchip Tech"),
    "ASML":  ("Semiconductors",   "ASML"),
    "MU":    ("Semiconductors",   "Micron"),
    "META":  ("Mega-Cap Tech",    "Meta"),
    "AMZN":  ("Mega-Cap Tech",    "Amazon"),
    "CRM":   ("Enterprise SaaS",  "Salesforce"),
    "AVGO":  ("Semiconductors",   "Broadcom"),
    "IBM":   ("Networking/Infra", "IBM"),
    "ZM":    ("Enterprise SaaS",  "Zoom"),
    "PLTR":  ("Enterprise SaaS",  "Palantir"),
    "QCOM":  ("Semiconductors",   "Qualcomm"),
    "TSLA":  ("Mega-Cap Tech",    "Tesla"),
    "INTC":  ("Semiconductors",   "Intel"),
    "WMT":   ("Retail",           "Walmart"),
    "LRCX":  ("Semiconductors",   "Lam Research"),
}

SECTOR_COLORS = {
    "Mega-Cap Tech":    "#1B4F8A",
    "Semiconductors":   "#E07B39",
    "Networking/Infra": "#2E8B57",
    "Enterprise SaaS":  "#8B1A8B",
    "Retail":           "#8B6914",
    "Unknown":          "#888888",
}

DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
METRICS    = ["ROI %", "Return %", "Final Value ($)",
              "Sharpe Ratio", "Sortino Ratio", "Volatility %",
              "Max Drawdown %", "Calmar Ratio", "Benchmark Sharpe",
              "Alpha vs Bench %"]
CV_METRIC  = "Coefficient of Variation %"
BAR_METRICS = METRICS + [CV_METRIC]
DIMENSIONS = ["Temperature", "Model", "Filter", "Period (mo)", "Start"]
AGG_FUNCS  = ["Mean", "Median", "Max", "Min", "Sharpe Ratio", "CV %"]

PLOT_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC",
]
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "+++"]

# ──────────────────────────────────────────────────────────────────────────────
#  Colour theme
# ──────────────────────────────────────────────────────────────────────────────

BG       = "#1e1e2e"
PANEL_BG = "#252538"
ACCENT   = "#7c6af7"
FG       = "#e0e0e0"
ENTRY_BG = "#2e2e46"
BTN_BG   = "#5a4fcf"
BTN_ACT  = "#7c6af7"
SEP      = "#3a3a56"
CBKG     = "#14142a"   # figure background
CAXES    = "#1c1c30"   # axes background
CGRID    = "#2a2a44"   # grid lines
CTX      = "#b8b8d0"   # tick/label text

_DARK_THEME  = dict(CBKG="#14142a", CAXES="#1c1c30", CGRID="#2a2a44", CTX="#b8b8d0",
                    FG="#e0e0e0", PANEL_BG="#252538", SEP="#3a3a56")
_LIGHT_THEME = dict(CBKG="#ffffff", CAXES="#f5f5f5", CGRID="#cccccc", CTX="#2c2c2c",
                    FG="#111111", PANEL_BG="#eeeeee", SEP="#aaaaaa")
_light_mode  = [False]

def _toggle_theme(btn=None):
    global CBKG, CAXES, CGRID, CTX, FG, PANEL_BG, SEP
    _light_mode[0] = not _light_mode[0]
    src = _LIGHT_THEME if _light_mode[0] else _DARK_THEME
    CBKG, CAXES, CGRID = src["CBKG"], src["CAXES"], src["CGRID"]
    CTX, FG, PANEL_BG, SEP = src["CTX"], src["FG"], src["PANEL_BG"], src["SEP"]
    if btn is not None:
        btn.config(text="🌙 Dark Charts" if _light_mode[0] else "☀ Light Charts")

# ──────────────────────────────────────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    frames = []
    for temp_str in ["0.3", "0.4", "0.5"]:
        fpath = os.path.join(DATA_DIR, f"permutation_results_T{temp_str}.xlsx")
        if not os.path.exists(fpath):
            continue
        for model in ["gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5.1"]:
            try:
                df = pd.read_excel(fpath, sheet_name=model)
                df["Temperature"] = float(temp_str)
                df["Model"]       = model
                frames.append(df)
            except Exception:
                pass
    if not frames:
        raise FileNotFoundError(
            f"No permutation Excel files found in:\n{DATA_DIR}\n\n"
            "Expected: permutation_results_T0.3.xlsx, T0.4.xlsx, T0.5.xlsx"
        )
    master = pd.concat(frames, ignore_index=True)
    master = master[master["ROI %"].notna()].copy()
    master["Temperature"] = master["Temperature"].astype(float)
    master["Period (mo)"] = master["Period (mo)"].astype(int)
    master["Start"]       = master["Start"].astype(str)
    master["End"]         = master["End"].astype(str)
    master["Filter"]      = master["Filter"].astype(str)
    master["Model"]       = master["Model"].astype(str)
    return master

# ──────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_companies(s: str) -> dict:
    """'GOOGL (40%)  |  MSFT (35%)  |  NVDA (25%)' → {'GOOGL':40, 'MSFT':35, 'NVDA':25}"""
    if not isinstance(s, str):
        return {}
    out = {}
    for part in s.split("|"):
        m = re.match(r"\s*([A-Z]+)\s*\((\d+(?:\.\d+)?)%\)", part.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def agg_apply(series: pd.Series, fn: str) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    if fn == "Sharpe Ratio":
        std = s.std()
        return float(s.mean() / std) if std > 0 else float("nan")
    if fn == "CV %":
        mean = s.mean()
        return float(s.std() / abs(mean) * 100) if mean != 0 else float("nan")
    return {"Mean": s.mean, "Median": s.median, "Max": s.max, "Min": s.min}[fn]()


def dim_values(df: pd.DataFrame, dim: str) -> list:
    vals = df[dim].dropna().unique().tolist()
    try:
        return sorted(vals)
    except TypeError:
        return sorted(vals, key=str)


def mask(df: pd.DataFrame, dim: str, val) -> pd.DataFrame:
    return df[df[dim] == val]


def lbl(v) -> str:
    """Human-readable label for a dimension value."""
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def dim_label(dim: str, val) -> str:
    short = {"Temperature": "T", "Period (mo)": "Period", "Filter": "Filter",
             "Model": "Model", "Start": "Start"}
    return f"{short.get(dim, dim)}={lbl(val)}"


# ──────────────────────────────────────────────────────────────────────────────
#  Index benchmark helpers
# ──────────────────────────────────────────────────────────────────────────────

INDEX_SYMBOLS = {
    "S&P 500":    "^GSPC",
    "NASDAQ 100": "^NDX",
    "Dow Jones":  "^DJI",
}

# Distinct visual style for each index overlay
INDEX_STYLES = {
    "S&P 500":    {"color": "#FFD700", "linestyle": "--",  "marker": "D", "lw": 2.0},
    "NASDAQ 100": {"color": "#00E676", "linestyle": ":",   "marker": "^", "lw": 2.0},
    "Dow Jones":  {"color": "#FF6E6E", "linestyle": "-.",  "marker": "s", "lw": 2.0},
}

_index_cache: dict = {}   # {(symbol, start_yyyymm, end_yyyymm): float | None}


def _fetch_index_return(symbol: str, start_ym: str, end_ym: str) -> float | None:
    """
    Return % price-change for *symbol* from the first trading day of start_ym
    to the last trading day of end_ym (both 'YYYY-MM').
    Results are cached so each unique period is only fetched once.
    """
    key = (symbol, start_ym, end_ym)
    if key in _index_cache:
        return _index_cache[key]
    try:
        import yfinance as yf
        from datetime import datetime

        sy, sm = int(start_ym[:4]), int(start_ym[5:7])
        ey, em = int(end_ym[:4]),   int(end_ym[5:7])
        # Inclusive end: fetch up to first day of the month AFTER end_ym
        if em == 12:
            end_excl = datetime(ey + 1, 1, 1)
        else:
            end_excl = datetime(ey, em + 1, 1)

        hist = yf.Ticker(symbol).history(
            start=f"{sy:04d}-{sm:02d}-01",
            end=end_excl.strftime("%Y-%m-%d"),
            auto_adjust=True)

        if hist.empty or len(hist) < 2:
            _index_cache[key] = None
            return None

        ret = (hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100
        _index_cache[key] = round(float(ret), 4)
    except Exception:
        _index_cache[key] = None
    return _index_cache[key]


def _index_returns_for_subset(df: pd.DataFrame, symbol: str) -> list[float]:
    """All index returns for the unique (Start, End) pairs present in df."""
    seen, results = set(), []
    for _, row in df.iterrows():
        pair = (str(row["Start"]), str(row["End"]))
        if pair in seen:
            continue
        seen.add(pair)
        ret = _fetch_index_return(symbol, pair[0], pair[1])
        if ret is not None:
            results.append(ret)
    return results


def _index_return_by_start(df: pd.DataFrame, symbol: str) -> dict[str, float]:
    """
    For every unique Start value in df, return the mean index return across all
    (Start, End) pairs that share that Start.  {start_str: mean_return}
    """
    out: dict[str, float] = {}
    for sv in df["Start"].unique():
        vals = _index_returns_for_subset(df[df["Start"] == sv], symbol)
        if vals:
            out[str(sv)] = float(np.mean(vals))
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Per-perm detail loader (individual stock returns from JSON files)
# ──────────────────────────────────────────────────────────────────────────────

_perm_detail_cache: dict | None = None   # loaded lazily


def load_perm_details() -> dict:
    """
    Scan all perm_XXXX_*.json files in DATA_DIR and build:
        {perm_id (int): {
            "allocations":        {ticker: avg_pct},
            "individual_returns": {ticker: avg_pct_return},
        }}
    Multiple runs of the same perm_id are averaged together.
    Result is cached after the first call.
    """
    global _perm_detail_cache
    if _perm_detail_cache is not None:
        return _perm_detail_cache

    import json
    from collections import defaultdict

    raw: dict[int, list] = defaultdict(list)

    for fname in os.listdir(DATA_DIR):
        if not (fname.startswith("perm_") and fname.endswith(".json")):
            continue
        fpath = os.path.join(DATA_DIR, fname)
        try:
            with open(fpath) as f:
                d = json.load(f)
        except Exception:
            continue

        # Skip pure-error files
        if set(d.keys()) == {"error"}:
            continue

        ra      = d.get("return_analysis", {})
        ind_ret = ra.get("individual_returns", {})
        if not ind_ret:
            continue

        recs   = d.get("final_recommendations", [])
        allocs = {r["ticker"]: float(r["final_allocation"]) for r in recs
                  if "ticker" in r and "final_allocation" in r}

        # Resolve perm_id from metadata or filename
        meta = d.get("metadata", {})
        pid  = meta.get("perm_id")
        if pid is None:
            import re as _re
            m = _re.match(r"perm_(\d+)_", fname)
            pid = int(m.group(1)) if m else None
        if pid is None:
            continue

        raw[int(pid)].append({
            "allocations":        allocs,
            "individual_returns": {t: float(v) for t, v in ind_ret.items()},
        })

    # Average across runs of the same perm_id
    _perm_detail_cache = {}
    for pid, runs in raw.items():
        all_tickers = set().union(*(r["allocations"].keys() for r in runs),
                                  *(r["individual_returns"].keys() for r in runs))
        avg_alloc, avg_ret = {}, {}
        for ticker in all_tickers:
            av = [r["allocations"][ticker] for r in runs if ticker in r["allocations"]]
            rv = [r["individual_returns"][ticker]
                  for r in runs if ticker in r["individual_returns"]]
            if av:
                avg_alloc[ticker] = float(np.mean(av))
            if rv:
                avg_ret[ticker]   = float(np.mean(rv))
        _perm_detail_cache[pid] = {
            "allocations":        avg_alloc,
            "individual_returns": avg_ret,
        }

    return _perm_detail_cache


def compute_sector_contributions(df_subset: pd.DataFrame) -> dict:
    """
    For each row in df_subset, look up per-ticker allocation & return from the
    JSON detail files (matched by "Perm ID").

    Returns:
        {sector: {"allocation": float,   # normalised % of total allocation
                  "profit":     float,   # normalised % of total weighted return
                  "ratio":      float}}  # profit% / allocation%
                                         # >1 = sector punching above its weight
    """
    details = load_perm_details()

    sector_alloc:  dict[str, float] = {}   # raw sum of allocation %
    sector_profit: dict[str, float] = {}   # raw sum of weighted returns

    for _, row in df_subset.iterrows():
        pid = row.get("Perm ID")
        if pd.isna(pid):
            continue
        pid = int(pid)
        if pid not in details:
            continue

        allocs   = details[pid]["allocations"]
        ind_rets = details[pid]["individual_returns"]

        for ticker, alloc_pct in allocs.items():
            ret_pct = ind_rets.get(ticker, 0.0)
            sector  = TICKER_INFO.get(ticker, ("Unknown", ticker))[0]
            sector_alloc[sector]  = sector_alloc.get(sector, 0.0)  + alloc_pct
            # Weighted return contribution (alloc% × return%) – keeps sign
            sector_profit[sector] = sector_profit.get(sector, 0.0) + alloc_pct * ret_pct

    if not sector_alloc:
        return {}

    total_alloc  = sum(sector_alloc.values())
    total_profit = sum(sector_profit.values())   # can be negative

    result: dict[str, dict] = {}
    for sector in sector_alloc:
        norm_alloc  = sector_alloc[sector]  / total_alloc  * 100 if total_alloc  else 0.0
        norm_profit = (sector_profit[sector] / total_profit * 100
                       if total_profit and total_profit != 0 else 0.0)
        ratio = norm_profit / norm_alloc if norm_alloc else 0.0
        result[sector] = {
            "allocation": norm_alloc,
            "profit":     norm_profit,
            "ratio":      ratio,
        }
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  BaseTab
# ──────────────────────────────────────────────────────────────────────────────

class BaseTab(ttk.Frame):
    def __init__(self, parent, df: pd.DataFrame):
        super().__init__(parent)
        self.master_df = df
        self.fig: plt.Figure | None = None
        self.canvas: FigureCanvasTkAgg | None = None
        self._filter_check_vars:   dict[str, dict[str, tk.BooleanVar]] = {}
        self._filter_summary_lbls: dict[str, tk.Label]                 = {}
        self._build()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        # ── Left scrollable controls panel ────────────────────────────────────
        ctrl_outer = tk.Frame(self, bg=PANEL_BG, width=330)
        ctrl_outer.pack(side="left", fill="y")
        ctrl_outer.pack_propagate(False)

        self._ctrl_canvas = tk.Canvas(ctrl_outer, bg=PANEL_BG, highlightthickness=0)
        sb = ttk.Scrollbar(ctrl_outer, orient="vertical",
                           command=self._ctrl_canvas.yview)
        self.ctrl_frame = tk.Frame(self._ctrl_canvas, bg=PANEL_BG)
        self.ctrl_frame.bind(
            "<Configure>",
            lambda e: self._ctrl_canvas.configure(
                scrollregion=self._ctrl_canvas.bbox("all")))
        self._ctrl_canvas.create_window((0, 0), window=self.ctrl_frame, anchor="nw")
        self._ctrl_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._ctrl_canvas.pack(side="left", fill="both", expand=True)
        self._ctrl_canvas.bind_all(
            "<MouseWheel>",
            lambda e: self._ctrl_canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        # ── Right chart area ───────────────────────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        btn_bar = tk.Frame(right, bg=BG)
        btn_bar.pack(side="bottom", fill="x", padx=6, pady=4)

        tk.Button(btn_bar, text="Export Image", command=self._export,
                  bg=BTN_BG, fg=FG, activebackground=BTN_ACT,
                  relief="flat", padx=16, pady=5,
                  font=("Segoe UI", 10, "bold"), cursor="hand2").pack(side="right")

        self.status_var = tk.StringVar(value="Ready — configure controls and press Generate")
        tk.Label(btn_bar, textvariable=self.status_var,
                 bg=BG, fg="#888888", font=("Segoe UI", 8)).pack(side="left", padx=4)

        self.chart_frame = tk.Frame(right, bg=CBKG)
        self.chart_frame.pack(fill="both", expand=True, padx=6, pady=(6, 0))

        # Build controls
        self._build_filters()
        self._build_chart_controls()

        # Generate button pinned at bottom of ctrl_frame
        tk.Button(self.ctrl_frame, text="Generate Chart",
                  command=self._safe_generate,
                  bg=ACCENT, fg="white", activebackground=BTN_ACT,
                  relief="flat", padx=16, pady=9,
                  font=("Segoe UI", 11, "bold"),
                  cursor="hand2").pack(fill="x", padx=10, pady=(18, 12))

    # ── Control helpers ────────────────────────────────────────────────────────

    def _sec(self, title: str):
        tk.Label(self.ctrl_frame, text=title, bg=PANEL_BG, fg=ACCENT,
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(
            fill="x", padx=10, pady=(12, 2))
        tk.Frame(self.ctrl_frame, bg=ACCENT, height=1).pack(fill="x", padx=10, pady=(0, 6))

    def _lbl(self, text: str):
        tk.Label(self.ctrl_frame, text=text, bg=PANEL_BG, fg=FG,
                 font=("Segoe UI", 9), anchor="w").pack(fill="x", padx=14, pady=(0, 1))

    def _sep(self):
        tk.Frame(self.ctrl_frame, bg=SEP, height=1).pack(fill="x", padx=10, pady=6)

    def _combo(self, values: list, default=None) -> tuple:
        var = tk.StringVar()
        cb = ttk.Combobox(self.ctrl_frame, textvariable=var, values=values,
                          state="readonly", font=("Segoe UI", 9))
        if default is not None and default in values:
            var.set(default)
        elif values:
            var.set(values[0])
        cb.pack(fill="x", padx=10, pady=(0, 6))
        return cb, var

    # ── Data filter controls ───────────────────────────────────────────────────

    def _build_filters(self):
        self._sec("Data Pre-Filters")
        self._lbl("Expand a dimension to include/exclude values.")

        for dim in DIMENSIONS:
            vals = [lbl(v) for v in dim_values(self.master_df, dim)]

            # One BooleanVar per value — all True (= include) by default
            cvars = {v: tk.BooleanVar(value=True) for v in vals}
            self._filter_check_vars[dim] = cvars

            # ── Header row (always visible, acts as toggle) ────────────
            hdr = tk.Frame(self.ctrl_frame, bg=PANEL_BG, cursor="hand2")
            hdr.pack(fill="x", padx=6, pady=(6, 0))

            arrow = tk.Label(hdr, text="▶", bg=PANEL_BG, fg=ACCENT,
                             font=("Segoe UI", 9, "bold"), width=2,
                             cursor="hand2")
            arrow.pack(side="left")

            dim_lbl = tk.Label(hdr, text=dim, bg=PANEL_BG, fg=FG,
                               font=("Segoe UI", 9, "bold"), cursor="hand2")
            dim_lbl.pack(side="left")

            summary = tk.Label(hdr, text="(all)", bg=PANEL_BG, fg=CTX,
                               font=("Segoe UI", 8), cursor="hand2")
            summary.pack(side="right", padx=4)
            self._filter_summary_lbls[dim] = summary

            # ── Collapsible content frame ──────────────────────────────
            content = tk.Frame(self.ctrl_frame, bg=ENTRY_BG)

            # Quick "All" / "None" buttons
            btn_row = tk.Frame(content, bg=ENTRY_BG)
            btn_row.pack(fill="x", padx=4, pady=(3, 1))
            tk.Button(btn_row, text="✓ All",
                      command=self._make_select_all(dim),
                      bg=ENTRY_BG, fg=ACCENT, relief="flat",
                      font=("Segoe UI", 7, "bold"),
                      cursor="hand2").pack(side="left", padx=(2, 0))
            tk.Button(btn_row, text="✗ None",
                      command=self._make_select_none(dim),
                      bg=ENTRY_BG, fg="#e06c75", relief="flat",
                      font=("Segoe UI", 7, "bold"),
                      cursor="hand2").pack(side="left", padx=2)

            # One checkbox per unique value
            for v_str, bvar in cvars.items():
                tk.Checkbutton(
                    content, text=v_str, variable=bvar,
                    command=lambda d=dim: self._update_filter_summary(d),
                    bg=ENTRY_BG, fg=FG, selectcolor=PANEL_BG,
                    activebackground=ENTRY_BG, activeforeground=FG,
                    font=("Segoe UI", 9), anchor="w", cursor="hand2",
                ).pack(fill="x", padx=10, pady=1)

            # Toggle expand / collapse (mutable list as closure state)
            _state = [False]

            def _make_toggle(arr=arrow, cont=content, hd=hdr, st=_state):
                def _toggle(*_):
                    if st[0]:
                        cont.pack_forget()
                        arr.config(text="▶")
                        st[0] = False
                    else:
                        # `after=hd` inserts directly below this section's header,
                        # not at the end of all packed widgets in ctrl_frame
                        cont.pack(after=hd, fill="x", padx=6, pady=(0, 6))
                        arr.config(text="▼")
                        st[0] = True
                return _toggle

            toggle = _make_toggle()
            for w in (hdr, arrow, dim_lbl, summary):
                w.bind("<Button-1>", toggle)

        self._sep()

    # ── Filter helper methods ─────────────────────────────────────────────────

    def _make_select_all(self, dim: str):
        def _fn():
            for bv in self._filter_check_vars[dim].values():
                bv.set(True)
            self._update_filter_summary(dim)
        return _fn

    def _make_select_none(self, dim: str):
        def _fn():
            for bv in self._filter_check_vars[dim].values():
                bv.set(False)
            self._update_filter_summary(dim)
        return _fn

    def _update_filter_summary(self, dim: str):
        cvars   = self._filter_check_vars.get(dim, {})
        checked = [v for v, bv in cvars.items() if bv.get()]
        total   = len(cvars)
        widget  = self._filter_summary_lbls.get(dim)
        if widget is None:
            return
        if len(checked) == total:
            widget.config(text="(all)", fg=CTX)
        elif len(checked) == 0:
            widget.config(text="(none)", fg="#e06c75")
        elif len(checked) <= 2:
            widget.config(text=", ".join(checked), fg=ACCENT)
        else:
            widget.config(text=f"{len(checked)}/{total}", fg=ACCENT)

    def _build_chart_controls(self):
        """Override in subclass."""

    # ── Filtered dataframe ────────────────────────────────────────────────────

    def get_df(self) -> pd.DataFrame:
        df = self.master_df.copy()
        for dim, cvars in self._filter_check_vars.items():
            checked = {v for v, bv in cvars.items() if bv.get()}
            if len(checked) == len(cvars):
                continue          # all selected → no filtering
            if not checked:
                return df.iloc[0:0]   # nothing selected → empty
            df = df[df[dim].apply(lambda x: lbl(x) in checked)]
        return df

    # ── Chart display ─────────────────────────────────────────────────────────

    def _show_chart(self, fig: plt.Figure):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        for w in self.chart_frame.winfo_children():
            w.destroy()
        self.fig = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.draw()
        tb = NavigationToolbar2Tk(self.canvas, self.chart_frame)
        tb.configure(bg=PANEL_BG)
        tb.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _export(self):
        if self.fig is None:
            messagebox.showinfo("No Chart", "Generate a chart first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"),
                       ("SVG Vector", "*.svg")])
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight",
                             facecolor=self.fig.get_facecolor())
            messagebox.showinfo("Saved", f"Chart saved to:\n{path}")

    def _safe_generate(self):
        try:
            self._status("Generating…")
            self._generate()
        except Exception as exc:
            messagebox.showerror("Chart Error", str(exc))
            self._status(f"Error: {exc}")

    def _generate(self):
        pass

    def _status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    # ── Index benchmark controls ───────────────────────────────────────────────

    def _build_index_controls(self, note: str = ""):
        """
        Adds a checkmark row for S&P 500 / NASDAQ 100 / Dow Jones.
        Sets self.idx_vars = {name: BooleanVar}.
        Call from subclass _build_chart_controls().
        """
        self._sec("Index Benchmarks")
        if note:
            self._lbl(note)
        self.idx_vars: dict[str, tk.BooleanVar] = {}
        row = tk.Frame(self.ctrl_frame, bg=PANEL_BG)
        row.pack(fill="x", padx=10, pady=(0, 8))
        for name, style in INDEX_STYLES.items():
            var = tk.BooleanVar(value=False)
            self.idx_vars[name] = var
            tk.Checkbutton(
                row, text=name, variable=var,
                bg=PANEL_BG, fg=style["color"],
                selectcolor=ENTRY_BG,
                activebackground=PANEL_BG, activeforeground=style["color"],
                font=("Segoe UI", 9, "bold"),
                cursor="hand2"
            ).pack(side="left", padx=(0, 6))

    def _checked_indices(self) -> list[str]:
        """Return list of index names whose checkbox is ticked."""
        return [n for n, v in getattr(self, "idx_vars", {}).items() if v.get()]

    # ── Axes styling ──────────────────────────────────────────────────────────

    def _style_ax(self, ax):
        ax.set_facecolor(CAXES)
        ax.tick_params(colors=CTX, labelsize=10)
        ax.xaxis.label.set_color(CTX)
        ax.yaxis.label.set_color(CTX)
        ax.title.set_color(CTX)
        for sp in ax.spines.values():
            sp.set_edgecolor(SEP)
        ax.grid(True, color=CGRID, linewidth=0.5, alpha=0.8, zorder=0)
        ax.set_axisbelow(True)


# ──────────────────────────────────────────────────────────────────────────────
#  Box Plot Tab
# ──────────────────────────────────────────────────────────────────────────────

class BoxPlotTab(BaseTab):

    def _build_chart_controls(self):
        self._sec("Metric")
        self._lbl("Quantitative value to plot:")
        _, self.metric_var = self._combo(METRICS, "ROI %")

        self._sec("Plot Style")
        self._lbl("How to visualize the distribution:")
        _, self.plot_style_var = self._combo(
            ["Box Plot", "Violin", "Box + Violin"], "Box Plot")

        self._sec("Orientation")
        self.horiz_var = tk.BooleanVar(value=False)
        _hz_row = tk.Frame(self.ctrl_frame, bg=PANEL_BG)
        _hz_row.pack(fill="x", padx=4, pady=(2, 8))
        tk.Checkbutton(
            _hz_row, text="Horizontal boxes",
            variable=self.horiz_var,
            bg=PANEL_BG, fg=FG, selectcolor=ENTRY_BG,
            activebackground=PANEL_BG, activeforeground=ACCENT,
            font=("Segoe UI", 9, "bold"), cursor="hand2"
        ).pack(side="left", padx=(10, 0))

        self._sec("Dimension 1 — Sub-plots")
        self._lbl("Create a separate subplot for each value of:")
        _, self.dim1_var = self._combo(["(none)"] + DIMENSIONS, "Temperature")

        self._sec("Dimension 2 — Box Groups")
        self._lbl("Within each subplot, one box per value of:")
        _, self.dim2_var = self._combo(DIMENSIONS, "Model")

        self._build_index_controls(
            note="Drawn as a dashed line at the mean index\n"
                 "return across all periods in each subplot.\n"
                 "(Only meaningful for ROI % / Return %)"
        )

        self._sec("Baseline Adjustment")
        self._lbl("Subtract a market index return from each\n"
                  "data point — values then represent alpha\n"
                  "(out/under-performance vs the index):")
        self.baseline_var = tk.BooleanVar(value=False)
        _bl_row = tk.Frame(self.ctrl_frame, bg=PANEL_BG, cursor="hand2")
        _bl_row.pack(fill="x", padx=4, pady=(2, 6))
        tk.Checkbutton(
            _bl_row, text="Subtract index baseline",
            variable=self.baseline_var,
            bg=PANEL_BG, fg=FG, selectcolor=ENTRY_BG,
            activebackground=PANEL_BG, activeforeground=ACCENT,
            font=("Segoe UI", 9, "bold"), cursor="hand2"
        ).pack(side="left", padx=(10, 0))
        _bl_row.bind("<Button-1>",
                     lambda e: self.baseline_var.set(not self.baseline_var.get()))
        self._lbl("Baseline index to subtract:")
        _, self.baseline_index_var = self._combo(
            list(INDEX_SYMBOLS.keys()), "S&P 500"
        )

    def _generate(self):
        metric   = self.metric_var.get()
        dim1_raw = self.dim1_var.get()
        dim2     = self.dim2_var.get()
        dim1     = None if dim1_raw == "(none)" else dim1_raw

        if dim1 and dim1 == dim2:
            messagebox.showwarning("Duplicate Dimensions",
                                   "Dimension 1 and Dimension 2 must be different.")
            return

        df = self.get_df()
        if df.empty:
            messagebox.showwarning("No Data", "No rows match the current filters.")
            return

        # ── Baseline adjustment setup ──────────────────────────────────────────
        use_baseline    = self.baseline_var.get()
        baseline_name   = self.baseline_index_var.get()
        baseline_symbol = INDEX_SYMBOLS.get(baseline_name, "")
        # Per-row adjustment keyed by pandas index: {row_index: index_return_float}
        baseline_adj: dict = {}
        if use_baseline and metric in ("ROI %", "Return %") and baseline_symbol:
            self._status(
                f"Fetching {baseline_name} baseline return for each period…")
            for i, row in df.iterrows():
                ret = _fetch_index_return(
                    baseline_symbol, str(row["Start"]), str(row["End"]))
                baseline_adj[i] = ret if ret is not None else 0.0

        plot_style = self.plot_style_var.get()

        d1_vals = dim_values(df, dim1) if dim1 else [None]
        d2_vals = dim_values(df, dim2)
        n_plots = len(d1_vals)
        cols    = min(n_plots, 3)
        rows    = (n_plots + cols - 1) // cols

        fig, axes_arr = plt.subplots(
            rows, cols,
            figsize=(max(6 * cols, 9), max(5 * rows, 5)),
            squeeze=False)
        fig.patch.set_facecolor(CBKG)

        colors = [PLOT_COLORS[i % len(PLOT_COLORS)] for i in range(len(d2_vals))]

        for idx, d1_val in enumerate(d1_vals):
            ax  = axes_arr[idx // cols][idx % cols]
            sub = mask(df, dim1, d1_val) if dim1 else df

            data_groups, labels, has_data = [], [], []
            for d2_val in d2_vals:
                grp = mask(sub, dim2, d2_val)
                if baseline_adj:
                    # Subtract per-row index return from each metric value
                    vals = []
                    for i, row in grp.iterrows():
                        v = row[metric]
                        if pd.isna(v):
                            continue
                        vals.append(float(v) - baseline_adj.get(i, 0.0))
                else:
                    vals = grp[metric].dropna().tolist()
                data_groups.append(vals if vals else [])
                labels.append(lbl(d2_val))
                has_data.append(bool(vals))

            self._style_ax(ax)
            positions = list(range(1, len(d2_vals) + 1))

            # Only pass non-empty groups
            nonempty_pos    = [p for p, h in zip(positions, has_data) if h]
            nonempty_groups = [g for g, h in zip(data_groups, has_data) if h]
            nonempty_colors = [c for c, h in zip(colors, has_data) if h]

            # ── Violin layer ───────────────────────────────────────────────────
            if plot_style in ("Violin", "Box + Violin") and nonempty_groups:
                # violinplot needs at least 2 distinct values per group
                vio_pos  = [p for p, g in zip(nonempty_pos, nonempty_groups)
                            if len(set(g)) >= 2]
                vio_data = [g for g in nonempty_groups if len(set(g)) >= 2]
                vio_cols = [c for c, g in zip(nonempty_colors, nonempty_groups)
                            if len(set(g)) >= 2]
                if vio_data:
                    parts = ax.violinplot(
                        vio_data, positions=vio_pos,
                        showmedians=True, showextrema=True,
                        widths=0.75)
                    for pc, color in zip(parts["bodies"], vio_cols):
                        pc.set_facecolor(color)
                        pc.set_alpha(0.45)
                        pc.set_edgecolor("#ffffff50")
                    for pname in ("cbars", "cmaxes", "cmins", "cmedians"):
                        if pname in parts:
                            parts[pname].set_color(CTX)
                            parts[pname].set_linewidth(1.2)
                    if "cmedians" in parts:
                        parts["cmedians"].set_color("#ffffff")
                        parts["cmedians"].set_linewidth(2)

            # ── Box-plot layer ─────────────────────────────────────────────────
            if plot_style in ("Box Plot", "Box + Violin") and nonempty_groups:
                # Narrower boxes when overlaid on violins
                bw = 0.30 if plot_style == "Box + Violin" else 0.55
                bp = ax.boxplot(nonempty_groups, positions=nonempty_pos,
                                patch_artist=True, widths=bw,
                                showfliers=(plot_style == "Box Plot"), zorder=3)
                for patch, color in zip(bp["boxes"], nonempty_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.85 if plot_style == "Box Plot" else 0.60)
                for med in bp["medians"]:
                    med.set_color("#ffffff")
                    med.set_linewidth(2)
                for elem in ("whiskers", "caps"):
                    for item in bp[elem]:
                        item.set_color(CTX)
                        item.set_linewidth(1.2)
                for fl in bp["fliers"]:
                    fl.set_markerfacecolor(CTX)
                    fl.set_markeredgecolor(CTX)
                    fl.set_markersize(3)

            # Zero line — when baseline-adjusted this represents "matched the index"
            ax.axhline(0, color="#666688", linewidth=0.9, linestyle="--", alpha=0.7)
            if baseline_adj:
                ax.text(len(d2_vals) + 0.6, 0, "= index",
                        color="#666688", fontsize=9, va="center",
                        clip_on=False)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=20, ha="right", color=CTX, fontsize=10)
            ax.yaxis.set_tick_params(labelcolor=CTX)

            # Y-axis label reflects whether baseline is subtracted
            if baseline_adj:
                ylabel = f"{metric}  −  {baseline_name}"
            else:
                ylabel = metric
            ax.set_ylabel(ylabel, color=CTX, fontsize=11)

            title = dim_label(dim1, d1_val) if dim1 else "All Data"
            ax.set_title(title, color=FG, fontsize=12, fontweight="bold", pad=8)

            # Annotate n per box
            for pos, grp in zip(positions, data_groups):
                if grp:
                    ax.text(pos, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else min(grp) - 1,
                            f"n={len(grp)}", ha="center", va="top",
                            color=CTX, fontsize=9, alpha=0.7)

            # ── Index benchmark overlay ────────────────────────────────────────
            # When baseline is active the overlay values are shifted by the mean
            # baseline return across this subplot's rows so everything stays on
            # the same relative scale.
            checked = self._checked_indices()
            if checked and metric in ("ROI %", "Return %"):
                # Mean baseline for this subplot (0 when no baseline adjustment)
                if baseline_adj:
                    sub_bl_vals = [baseline_adj.get(i, 0.0) for i in sub.index]
                    sub_bl_mean = float(np.mean(sub_bl_vals)) if sub_bl_vals else 0.0
                else:
                    sub_bl_mean = 0.0

                for idx_name in checked:
                    self._status(f"Fetching {idx_name} data…")
                    symbol   = INDEX_SYMBOLS[idx_name]
                    style    = INDEX_STYLES[idx_name]
                    idx_rets = _index_returns_for_subset(sub, symbol)
                    if not idx_rets:
                        continue
                    # Shift by baseline mean so line sits on the adjusted scale
                    idx_mean = float(np.mean(idx_rets)) - sub_bl_mean
                    idx_min  = float(np.min(idx_rets))  - sub_bl_mean
                    idx_max  = float(np.max(idx_rets))  - sub_bl_mean
                    line_lbl = (f"{idx_name}  μ={idx_mean:+.1f}%"
                                if baseline_adj else
                                f"{idx_name}  μ={idx_mean:.1f}%")
                    # Mean line
                    ax.axhline(idx_mean,
                               color=style["color"], linestyle=style["linestyle"],
                               linewidth=style["lw"], alpha=0.90, zorder=6,
                               label=line_lbl)
                    # Shaded range (min–max across the periods in this subplot)
                    ax.axhspan(idx_min, idx_max,
                               color=style["color"], alpha=0.07, zorder=2)
                    # Annotate mean value at right edge
                    ann_txt = (f"{idx_name}\n{idx_mean:+.1f}%"
                               if baseline_adj else
                               f"{idx_name}\n{idx_mean:.1f}%")
                    ax.annotate(
                        ann_txt,
                        xy=(len(d2_vals) + 0.55, idx_mean),
                        xycoords="data", color=style["color"],
                        fontsize=9, va="center",
                        annotation_clip=False)

        # Hide unused axes
        for idx in range(len(d1_vals), rows * cols):
            a = axes_arr[idx // cols][idx % cols]
            a.set_visible(False)

        # Legend — strategy boxes + optional index entries
        leg_handles: list = [mpatches.Patch(color=c) for c in colors]
        leg_labels:  list = [lbl(v) for v in d2_vals]
        checked = self._checked_indices()
        if checked and metric in ("ROI %", "Return %"):
            adj_note = "  (adj.)" if baseline_adj else ""
            for idx_name in checked:
                st = INDEX_STYLES[idx_name]
                leg_handles.append(mpatches.Patch(facecolor=st["color"],
                                                  edgecolor=st["color"]))
                leg_labels.append(f"{idx_name} mean ± range{adj_note}")

        fig.legend(leg_handles, leg_labels,
                   title=dim2 + (" + indices" if checked else ""),
                   loc="lower center",
                   ncol=min(len(leg_handles), 6),
                   facecolor=PANEL_BG, labelcolor=FG,
                   title_fontproperties={"size": 11, "weight": "bold"},
                   framealpha=0.9, edgecolor=SEP,
                   bbox_to_anchor=(0.5, 0))

        suptitle = f"Distribution of {metric}"
        if baseline_adj:
            suptitle += f"  (relative to {baseline_name})"
        if dim1:
            suptitle += f"  ·  split by {dim1}  &  {dim2}"
        else:
            suptitle += f"  ·  grouped by {dim2}"
        fig.suptitle(suptitle, color=FG, fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout(rect=[0, 0.07, 1, 1])
        self._show_chart(fig)
        n_total = len(df)
        bl_note = f"  ·  baseline: {baseline_name}" if baseline_adj else ""
        self._status(f"Box plots generated  ·  {n_total} total rows{bl_note}")


# ──────────────────────────────────────────────────────────────────────────────
#  Bar Graph Tab
# ──────────────────────────────────────────────────────────────────────────────

class BarGraphTab(BaseTab):

    def _build_chart_controls(self):
        self._sec("Metric & Aggregation")
        self._lbl("Quantitative value:")
        _, self.metric_var = self._combo(BAR_METRICS, "ROI %")
        self._lbl("Aggregation function:")
        _, self.agg_var = self._combo(AGG_FUNCS, "Mean")
        self._lbl("Coefficient of Variation % uses ROI % and ignores aggregation.")

        self._sec("Dimension 1 — Separate Graphs")
        self._lbl("(none) = single graph")
        _, self.dim1_var = self._combo(["(none)"] + DIMENSIONS, "(none)")

        self._sec("Dimension 2 — X-Axis Groups")
        self._lbl("Required: x-axis tick grouping:")
        _, self.dim2_var = self._combo(DIMENSIONS, "Start")

        self._sec("Dimension 3 — Bar Color")
        self._lbl("(none) = single color series:")
        _, self.dim3_var = self._combo(["(none)"] + DIMENSIONS, "Model")

        self._sec("Dimension 4 — Bar Hatch Pattern")
        self._lbl("(none) = no hatch:")
        _, self.dim4_var = self._combo(["(none)"] + DIMENSIONS, "(none)")

        self._build_index_controls(
            note="If X-axis = Start → per-period line.\n"
                 "Otherwise → horizontal mean line.\n"
                 "(Only meaningful for ROI % / Return %)"
        )

    def _generate(self):
        metric   = self.metric_var.get()
        agg_name = self.agg_var.get()
        is_cv_metric = metric == CV_METRIC
        is_cv_chart = is_cv_metric or agg_name == "CV %"
        value_metric = "ROI %" if is_cv_metric else metric
        value_agg = "CV %" if is_cv_metric else agg_name
        dim1 = None if self.dim1_var.get() == "(none)" else self.dim1_var.get()
        dim2 = self.dim2_var.get()
        dim3 = None if self.dim3_var.get() == "(none)" else self.dim3_var.get()
        dim4 = None if self.dim4_var.get() == "(none)" else self.dim4_var.get()

        active = [d for d in [dim1, dim2, dim3, dim4] if d]
        if len(active) != len(set(active)):
            messagebox.showwarning("Duplicate Dimensions",
                                   "All selected dimensions must be different.")
            return

        df = self.get_df()
        if df.empty:
            messagebox.showwarning("No Data", "No rows match the current filters.")
            return

        d1_vals = dim_values(df, dim1) if dim1 else [None]
        d2_vals = dim_values(df, dim2)
        d3_vals = dim_values(df, dim3) if dim3 else [None]
        d4_vals = dim_values(df, dim4) if dim4 else [None]

        n_plots = len(d1_vals)
        cols    = min(n_plots, 2)
        rows    = (n_plots + cols - 1) // cols

        fig, axes_arr = plt.subplots(
            rows, cols,
            figsize=(max(9 * cols, 11), max(5 * rows, 5)),
            squeeze=False)
        fig.patch.set_facecolor(CBKG)

        n_series      = len(d3_vals) * len(d4_vals)
        bar_width     = max(0.05, min(0.7 / n_series, 0.28))
        d3_colors     = {v: PLOT_COLORS[i % len(PLOT_COLORS)] for i, v in enumerate(d3_vals)}
        d4_hatches    = {v: HATCHES[i % len(HATCHES)] for i, v in enumerate(d4_vals)}
        x_positions   = np.arange(len(d2_vals))

        for plot_idx, d1_val in enumerate(d1_vals):
            ax   = axes_arr[plot_idx // cols][plot_idx % cols]
            sub1 = mask(df, dim1, d1_val) if dim1 else df
            self._style_ax(ax)

            series_idx = 0
            for d3_val in d3_vals:
                for d4_val in d4_vals:
                    heights = []
                    for d2_val in d2_vals:
                        grp = mask(sub1, dim2, d2_val)
                        if dim3:
                            grp = mask(grp, dim3, d3_val)
                        if dim4:
                            grp = mask(grp, dim4, d4_val)
                        h = agg_apply(grp[value_metric].dropna(), value_agg)
                        heights.append(np.nan if np.isnan(h) else h)

                    offset = (series_idx - n_series / 2 + 0.5) * bar_width
                    color  = d3_colors[d3_val]
                    hatch  = d4_hatches[d4_val]

                    label_parts = []
                    if dim3:
                        label_parts.append(lbl(d3_val))
                    if dim4:
                        label_parts.append(lbl(d4_val))
                    label = " / ".join(label_parts) if label_parts else ""

                    ax.bar(x_positions + offset, heights,
                           width=bar_width, color=color, hatch=hatch,
                           edgecolor="#ffffff30", linewidth=0.5,
                           label=label, alpha=0.86, zorder=3)
                    series_idx += 1

            ax.axhline(0, color="#666688", linewidth=0.9, linestyle="--", alpha=0.7)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([lbl(v) for v in d2_vals],
                               rotation=30, ha="right", color=CTX, fontsize=10)
            ax.yaxis.set_tick_params(labelcolor=CTX)
            ylabel = f"Coefficient of Variation of {value_metric} (%)" if is_cv_chart else f"{agg_name}  {metric}"
            ax.set_ylabel(ylabel, color=CTX, fontsize=11)
            title = dim_label(dim1, d1_val) if dim1 else "All Data"
            ax.set_title(title, color=FG, fontsize=12, fontweight="bold", pad=8)

            # ── Index benchmark overlay ────────────────────────────────────────
            checked = self._checked_indices()
            if checked and not is_cv_chart and metric in ("ROI %", "Return %"):
                for idx_name in checked:
                    self._status(f"Fetching {idx_name} data…")
                    symbol = INDEX_SYMBOLS[idx_name]
                    style  = INDEX_STYLES[idx_name]

                    if dim2 == "Start":
                        # Per-period line: one point per x-tick
                        by_start = _index_return_by_start(sub1, symbol)
                        ys = [by_start.get(str(v), None) for v in d2_vals]
                        valid_xs = [x for x, y in zip(x_positions, ys) if y is not None]
                        valid_ys = [y for y in ys if y is not None]
                        if valid_xs:
                            ax.plot(valid_xs, valid_ys,
                                    color=style["color"],
                                    linestyle=style["linestyle"],
                                    linewidth=style["lw"],
                                    marker=style["marker"],
                                    markersize=7,
                                    zorder=7,
                                    label=idx_name)
                    else:
                        # Single horizontal mean line across all periods in this subplot
                        idx_rets = _index_returns_for_subset(sub1, symbol)
                        if idx_rets:
                            idx_mean = float(np.mean(idx_rets))
                            ax.axhline(idx_mean,
                                       color=style["color"],
                                       linestyle=style["linestyle"],
                                       linewidth=style["lw"],
                                       alpha=0.90, zorder=7,
                                       label=f"{idx_name}  μ={idx_mean:.1f}%")

            # Re-draw legend (bar series + any index lines)
            # Filter out entries with blank labels that come from unlabelled bars
            _h, _l = ax.get_legend_handles_labels()
            _h, _l = zip(*[(h, l) for h, l in zip(_h, _l) if l]) if _h else ([], [])
            if _h:
                leg_title = " / ".join(d for d in [dim3, dim4] if d)
                ax.legend(list(_h), list(_l),
                          facecolor=PANEL_BG, labelcolor=FG, fontsize=9,
                          framealpha=0.9, edgecolor=SEP,
                          title=leg_title or None, title_fontsize=10)

        for idx in range(len(d1_vals), rows * cols):
            axes_arr[idx // cols][idx % cols].set_visible(False)

        dims_str = f"by {dim2}"
        if dim3:
            dims_str += f" × {dim3}"
        if dim4:
            dims_str += f" × {dim4} (hatch)"
        if dim1:
            dims_str += f"  (split: {dim1})"
        title_metric = f"Coefficient of Variation of {value_metric}" if is_cv_chart else f"{agg_name} {metric}"
        fig.suptitle(f"{title_metric}  ·  {dims_str}",
                     color=FG, fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._show_chart(fig)
        cv_note = "  ·  lower CV = more consistent" if is_cv_chart else ""
        self._status(f"Bar graph generated  ·  {len(df)} total rows{cv_note}")


# ──────────────────────────────────────────────────────────────────────────────
#  Pie Chart Tab
# ──────────────────────────────────────────────────────────────────────────────

class PieChartTab(BaseTab):

    def _build_chart_controls(self):
        self._sec("Chart Type")
        self.chart_type_var = tk.StringVar(value="allocation")
        for text, val in [("Sector Allocation (Pie)", "allocation"),
                          ("Profit Contribution Analysis", "profit")]:
            tk.Radiobutton(
                self.ctrl_frame, text=text, variable=self.chart_type_var, value=val,
                bg=PANEL_BG, fg=FG, selectcolor=ENTRY_BG,
                activebackground=PANEL_BG, activeforeground=ACCENT,
                font=("Segoe UI", 9), cursor="hand2"
            ).pack(anchor="w", padx=14, pady=1)

        self._sec("Dimension 1 — Separate Charts")
        self._lbl("One chart per value of:")
        _, self.dim1_var = self._combo(["(none)"] + DIMENSIONS, "Model")
        self._lbl("(none) = single chart for all filtered data")

        self._sec("Allocation Weighting")
        self._lbl("Applies to Pie mode only:")
        _, self.weight_var = self._combo(
            ["Sum of allocations", "Average per row"], "Sum of allocations")
        self._lbl(
            "Sum: total % weight across all rows\n"
            "Average: normalised per row first"
        )

        self._sec("Profit Analysis Note")
        self._lbl(
            "Profit Contribution requires per-stock\n"
            "return data from the JSON files in\n"
            "results/. First load may take a moment."
        )

        self._sector_export_data: list[dict] | None = None
        self._sep()
        tk.Button(self.ctrl_frame, text="Export Sector Data to Excel",
                  command=self._export_sector_excel,
                  bg=BTN_BG, fg=FG, activebackground=BTN_ACT,
                  relief="flat", padx=10, pady=5,
                  font=("Segoe UI", 9, "bold"),
                  cursor="hand2").pack(fill="x", padx=10, pady=(0, 4))

    def _generate(self):
        chart_type  = self.chart_type_var.get()
        dim1_raw    = self.dim1_var.get()
        dim1        = None if dim1_raw == "(none)" else dim1_raw
        weight_mode = self.weight_var.get()

        df = self.get_df()
        if df.empty:
            messagebox.showwarning("No Data", "No rows match the current filters.")
            return

        if chart_type == "profit":
            self._generate_profit(df, dim1)
        else:
            self._generate_pie(df, dim1, weight_mode)

    # ── Allocation pie charts (original behaviour) ─────────────────────────────

    def _generate_pie(self, df: pd.DataFrame, dim1, weight_mode: str):
        d1_vals = dim_values(df, dim1) if dim1 else [None]
        n_pies  = len(d1_vals)
        cols    = min(n_pies, 4)
        rows    = (n_pies + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(max(5 * cols, 7), max(5 * rows, 5)),
                                 squeeze=False)
        fig.patch.set_facecolor(CBKG)

        for idx, d1_val in enumerate(d1_vals):
            ax  = axes[idx // cols][idx % cols]
            ax.set_facecolor(CBKG)
            sub = mask(df, dim1, d1_val) if dim1 else df

            sector_totals: dict[str, float] = {}
            for _, row in sub.iterrows():
                companies = parse_companies(row.get("Companies", ""))
                if not companies:
                    continue
                row_total = sum(companies.values())
                if row_total == 0:
                    continue
                for ticker, pct in companies.items():
                    sector = TICKER_INFO.get(ticker, ("Unknown", ticker))[0]
                    if weight_mode == "Average per row":
                        sector_totals[sector] = (
                            sector_totals.get(sector, 0) + pct / row_total * 100)
                    else:
                        sector_totals[sector] = sector_totals.get(sector, 0) + pct

            title = dim_label(dim1, d1_val) if dim1 else "All Data"

            if not sector_totals:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        color=CTX, fontsize=12)
                ax.set_facecolor(CAXES)
                ax.set_title(title, color=FG, fontsize=12, fontweight="bold")
                ax.axis("off")
                continue

            sectors = sorted(sector_totals, key=lambda s: -sector_totals[s])
            values  = [sector_totals[s] for s in sectors]
            colors  = [SECTOR_COLORS.get(s, "#888888") for s in sectors]

            def _autopct(pct):
                return f"{pct:.1f}%" if pct >= 3 else ""

            wedges, _, autotexts = ax.pie(
                values, labels=None, colors=colors, autopct=_autopct,
                wedgeprops={"linewidth": 1.8, "edgecolor": CBKG},
                startangle=90, pctdistance=0.75)
            for at in autotexts:
                at.set_color("white")
                at.set_fontsize(10)
                at.set_fontweight("bold")

            ax.set_title(f"{title}\n({len(sub)} runs)",
                         color=FG, fontsize=12, fontweight="bold", pad=10)

        for idx in range(n_pies, rows * cols):
            a = axes[idx // cols][idx % cols]
            a.set_visible(False)

        all_sectors_used: set[str] = set()
        for d1_val in d1_vals:
            sub = mask(df, dim1, d1_val) if dim1 else df
            for _, row in sub.iterrows():
                for ticker in parse_companies(row.get("Companies", "")):
                    all_sectors_used.add(TICKER_INFO.get(ticker, ("Unknown",))[0])

        legend_handles = [
            mpatches.Patch(color=SECTOR_COLORS.get(s, "#888"), label=s)
            for s in SECTOR_COLORS if s in all_sectors_used
        ]
        if legend_handles:
            fig.legend(handles=legend_handles, loc="lower center",
                       ncol=min(len(legend_handles), 5),
                       facecolor=PANEL_BG, labelcolor=FG,
                       fontsize=11, framealpha=0.9, edgecolor=SEP,
                       bbox_to_anchor=(0.5, 0))

        mode_label = "normalised" if weight_mode == "Average per row" else "summed"
        suptitle = f"Sector Allocation ({mode_label})"
        if dim1:
            suptitle += f"  ·  split by {dim1}"
        fig.suptitle(suptitle, color=FG, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        self._show_chart(fig)
        # Store for Excel export
        rows_out = []
        for d1_val in d1_vals:
            sub = mask(df, dim1, d1_val) if dim1 else df
            group_lbl = dim_label(dim1, d1_val) if dim1 else "All Data"
            sector_totals_local: dict[str, float] = {}
            for _, row in sub.iterrows():
                companies = parse_companies(row.get("Companies", ""))
                if not companies:
                    continue
                row_total = sum(companies.values())
                if row_total == 0:
                    continue
                for ticker, pct in companies.items():
                    sector = TICKER_INFO.get(ticker, ("Unknown", ticker))[0]
                    if weight_mode == "Average per row":
                        sector_totals_local[sector] = sector_totals_local.get(sector, 0) + pct / row_total * 100
                    else:
                        sector_totals_local[sector] = sector_totals_local.get(sector, 0) + pct
            total = sum(sector_totals_local.values()) or 1
            for sector, val in sorted(sector_totals_local.items(), key=lambda x: -x[1]):
                rows_out.append({"Group": group_lbl, "Sector": sector,
                                 "Allocation %": round(val / total * 100, 4)})
        self._sector_export_data = rows_out
        self._status(f"Pie charts generated  ·  {len(df)} total rows")

    # ── Profit contribution analysis ───────────────────────────────────────────

    def _generate_profit(self, df: pd.DataFrame, dim1):
        if _perm_detail_cache is not None:
            self._render_profit(df, dim1)
            return
        self._status("Loading JSON files in background — chart will appear when ready…")
        def _load():
            load_perm_details()
            def _render():
                try:
                    self._render_profit(df, dim1)
                except Exception as exc:
                    messagebox.showerror("Chart Error", str(exc))
                    self._status(f"Error: {exc}")
            self.after(0, _render)
        threading.Thread(target=_load, daemon=True).start()

    def _render_profit(self, df: pd.DataFrame, dim1):
        self._status("Rendering profit contribution chart…")
        d1_vals  = dim_values(df, dim1) if dim1 else [None]
        n_charts = len(d1_vals)
        cols     = min(n_charts, 3)
        rows     = (n_charts + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(max(6 * cols, 9), max(5.5 * rows, 5)),
                                 squeeze=False)
        fig.patch.set_facecolor(CBKG)

        # Collect all sectors that appear across all groups so axes share the same x-ticks
        all_sectors_seen: list[str] = []
        group_data: list[dict] = []
        for d1_val in d1_vals:
            sub  = mask(df, dim1, d1_val) if dim1 else df
            contrib = compute_sector_contributions(sub)
            group_data.append(contrib)
            for s in contrib:
                if s not in all_sectors_seen:
                    all_sectors_seen.append(s)

        # Sort sectors by mean allocation descending
        all_sectors_seen.sort(
            key=lambda s: -np.mean([gd[s]["allocation"] for gd in group_data if s in gd]))

        x_pos    = np.arange(len(all_sectors_seen))
        bw       = 0.35          # bar width
        alloc_c  = "#4E79A7"     # blue  – allocation
        profit_c = "#59A14F"     # green – profit  (red for negative handled per bar)

        for idx, (d1_val, contrib) in enumerate(zip(d1_vals, group_data)):
            ax    = axes[idx // cols][idx % cols]
            sub   = mask(df, dim1, d1_val) if dim1 else df
            self._style_ax(ax)

            alloc_vals  = [contrib.get(s, {}).get("allocation", 0) for s in all_sectors_seen]
            profit_vals = [contrib.get(s, {}).get("profit",     0) for s in all_sectors_seen]
            ratios      = [contrib.get(s, {}).get("ratio",      0) for s in all_sectors_seen]
            sec_colors  = [SECTOR_COLORS.get(s, "#888888") for s in all_sectors_seen]

            # Allocation bars (always positive, use sector colour)
            ax.bar(x_pos - bw / 2, alloc_vals, width=bw,
                   color=sec_colors, alpha=0.75, edgecolor="#ffffff30",
                   linewidth=0.5, label="Allocation %", zorder=3)

            # Profit bars (green if positive, red if negative)
            p_colors = ["#59A14F" if v >= 0 else "#E15759" for v in profit_vals]
            ax.bar(x_pos + bw / 2, profit_vals, width=bw,
                   color=p_colors, alpha=0.85, edgecolor="#ffffff30",
                   linewidth=0.5, label="Profit Contribution %", zorder=3)

            # Ratio annotations above each sector pair
            for xi, (av, pv, ratio) in enumerate(zip(alloc_vals, profit_vals, ratios)):
                if av == 0:
                    continue
                top   = max(av, pv, 0) + 0.5
                color = "#FFD700" if ratio >= 1 else "#FF9DA7"
                ax.text(xi, top, f"×{ratio:.2f}",
                        ha="center", va="bottom",
                        fontsize=9, color=color, fontweight="bold")

            ax.axhline(0, color="#666688", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(all_sectors_seen, rotation=22, ha="right",
                               color=CTX, fontsize=10)
            ax.yaxis.set_tick_params(labelcolor=CTX)
            ax.set_ylabel("Normalised share  (%)", color=CTX, fontsize=11)

            title = dim_label(dim1, d1_val) if dim1 else "All Data"
            n_matched = sum(1 for _, row in sub.iterrows()
                            if int(row.get("Perm ID", -1)) in load_perm_details())
            ax.set_title(f"{title}\n({n_matched}/{len(sub)} runs matched)",
                         color=FG, fontsize=12, fontweight="bold", pad=8)

        for idx in range(n_charts, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        # Legend
        leg_handles = [
            mpatches.Patch(color=alloc_c,  label="Allocation %"),
            mpatches.Patch(color=profit_c, label="Profit Contribution %  (red = loss)"),
            mpatches.Patch(color="#FFD700", label="Ratio ≥ 1  (over-performs alloc)"),
            mpatches.Patch(color="#FF9DA7", label="Ratio < 1  (under-performs alloc)"),
        ]
        fig.legend(handles=leg_handles, loc="lower center", ncol=2,
                   facecolor=PANEL_BG, labelcolor=FG,
                   fontsize=10, framealpha=0.9, edgecolor=SEP,
                   bbox_to_anchor=(0.5, 0))

        suptitle = "Sector Profit Contribution vs Allocation"
        if dim1:
            suptitle += f"  ·  split by {dim1}"
        suptitle += "\n(×ratio = profit share ÷ allocation share)"
        fig.suptitle(suptitle, color=FG, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0.10, 1, 0.96])
        self._show_chart(fig)
        # Store for Excel export
        rows_out = []
        for d1_val, contrib in zip(d1_vals, group_data):
            group_lbl = dim_label(dim1, d1_val) if dim1 else "All Data"
            for sector, vals in sorted(contrib.items(), key=lambda x: -x[1]["allocation"]):
                rows_out.append({
                    "Group":             group_lbl,
                    "Sector":            sector,
                    "Allocation %":      round(vals["allocation"], 4),
                    "Profit Contrib %":  round(vals["profit"],     4),
                    "Efficiency Ratio":  round(vals["ratio"],      4),
                })
        self._sector_export_data = rows_out
        self._status(f"Profit contribution chart generated  ·  {len(df)} total rows")

    def _export_sector_excel(self):
        if not self._sector_export_data:
            messagebox.showinfo("No Data", "Generate a chart first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "*.xlsx")])
        if not path:
            return
        df_out = pd.DataFrame(self._sector_export_data)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for group, grp_df in df_out.groupby("Group", sort=False):
                sheet = str(group)[:31]   # Excel sheet names ≤ 31 chars
                grp_df.drop(columns="Group").to_excel(writer, sheet_name=sheet, index=False)
        messagebox.showinfo("Saved", f"Sector data saved to:\n{path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Statistics Tab  (95 % CI  ·  Sharpe Ratio)
# ──────────────────────────────────────────────────────────────────────────────

class StatisticsTab(BaseTab):

    def _build_chart_controls(self):
        self._sec("Metric")
        self._lbl("Quantitative value to analyse:")
        _, self.metric_var = self._combo(METRICS, "ROI %")

        self._sec("Group By (X-Axis)")
        self._lbl("One bar cluster per value of:")
        _, self.group_var = self._combo(DIMENSIONS, "Model")

        self._sec("Split By (Separate Charts)")
        self._lbl("(none) = single chart:")
        _, self.split_var = self._combo(["(none)"] + DIMENSIONS, "(none)")

        self._sec("CV % Note")
        self._lbl(
            "CV % = Std Dev / |Mean| × 100\n"
            "Lower = more consistent returns.\n"
            "Green = low CV (tight distribution).\n\n"
            "95 % CI uses the t-distribution\n"
            "(falls back to z=1.96 if scipy absent)."
        )

    def _generate(self):
        metric    = self.metric_var.get()
        group_dim = self.group_var.get()
        split_raw = self.split_var.get()
        split_dim = None if split_raw == "(none)" else split_raw

        df = self.get_df()
        if df.empty:
            messagebox.showwarning("No Data", "No rows match the current filters.")
            return

        split_vals = dim_values(df, split_dim) if split_dim else [None]
        group_vals = dim_values(df, group_dim)

        # ── Compute all statistics up-front ───────────────────────────────────
        all_blocks: list[tuple] = []   # (split_val, stat_rows)
        for split_val in split_vals:
            sub       = mask(df, split_dim, split_val) if split_dim else df
            stat_rows = []
            for gv in group_vals:
                grp  = mask(sub, group_dim, gv)
                vals = grp[metric].dropna().tolist()
                n    = len(vals)
                if n < 2:
                    continue
                arr  = np.array(vals, dtype=float)
                mean = float(arr.mean())
                std  = float(arr.std(ddof=1))
                se   = std / np.sqrt(n)
                try:
                    from scipy import stats as _sp
                    ci_lo, ci_hi = _sp.t.interval(
                        0.95, df=n - 1, loc=mean, scale=se)
                except Exception:
                    ci_lo = mean - 1.96 * se
                    ci_hi = mean + 1.96 * se
                cv = (std / abs(mean) * 100) if mean != 0 else float("nan")
                stat_rows.append({
                    "label": lbl(gv),
                    "n":     n,
                    "mean":  mean,
                    "std":   std,
                    "se":    se,
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                    "cv":    cv,
                })
            all_blocks.append((split_val, stat_rows))

        n_blocks   = len(all_blocks)
        row_height = 0.55          # inches per data row
        hdr_height = 1.8           # inches for title + column headers
        max_rows   = max((len(s) for _, s in all_blocks), default=1)
        fig_h      = max(4.0, hdr_height * n_blocks + row_height * max_rows * n_blocks)

        fig, axes_arr = plt.subplots(
            n_blocks, 1,
            figsize=(13, fig_h),
            squeeze=False)
        fig.patch.set_facecolor(CBKG)

        COL_HEADERS = [
            group_dim, "n", "Mean", "Std Dev",
            "Std Error", "95 % CI  Low", "95 % CI  High", "CV %",
        ]
        # Column alignments (index → 'left' or 'center')
        COL_ALIGN = ["left"] + ["center"] * (len(COL_HEADERS) - 1)

        for bi, (split_val, stat_rows) in enumerate(all_blocks):
            ax = axes_arr[bi][0]
            ax.set_facecolor(CBKG)
            ax.axis("off")

            title = dim_label(split_dim, split_val) if split_dim else "All Data"
            ax.set_title(
                f"{title}  ·  {metric}  grouped by {group_dim}",
                color=FG, fontsize=12, fontweight="bold", pad=10,
                loc="left")

            if not stat_rows:
                ax.text(0.5, 0.5,
                        "Insufficient data  (need n ≥ 2 per group)",
                        ha="center", va="center", color=CTX,
                        fontsize=13, transform=ax.transAxes)
                continue

            valid_cvs = [r["cv"] for r in stat_rows if not np.isnan(r["cv"])]
            max_cv = max(valid_cvs) if valid_cvs else 1.0

            cell_text:   list[list[str]] = []
            cell_colors: list[list]      = []

            for r in stat_rows:
                # CV cell: lower = more consistent = greener background
                cv_val = r["cv"]
                if np.isnan(cv_val) or max_cv == 0:
                    cv_bg = CAXES
                else:
                    ratio = 1.0 - min(cv_val, max_cv) / max_cv  # 1=low CV (good)
                    cv_bg = (0.10, 0.30 + 0.40 * ratio, 0.15)

                # Mean cell: green if positive, red if negative
                mean_bg = (0.10, 0.28, 0.15) if r["mean"] >= 0 else (0.28, 0.10, 0.10)

                cell_text.append([
                    r["label"],
                    str(r["n"]),
                    f"{r['mean']:+.2f}%",
                    f"{r['std']:.2f}%",
                    f"{r['se']:.2f}%",
                    f"{r['ci_lo']:+.2f}%",
                    f"{r['ci_hi']:+.2f}%",
                    f"{cv_val:.1f}%" if not np.isnan(cv_val) else "—",
                ])
                cell_colors.append([
                    CAXES, CAXES, mean_bg, CAXES,
                    CAXES, CAXES, CAXES, cv_bg,
                ])

            tbl = ax.table(
                cellText=cell_text,
                colLabels=COL_HEADERS,
                cellColours=cell_colors,
                cellLoc="center",
                loc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(11)
            tbl.auto_set_column_width(list(range(len(COL_HEADERS))))
            tbl.scale(1, 1.6)   # taller rows

            # Style header row
            for j in range(len(COL_HEADERS)):
                cell = tbl[0, j]
                cell.set_facecolor(ACCENT)
                cell.set_text_props(color="white", fontweight="bold",
                                    fontsize=11)
                cell.set_edgecolor(SEP)

            # Style data rows
            for i in range(len(stat_rows)):
                for j in range(len(COL_HEADERS)):
                    cell = tbl[i + 1, j]
                    # Columns 2 (Mean) and 7 (CV %) have coloured backgrounds —
                    # always use white text there regardless of theme.
                    txt_color = "white" if j in (2, 7) else FG
                    cell.set_text_props(color=txt_color, fontsize=11,
                                        ha=COL_ALIGN[j])
                    cell.set_edgecolor(SEP)

        suptitle = f"Statistical Summary  ·  {metric}  ·  grouped by {group_dim}"
        if split_dim:
            suptitle += f"  (split by {split_dim})"
        suptitle += (
            "\nCV % = Std Dev ÷ |Mean| × 100  ·  "
            "95 % CI via t-distribution  ·  green mean = positive return  ·  green CV = consistent (low variation)"
        )
        fig.suptitle(suptitle, color=FG, fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        self._show_chart(fig)
        total_groups = sum(len(s) for _, s in all_blocks)
        self._status(
            f"Statistics generated  ·  {len(df)} total rows  ·  "
            f"{total_groups} groups with n ≥ 2"
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Bubble Chart Tab
# ──────────────────────────────────────────────────────────────────────────────

class BubbleChartTab(BaseTab):
    """Bubble chart — defaults to the two-stage filter grid
    (x = Stage-Two size, y = Stage-One size, bubble area = mean ROI)."""

    _AXIS_CHOICES = ["final", "initial"] + METRICS
    _SIZE_CHOICES = ["ROI %", "Count"] + [m for m in METRICS if m != "ROI %"]

    def _build_chart_controls(self):
        self._sec("X-Axis")
        self._lbl("'final' = Stage-Two size, or a metric:")
        _, self.x_var = self._combo(self._AXIS_CHOICES, "final")

        self._sec("Y-Axis")
        self._lbl("'initial' = Stage-One size, or a metric:")
        _, self.y_var = self._combo(self._AXIS_CHOICES, "initial")

        self._sec("Bubble Size")
        self._lbl("Area encodes this value:")
        _, self.size_var = self._combo(self._SIZE_CHOICES, "ROI %")
        self._lbl("Aggregation (ignored for Count):")
        _, self.size_agg_var = self._combo(["Mean", "Median", "Max", "Min"], "Mean")

        self._sec("One Bubble Per")
        _, self.group_var = self._combo(DIMENSIONS, "Filter")

        self._sec("Colour Series")
        self._lbl("(none) = single colour by group:")
        _, self.color_var = self._combo(["(none)"] + DIMENSIONS, "(none)")

        self._sec("Stack Series")
        self.stack_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self.ctrl_frame, text="Stack colour series concentrically",
            variable=self.stack_var,
            bg=PANEL_BG, fg=FG, selectcolor=ENTRY_BG,
            activebackground=PANEL_BG, activeforeground=ACCENT,
            font=("Segoe UI", 9, "bold"), cursor="hand2", anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 4))
        self._lbl("Nests each cell's bubbles (largest behind)\n"
                  "instead of spreading them apart.")

    def _generate(self):
        color = None if self.color_var.get() == "(none)" else self.color_var.get()
        df = self.get_df()
        if df.empty:
            messagebox.showwarning("No Data", "No rows match the current filters.")
            return
        fig = render_bubble(
            df, x=self.x_var.get(), y=self.y_var.get(),
            size=self.size_var.get(), size_agg=self.size_agg_var.get(),
            group_dim=self.group_var.get(), color_dim=color,
            stack=self.stack_var.get())
        self._show_chart(fig)
        self._status(f"Bubble chart generated  ·  {len(df)} total rows")


# ──────────────────────────────────────────────────────────────────────────────
#  Main Application
# ──────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Investment Strategy Chart Visualizer")
        self.geometry("1440x860")
        self.minsize(1100, 700)
        self.configure(bg=BG)

        # ── ttk styles ────────────────────────────────────────────────────────
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",     background=BG,       borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL_BG, foreground=FG,
                        padding=[18, 7], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#ffffff")])
        style.configure("TFrame", background=BG)
        style.configure("TScrollbar", background=PANEL_BG, troughcolor=BG,
                        arrowcolor=FG, bordercolor=SEP)
        style.configure("Vertical.TScrollbar", background=PANEL_BG,
                        troughcolor=BG, arrowcolor=FG)
        style.configure("TCombobox", fieldbackground=ENTRY_BG,
                        background=ENTRY_BG, foreground=FG,
                        arrowcolor=FG, bordercolor=SEP,
                        selectbackground=ACCENT, selectforeground="white")
        style.map("TCombobox",
                  fieldbackground=[("readonly", ENTRY_BG)],
                  foreground=[("readonly", FG)])

        # ── Header ────────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=ACCENT, height=50)
        header.pack(fill="x")
        tk.Label(header,
                 text="  ⬡  Investment Strategy Chart Visualizer",
                 bg=ACCENT, fg="white",
                 font=("Segoe UI", 14, "bold")).pack(side="left", pady=10)
        tk.Label(header,
                 text="T0.3 · T0.4 · T0.5  |  gpt-4o · gpt-4o-mini · gpt-5 · gpt-5.1  ",
                 bg=ACCENT, fg="#ddd8ff",
                 font=("Segoe UI", 9)).pack(side="right", pady=14)
        self._theme_btn = tk.Button(
            header, text="☀ Light Charts",
            command=lambda: _toggle_theme(self._theme_btn),
            bg=BTN_BG, fg="white", activebackground=BTN_ACT,
            relief="flat", padx=12, pady=4,
            font=("Segoe UI", 9, "bold"), cursor="hand2")
        self._theme_btn.pack(side="right", padx=(0, 14), pady=8)

        # ── Load data ─────────────────────────────────────────────────────────
        try:
            df = load_data()
        except Exception as exc:
            messagebox.showerror("Data Load Error", str(exc))
            self.destroy()
            return

        row_counts = (
            f"Loaded {len(df)} rows with ROI data  ·  "
            f"Models: {', '.join(sorted(df['Model'].unique()))}  ·  "
            f"Temps: {', '.join(str(t) for t in sorted(df['Temperature'].unique()))}"
        )
        tk.Label(self, text=row_counts, bg=BG, fg="#777799",
                 font=("Segoe UI", 8)).pack(anchor="w", padx=12, pady=(4, 0))

        # ── Notebook ──────────────────────────────────────────────────────────
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=(4, 6))

        nb.add(BoxPlotTab(nb, df),    text="  Box Plots  ")
        nb.add(BarGraphTab(nb, df),   text="  Bar Graphs  ")
        nb.add(PieChartTab(nb, df),   text="  Pie Charts  ")
        nb.add(BubbleChartTab(nb, df), text="  Bubble  ")
        nb.add(StatisticsTab(nb, df), text="  Statistics  ")

        # Pre-load JSON detail files in background so Profit Contribution
        # chart is ready without lag on first use.
        threading.Thread(target=load_perm_details, daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────────




# -----------------------------------------------------------------------------
#  Paper Figure Export
# -----------------------------------------------------------------------------

PAPER_FIG_DIR = os.path.join(DATA_DIR, "figures")

PAPER_DISPLAY = {
    "10->5": "10->5",
    "20->5": "20->5",
    "unfiltered": "Unfiltered",
    "10->3": "10->3",
    "20->10": "20->10",
    "ranked_final": "Ranked Final",
    "30->15": "30->15",
    "5->3": "5->3",
    "30->10": "30->10",
    "30->5": "30->5",
    "gpt-5.1": "GPT-5.1",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
}


def _paper_label(value) -> str:
    return PAPER_DISPLAY.get(str(value), str(value))


def _paper_setup_ax(ax, title: str, xlabel: str | None = None, ylabel: str | None = None):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#94A3B8")
    ax.spines["left"].set_color("#94A3B8")
    ax.tick_params(axis="both", labelsize=9)


def _paper_annotate_bars(ax, values, fmt="{:.1f}%"):
    ymax = max(values) if len(values) else 0
    pad = ymax * 0.025 if ymax else 0.5
    for idx, value in enumerate(values):
        ax.text(idx, value + pad, fmt.format(value), ha="center", va="bottom",
                fontsize=9, color="#334155")


def _paper_format_xlabels(ax, rotation=35):
    ax.tick_params(axis="x", rotation=rotation)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def _paper_save(fig, filename: str):
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    out_path = os.path.join(PAPER_FIG_DIR, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


def _paper_config_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Filter", dropna=False)["ROI %"]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": "Mean ROI",
            "median": "Median ROI",
            "std": "SD",
            "min": "Min ROI",
            "max": "Max ROI",
            "count": "Sample Size",
        })
    )


def _paper_win_rate_vs_unfiltered(df: pd.DataFrame) -> pd.DataFrame:
    context_cols = ["Model", "Temperature", "Period (mo)", "Start", "End"]
    means = df.groupby(context_cols + ["Filter"], dropna=False)["ROI %"].mean().reset_index()
    baseline = means[means["Filter"] == "unfiltered"][context_cols + ["ROI %"]].rename(
        columns={"ROI %": "Unfiltered ROI"}
    )
    comp = means[means["Filter"] != "unfiltered"].merge(baseline, on=context_cols, how="inner")
    comp["Beat Unfiltered"] = comp["ROI %"] > comp["Unfiltered ROI"]
    return comp.groupby("Filter", dropna=False).agg(
        **{"Win Rate": ("Beat Unfiltered", "mean"), "Comparisons": ("Beat Unfiltered", "count")}
    ).reset_index()


def _paper_vertical_bar(labels, values, title, xlabel, ylabel, filename, color="#2563EB", threshold=None):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if isinstance(color, list):
        colors = color
    else:
        colors = [color] * len(values)
    ax.bar(labels, values, color=colors)
    if threshold is not None:
        ax.axhline(threshold, color="#0F172A", linewidth=1, linestyle="--", alpha=0.75)
    _paper_setup_ax(ax, title, xlabel, ylabel)
    _paper_annotate_bars(ax, values)
    _paper_format_xlabels(ax)
    ax.set_ylim(0, max(values) * 1.18 if values else 1)
    _paper_save(fig, filename)


def _paper_export_mean_roi_by_filter(df: pd.DataFrame):
    order = (
        df.groupby("Filter", dropna=False)["ROI %"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    model_order = ["gpt-4o", "gpt-4o-mini", "gpt-5.1"]
    pivot = (
        df.groupby(["Filter", "Model"], dropna=False)["ROI %"]
        .mean()
        .unstack("Model")
        .reindex(order)
    )

    labels = [_paper_label(v) for v in pivot.index]
    x = np.arange(len(labels))
    width = 0.24
    colors = {
        "gpt-4o": "#2563EB",
        "gpt-4o-mini": "#059669",
        "gpt-5.1": "#DC2626",
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for idx, model in enumerate(model_order):
        if model not in pivot.columns:
            continue
        vals = pivot[model].astype(float).tolist()
        offset = (idx - (len(model_order) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width, color=colors.get(model), label=_paper_label(model))
        for bar, value in zip(bars, vals):
            if pd.isna(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(pivot.max(numeric_only=True)) * 0.018,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#334155",
                rotation=90,
            )

    _paper_setup_ax(
        ax,
        "Figure 4.2.1 - Mean ROI by Filtering Configuration and Model",
        "Filtering configuration",
        "Mean return on investment (%)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _paper_format_xlabels(ax)
    ax.set_ylim(0, float(pivot.max(numeric_only=True).max()) * 1.24)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _paper_save(fig, "figure_4_2_1_mean_roi_by_filter.png")


def _paper_export_win_rate_vs_unfiltered(df: pd.DataFrame):
    context_cols = ["Model", "Temperature", "Period (mo)", "Start", "End"]
    means = df.groupby(context_cols + ["Filter"], dropna=False)["ROI %"].mean().reset_index()
    baseline = means[means["Filter"] == "unfiltered"][context_cols + ["ROI %"]].rename(
        columns={"ROI %": "Unfiltered ROI"}
    )
    comp = means[means["Filter"] != "unfiltered"].merge(baseline, on=context_cols, how="inner")
    comp["Beat Unfiltered"] = comp["ROI %"] > comp["Unfiltered ROI"]
    win = (
        comp.groupby(["Filter", "Model"], dropna=False)["Beat Unfiltered"]
        .mean()
        .mul(100)
        .reset_index(name="Win Rate %")
    )
    model_order = ["gpt-4o", "gpt-4o-mini", "gpt-5.1"]
    filter_order = (
        comp.groupby("Filter", dropna=False)["Beat Unfiltered"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    pivot = win.pivot(index="Model", columns="Filter", values="Win Rate %").reindex(model_order)

    palette = {
        "10->5": "#2563EB",
        "20->5": "#059669",
        "10->3": "#DC2626",
        "5->3": "#7C3AED",
        "20->10": "#F59E0B",
        "30->15": "#0F766E",
        "ranked_final": "#64748B",
        "30->10": "#DB2777",
        "30->5": "#9333EA",
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.9))
    max_val = float(pivot.max(numeric_only=True).max())
    group_gap = 0.35
    bar_width = 0.12
    n_filters = len(filter_order)
    group_width = n_filters * bar_width
    model_centers = []

    for model_idx, model in enumerate(model_order):
        if model not in pivot.index:
            continue
        base = model_idx * (group_width + group_gap)
        model_centers.append(base + group_width / 2 - bar_width / 2)
        for filter_idx, filt in enumerate(filter_order):
            if filt not in pivot.columns:
                continue
            value = pivot.loc[model, filt]
            if pd.isna(value):
                continue
            x_pos = base + filter_idx * bar_width
            bar = ax.bar(
                x_pos,
                value,
                width=bar_width * 0.92,
                color=palette.get(filt, "#475569"),
                label=_paper_label(filt) if model_idx == 0 else None,
            )[0]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max_val * 0.018,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#334155",
                rotation=90,
            )

    ax.axhline(50, color="#0F172A", linewidth=1, linestyle="--", alpha=0.75)
    _paper_setup_ax(
        ax,
        "Figure 4.2.2 - Filtering Win Rate vs Unfiltered Baseline by Model",
        "Language model",
        "Matched evaluation windows beating unfiltered (%)",
    )
    ax.set_xticks(model_centers)
    ax.set_xticklabels([_paper_label(m) for m in model_order])
    ax.tick_params(axis="x", rotation=0)
    ax.set_ylim(0, max_val * 1.24)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _paper_save(fig, "figure_4_2_2_win_rate_vs_unfiltered.png")


def _paper_export_mean_roi_by_model(df: pd.DataFrame):
    model_order = ["gpt-5.1", "gpt-4o", "gpt-4o-mini"]
    stats_df = df.groupby("Model", dropna=False)["ROI %"].mean().reindex(model_order).dropna().reset_index(name="Mean ROI")
    _paper_vertical_bar(
        [_paper_label(v) for v in stats_df["Model"]],
        stats_df["Mean ROI"].astype(float).tolist(),
        "Figure 4.3.1 - Mean ROI by Language Model",
        "Language model",
        "Mean ROI (%)",
        "figure_4_3_1_mean_roi_by_model.png",
        color=["#2563EB", "#475569", "#059669"][:len(stats_df)],
    )


def _paper_export_roi_boxplot_by_model(df: pd.DataFrame):
    model_order = ["gpt-5.1", "gpt-4o", "gpt-4o-mini"]
    labels = []
    series = []
    for model in model_order:
        values = df.loc[df["Model"] == model, "ROI %"].dropna().astype(float)
        if values.empty:
            continue
        labels.append(_paper_label(model))
        series.append(values)

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    colors = ["#2563EB", "#475569", "#059669"][: len(series)]
    box = ax.boxplot(
        series,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#F59E0B",
            "markeredgecolor": "#92400E",
            "markersize": 5,
        },
        medianprops={"color": "#0F172A", "linewidth": 1.5},
        boxprops={"linewidth": 1.2, "color": "#334155"},
        whiskerprops={"linewidth": 1.1, "color": "#334155"},
        capprops={"linewidth": 1.1, "color": "#334155"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#CBD5E1",
            "markeredgecolor": "#64748B",
            "markersize": 3.5,
            "alpha": 0.75,
        },
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)

    _paper_setup_ax(
        ax,
        "Figure 4.3.3 - ROI Distribution by Language Model",
        "Language model",
        "Return on investment (%)",
    )
    ax.tick_params(axis="x", rotation=0)
    ax.axhline(0, color="#0F172A", linewidth=1, alpha=0.35)
    _paper_save(fig, "figure_4_3_3_roi_boxplot_by_model.png")


def _paper_export_gpt51_by_filter(df: pd.DataFrame):
    stats_df = (
        df[df["Model"] == "gpt-5.1"]
        .groupby("Filter", dropna=False)["ROI %"]
        .mean()
        .reset_index(name="Mean ROI")
        .sort_values("Mean ROI", ascending=False)
    )
    _paper_vertical_bar(
        [_paper_label(v) for v in stats_df["Filter"]],
        stats_df["Mean ROI"].astype(float).tolist(),
        "Figure 4.3.2 - GPT-5.1 ROI Across Filtering Configurations",
        "Filtering configuration",
        "Mean return on investment (%)",
        "figure_4_3_2_gpt51_by_filter.png",
        color="#059669",
    )


def _paper_sp_returns_by_window(periods: set[int]) -> pd.DataFrame:
    analysis_path = os.path.join(DATA_DIR, "paper_analysis_tables.xlsx")
    if not os.path.exists(analysis_path):
        return pd.DataFrame(columns=["Period (mo)", "Start", "End", "S&P Return"])
    try:
        sp_df = pd.read_excel(analysis_path, sheet_name="Evaluation Window Analysis")
    except Exception:
        return pd.DataFrame(columns=["Period (mo)", "Start", "End", "S&P Return"])

    needed = ["Period (mo)", "Start", "End", "S&P Return"]
    if any(col not in sp_df.columns for col in needed):
        return pd.DataFrame(columns=needed)
    sp_df = sp_df[needed].dropna(subset=["S&P Return"]).copy()
    sp_df["Period (mo)"] = sp_df["Period (mo)"].astype(int)
    sp_df["Start"] = sp_df["Start"].astype(str)
    sp_df["End"] = sp_df["End"].astype(str)
    return sp_df[sp_df["Period (mo)"].isin(periods)]


def _paper_export_roi_by_window(df: pd.DataFrame):
    grouped = (
        df.groupby(["Period (mo)", "Start", "End"], dropna=False)["ROI %"]
        .mean()
        .reset_index(name="Mean ROI")
        .sort_values(["Period (mo)", "Start"])
    )
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for period, grp in grouped.groupby("Period (mo)"):
        labels = [f"{s}-{e}" for s, e in zip(grp["Start"], grp["End"])]
        ax.plot(labels, grp["Mean ROI"], marker="o", linewidth=2, label=f"{int(period)} mo")

    sp_returns = _paper_sp_returns_by_window(set(grouped["Period (mo)"].dropna().astype(int)))
    if not sp_returns.empty:
        for period, grp in sp_returns.sort_values(["Period (mo)", "Start"]).groupby("Period (mo)"):
            labels = [f"{s}-{e}" for s, e in zip(grp["Start"], grp["End"])]
            ax.plot(
                labels,
                grp["S&P Return"].astype(float),
                marker="D",
                markersize=4,
                linewidth=2,
                linestyle="--",
                color="#F59E0B",
                label=f"S&P 500 ({int(period)} mo)",
            )
    _paper_setup_ax(ax, "Figure 4.4.1 - ROI by Evaluation Window", "Evaluation window", "Mean ROI (%)")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=45)
    _paper_save(fig, "figure_4_4_1_roi_by_evaluation_window.png")


def _paper_export_cv_by_filter(df: pd.DataFrame):
    def cv_percent(series: pd.Series) -> float:
        values = series.dropna().astype(float)
        mean = values.mean()
        return float(values.std() / abs(mean) * 100) if len(values) > 1 and mean else float("nan")

    order = (
        df.groupby("Filter", dropna=False)["ROI %"]
        .agg(cv_percent)
        .sort_values(ascending=True)
        .index
        .tolist()
    )
    model_order = ["gpt-4o", "gpt-4o-mini", "gpt-5.1"]
    pivot = (
        df.groupby(["Filter", "Model"], dropna=False)["ROI %"]
        .agg(cv_percent)
        .unstack("Model")
        .reindex(order)
    )

    labels = [_paper_label(v) for v in pivot.index]
    x = np.arange(len(labels))
    width = 0.24
    colors = {
        "gpt-4o": "#2563EB",
        "gpt-4o-mini": "#059669",
        "gpt-5.1": "#DC2626",
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    max_val = float(pivot.max(numeric_only=True).max())
    for idx, model in enumerate(model_order):
        if model not in pivot.columns:
            continue
        vals = pivot[model].astype(float).tolist()
        offset = (idx - (len(model_order) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width, color=colors.get(model), label=_paper_label(model))
        for bar, value in zip(bars, vals):
            if pd.isna(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max_val * 0.018,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#334155",
                rotation=90,
            )

    _paper_setup_ax(
        ax,
        "Figure 4.4.1 - Coefficient of Variation by Filtering Configuration and Model",
        "Filtering configuration",
        "Coefficient of variation (%)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _paper_format_xlabels(ax)
    ax.set_ylim(0, max_val * 1.24)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _paper_save(fig, "figure_4_4_1_cv_by_filter.png")


def _paper_export_temperature_roi(df: pd.DataFrame):
    """Figure 4.7.1 — grouped bars of mean ROI by temperature for each model."""
    model_order = ["gpt-4o", "gpt-4o-mini", "gpt-5.1"]
    temps = sorted(df["Temperature"].dropna().unique())
    pivot = (
        df.groupby(["Temperature", "Model"], dropna=False)["ROI %"]
        .mean()
        .unstack("Model")
        .reindex(temps)
    )

    x = np.arange(len(temps))
    width = 0.24
    colors = {"gpt-4o": "#2563EB", "gpt-4o-mini": "#059669", "gpt-5.1": "#DC2626"}

    fig, ax = plt.subplots(figsize=(9, 5.2))
    max_val = float(pivot.max(numeric_only=True).max())
    for idx, model in enumerate(model_order):
        if model not in pivot.columns:
            continue
        vals = pivot[model].astype(float).tolist()
        offset = (idx - (len(model_order) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width,
                      color=colors.get(model), label=_paper_label(model))
        for bar, value in zip(bars, vals):
            if pd.isna(value):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, value + max_val * 0.018,
                    f"{value:.1f}%", ha="center", va="bottom",
                    fontsize=8, color="#334155")

    _paper_setup_ax(
        ax,
        "Figure 4.7.1 - Effect of Temperature on ROI by Model",
        "Generation temperature",
        "Mean return on investment (%)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"T = {t:g}" for t in temps])
    ax.tick_params(axis="x", rotation=0)
    ax.set_ylim(0, max_val * 1.22)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _paper_save(fig, "figure_4_7_1_temperature_roi.png")


def _paper_export_sector_contribution(df: pd.DataFrame):
    """Figure 5.3.1 — per-model allocation share vs realised profit share by sector."""
    model_order = ["gpt-4o", "gpt-4o-mini", "gpt-5.1"]
    models = [m for m in model_order if m in df["Model"].unique()]
    if not models:
        return

    # Gather contributions for every model first so subplots share x-ticks.
    group_data = [compute_sector_contributions(df[df["Model"] == m]) for m in models]
    sectors_seen: list[str] = []
    for gd in group_data:
        for s in gd:
            if s not in sectors_seen:
                sectors_seen.append(s)
    if not sectors_seen:
        return
    sectors_seen.sort(
        key=lambda s: -np.mean([gd[s]["allocation"] for gd in group_data if s in gd]))

    x_pos = np.arange(len(sectors_seen))
    bw = 0.38
    fig, axes = plt.subplots(1, len(models),
                             figsize=(max(5.2 * len(models), 7), 5.4),
                             squeeze=False)

    for idx, (model, contrib) in enumerate(zip(models, group_data)):
        ax = axes[0][idx]
        alloc_vals = [contrib.get(s, {}).get("allocation", 0) for s in sectors_seen]
        profit_vals = [contrib.get(s, {}).get("profit", 0) for s in sectors_seen]
        sec_colors = [SECTOR_COLORS.get(s, "#888888") for s in sectors_seen]

        ax.bar(x_pos - bw / 2, alloc_vals, width=bw, color=sec_colors, alpha=0.65,
               edgecolor="#33415530", linewidth=0.5, label="Allocation share %")
        p_colors = ["#1B7F4B" if v >= 0 else "#C0392B" for v in profit_vals]
        ax.bar(x_pos + bw / 2, profit_vals, width=bw, color=p_colors, alpha=0.95,
               edgecolor="#33415530", linewidth=0.5, label="ROI share %")

        _paper_setup_ax(ax, _paper_label(model), None, "Share of total (%)" if idx == 0 else None)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sectors_seen)
        _paper_format_xlabels(ax, rotation=30)
        ax.axhline(0, color="#94A3B8", linewidth=0.8)

    handles = [
        mpatches.Patch(color="#64748B", alpha=0.65, label="Allocation share (% of recommended weight)"),
        mpatches.Patch(color="#1B7F4B", label="ROI share (% of realised return)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Figure 5.3.1 - Sector Allocation vs Realised ROI Share by Model",
                 x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    _paper_save(fig, "figure_5_3_1_sector_contribution.png")


def export_paper_figures():
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "white"})
    df = load_data()
    df = df[df["Period (mo)"] == 12].copy()
    _paper_export_mean_roi_by_filter(df)
    _paper_export_win_rate_vs_unfiltered(df)
    _paper_export_mean_roi_by_model(df)
    _paper_export_roi_boxplot_by_model(df)
    _paper_export_gpt51_by_filter(df)
    _paper_export_roi_by_window(df)
    _paper_export_cv_by_filter(df)
    _paper_export_temperature_roi(df)
    _paper_export_sector_contribution(df)
    print(f"Exported paper figures from chart_visualizer.py using {len(df)} 12-month rows.")


# ──────────────────────────────────────────────────────────────────────────────
#  Headless rendering engine  (no tkinter required)
# ──────────────────────────────────────────────────────────────────────────────
#  Every chart type available through the GUI can also be produced here as a
#  standalone matplotlib Figure, plus a new Bubble chart.  These functions share
#  the same data helpers as the GUI and honour the module colour-theme globals,
#  so a single code path drives both interactive and headless output.

def hl_set_theme(dark: bool = False):
    """Apply the dark or light colour theme to the module globals."""
    global CBKG, CAXES, CGRID, CTX, FG, PANEL_BG, SEP
    src = _DARK_THEME if dark else _LIGHT_THEME
    CBKG, CAXES, CGRID = src["CBKG"], src["CAXES"], src["CGRID"]
    CTX, FG, PANEL_BG, SEP = src["CTX"], src["FG"], src["PANEL_BG"], src["SEP"]


def _hl_style_ax(ax):
    ax.set_facecolor(CAXES)
    ax.tick_params(colors=CTX, labelsize=10)
    ax.xaxis.label.set_color(CTX)
    ax.yaxis.label.set_color(CTX)
    ax.title.set_color(CTX)
    for sp in ax.spines.values():
        sp.set_edgecolor(SEP)
    ax.grid(True, color=CGRID, linewidth=0.5, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)


def _hl_new_fig(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(CBKG)
    return fig, ax


# Aliases that turn a Filter label such as "30->5" into its two stage sizes.
_FILTER_AXIS_ALIASES = {
    "initialsize": "initial", "initial": "initial", "stage1": "initial",
    "stageone": "initial", "in": "initial", "pool": "initial",
    "finalsize": "final", "final": "final", "stage2": "final",
    "stagetwo": "final", "out": "final", "output": "final",
}


def parse_filter_sizes(filter_value) -> tuple[float | None, float | None]:
    """'30->5' → (30, 5).  Non X→Y configs (unfiltered, ranked_final) → (None, None)."""
    m = re.match(r"\s*(\d+)\s*-+>\s*(\d+)\s*$", str(filter_value))
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


# ── Bar graph ───────────────────────────────────────────────────────────────

def render_bar(df, metric="ROI %", agg="Mean",
               dim1=None, dim2="Filter", dim3="Model", dim4=None):
    is_cv_metric = metric == CV_METRIC
    is_cv_chart = is_cv_metric or agg == "CV %"
    value_metric = "ROI %" if is_cv_metric else metric
    value_agg = "CV %" if is_cv_metric else agg

    d1_vals = dim_values(df, dim1) if dim1 else [None]
    d2_vals = dim_values(df, dim2)
    d3_vals = dim_values(df, dim3) if dim3 else [None]
    d4_vals = dim_values(df, dim4) if dim4 else [None]

    n_plots = len(d1_vals)
    cols = min(n_plots, 2)
    rows = (n_plots + cols - 1) // cols
    fig, axes_arr = _hl_new_fig(rows, cols,
                                figsize=(max(9 * cols, 11), max(5 * rows, 5)),
                                squeeze=False)

    n_series = len(d3_vals) * len(d4_vals)
    bar_width = max(0.05, min(0.7 / n_series, 0.28))
    d3_colors = {v: PLOT_COLORS[i % len(PLOT_COLORS)] for i, v in enumerate(d3_vals)}
    d4_hatches = {v: HATCHES[i % len(HATCHES)] for i, v in enumerate(d4_vals)}
    x_positions = np.arange(len(d2_vals))

    for plot_idx, d1_val in enumerate(d1_vals):
        ax = axes_arr[plot_idx // cols][plot_idx % cols]
        sub1 = mask(df, dim1, d1_val) if dim1 else df
        _hl_style_ax(ax)

        series_idx = 0
        for d3_val in d3_vals:
            for d4_val in d4_vals:
                heights = []
                for d2_val in d2_vals:
                    grp = mask(sub1, dim2, d2_val)
                    if dim3:
                        grp = mask(grp, dim3, d3_val)
                    if dim4:
                        grp = mask(grp, dim4, d4_val)
                    h = agg_apply(grp[value_metric].dropna(), value_agg)
                    heights.append(np.nan if np.isnan(h) else h)
                offset = (series_idx - n_series / 2 + 0.5) * bar_width
                label_parts = []
                if dim3:
                    label_parts.append(lbl(d3_val))
                if dim4:
                    label_parts.append(lbl(d4_val))
                ax.bar(x_positions + offset, heights, width=bar_width,
                       color=d3_colors[d3_val], hatch=d4_hatches[d4_val],
                       edgecolor="#ffffff30", linewidth=0.5,
                       label=" / ".join(label_parts), alpha=0.86, zorder=3)
                series_idx += 1

        ax.axhline(0, color="#666688", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([lbl(v) for v in d2_vals], rotation=30, ha="right",
                           color=CTX, fontsize=10)
        ylabel = (f"Coefficient of Variation of {value_metric} (%)"
                  if is_cv_chart else f"{agg}  {metric}")
        ax.set_ylabel(ylabel, color=CTX, fontsize=11)
        ax.set_title(dim_label(dim1, d1_val) if dim1 else "All Data",
                     color=FG, fontsize=12, fontweight="bold", pad=8)
        _h, _l = ax.get_legend_handles_labels()
        _h, _l = zip(*[(h, l) for h, l in zip(_h, _l) if l]) if _h else ([], [])
        if _h:
            ax.legend(list(_h), list(_l), facecolor=PANEL_BG, labelcolor=FG,
                      fontsize=9, framealpha=0.9, edgecolor=SEP,
                      title=" / ".join(d for d in [dim3, dim4] if d) or None)

    for idx in range(len(d1_vals), rows * cols):
        axes_arr[idx // cols][idx % cols].set_visible(False)

    dims_str = f"by {dim2}" + (f" × {dim3}" if dim3 else "") + (f" × {dim4} (hatch)" if dim4 else "")
    if dim1:
        dims_str += f"  (split: {dim1})"
    title_metric = (f"Coefficient of Variation of {value_metric}"
                    if is_cv_chart else f"{agg} {metric}")
    fig.suptitle(f"{title_metric}  ·  {dims_str}", color=FG, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Box / violin distribution ─────────────────────────────────────────────────

def render_box(df, metric="ROI %", plot_style="Box Plot",
               dim1="Temperature", dim2="Model"):
    d1_vals = dim_values(df, dim1) if dim1 else [None]
    d2_vals = dim_values(df, dim2)
    n_plots = len(d1_vals)
    cols = min(n_plots, 3)
    rows = (n_plots + cols - 1) // cols
    fig, axes_arr = _hl_new_fig(rows, cols,
                                figsize=(max(6 * cols, 9), max(5 * rows, 5)),
                                squeeze=False)
    colors = [PLOT_COLORS[i % len(PLOT_COLORS)] for i in range(len(d2_vals))]

    for idx, d1_val in enumerate(d1_vals):
        ax = axes_arr[idx // cols][idx % cols]
        sub = mask(df, dim1, d1_val) if dim1 else df
        _hl_style_ax(ax)
        data_groups, has_data = [], []
        for d2_val in d2_vals:
            vals = mask(sub, dim2, d2_val)[metric].dropna().tolist()
            data_groups.append(vals)
            has_data.append(bool(vals))
        positions = list(range(1, len(d2_vals) + 1))
        ne_pos = [p for p, h in zip(positions, has_data) if h]
        ne_grp = [g for g, h in zip(data_groups, has_data) if h]
        ne_col = [c for c, h in zip(colors, has_data) if h]

        if plot_style in ("Violin", "Box + Violin") and ne_grp:
            vio_pos = [p for p, g in zip(ne_pos, ne_grp) if len(set(g)) >= 2]
            vio_data = [g for g in ne_grp if len(set(g)) >= 2]
            vio_col = [c for c, g in zip(ne_col, ne_grp) if len(set(g)) >= 2]
            if vio_data:
                parts = ax.violinplot(vio_data, positions=vio_pos,
                                      showmedians=True, showextrema=True, widths=0.75)
                for pc, color in zip(parts["bodies"], vio_col):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.45)
                for pname in ("cbars", "cmaxes", "cmins", "cmedians"):
                    if pname in parts:
                        parts[pname].set_color(CTX)
        if plot_style in ("Box Plot", "Box + Violin") and ne_grp:
            bw = 0.30 if plot_style == "Box + Violin" else 0.55
            bp = ax.boxplot(ne_grp, positions=ne_pos, patch_artist=True, widths=bw,
                            showfliers=(plot_style == "Box Plot"), zorder=3)
            for patch, color in zip(bp["boxes"], ne_col):
                patch.set_facecolor(color)
                patch.set_alpha(0.85 if plot_style == "Box Plot" else 0.60)
            for med in bp["medians"]:
                med.set_color("#ffffff" if _DARK_THEME["CBKG"] == CBKG else "#0F172A")
                med.set_linewidth(2)
            for elem in ("whiskers", "caps"):
                for item in bp[elem]:
                    item.set_color(CTX)

        ax.axhline(0, color="#666688", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels([lbl(v) for v in d2_vals], rotation=20, ha="right",
                           color=CTX, fontsize=10)
        ax.set_ylabel(metric, color=CTX, fontsize=11)
        ax.set_title(dim_label(dim1, d1_val) if dim1 else "All Data",
                     color=FG, fontsize=12, fontweight="bold", pad=8)

    for idx in range(len(d1_vals), rows * cols):
        axes_arr[idx // cols][idx % cols].set_visible(False)

    leg = [mpatches.Patch(color=c) for c in colors]
    fig.legend(leg, [lbl(v) for v in d2_vals], title=dim2, loc="lower center",
               ncol=min(len(leg), 6), facecolor=PANEL_BG, labelcolor=FG,
               framealpha=0.9, edgecolor=SEP, bbox_to_anchor=(0.5, 0))
    suptitle = f"Distribution of {metric}"
    suptitle += f"  ·  split by {dim1}  &  {dim2}" if dim1 else f"  ·  grouped by {dim2}"
    fig.suptitle(suptitle, color=FG, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# ── Sector allocation pie ─────────────────────────────────────────────────────

def render_pie(df, dim1="Model", weight_mode="Sum of allocations"):
    d1_vals = dim_values(df, dim1) if dim1 else [None]
    n_pies = len(d1_vals)
    cols = min(n_pies, 4)
    rows = (n_pies + cols - 1) // cols
    fig, axes = _hl_new_fig(rows, cols,
                            figsize=(max(5 * cols, 7), max(5 * rows, 5)), squeeze=False)
    used: set[str] = set()
    for idx, d1_val in enumerate(d1_vals):
        ax = axes[idx // cols][idx % cols]
        ax.set_facecolor(CBKG)
        sub = mask(df, dim1, d1_val) if dim1 else df
        totals: dict[str, float] = {}
        for _, row in sub.iterrows():
            companies = parse_companies(row.get("Companies", ""))
            row_total = sum(companies.values())
            if row_total == 0:
                continue
            for ticker, pct in companies.items():
                sector = TICKER_INFO.get(ticker, ("Unknown", ticker))[0]
                used.add(sector)
                add = pct / row_total * 100 if weight_mode == "Average per row" else pct
                totals[sector] = totals.get(sector, 0) + add
        title = dim_label(dim1, d1_val) if dim1 else "All Data"
        if not totals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=CTX)
            ax.axis("off")
            continue
        sectors = sorted(totals, key=lambda s: -totals[s])
        ax.pie([totals[s] for s in sectors],
               colors=[SECTOR_COLORS.get(s, "#888888") for s in sectors],
               autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
               wedgeprops={"linewidth": 1.8, "edgecolor": CBKG},
               startangle=90, pctdistance=0.75,
               textprops={"color": "white", "fontsize": 10, "fontweight": "bold"})
        ax.set_title(f"{title}\n({len(sub)} runs)", color=FG, fontsize=12,
                     fontweight="bold", pad=10)
    for idx in range(n_pies, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    handles = [mpatches.Patch(color=SECTOR_COLORS.get(s, "#888"), label=s)
               for s in SECTOR_COLORS if s in used]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
                   facecolor=PANEL_BG, labelcolor=FG, framealpha=0.9, edgecolor=SEP,
                   bbox_to_anchor=(0.5, 0))
    mode_label = "normalised" if weight_mode == "Average per row" else "summed"
    suptitle = f"Sector Allocation ({mode_label})"
    if dim1:
        suptitle += f"  ·  split by {dim1}"
    fig.suptitle(suptitle, color=FG, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig


# ── Sector profit-contribution ────────────────────────────────────────────────

def render_profit(df, dim1="Model"):
    d1_vals = dim_values(df, dim1) if dim1 else [None]
    group_data = [compute_sector_contributions(mask(df, dim1, v) if dim1 else df)
                  for v in d1_vals]
    sectors_seen: list[str] = []
    for gd in group_data:
        for s in gd:
            if s not in sectors_seen:
                sectors_seen.append(s)
    sectors_seen.sort(
        key=lambda s: -np.mean([gd[s]["allocation"] for gd in group_data if s in gd] or [0]))

    n = len(d1_vals)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = _hl_new_fig(rows, cols,
                            figsize=(max(6 * cols, 9), max(5.5 * rows, 5)), squeeze=False)
    x_pos = np.arange(len(sectors_seen))
    bw = 0.35
    for idx, (d1_val, contrib) in enumerate(zip(d1_vals, group_data)):
        ax = axes[idx // cols][idx % cols]
        _hl_style_ax(ax)
        alloc = [contrib.get(s, {}).get("allocation", 0) for s in sectors_seen]
        profit = [contrib.get(s, {}).get("profit", 0) for s in sectors_seen]
        ratios = [contrib.get(s, {}).get("ratio", 0) for s in sectors_seen]
        sec_colors = [SECTOR_COLORS.get(s, "#888888") for s in sectors_seen]
        ax.bar(x_pos - bw / 2, alloc, width=bw, color=sec_colors, alpha=0.75,
               edgecolor="#ffffff30", linewidth=0.5, label="Allocation %", zorder=3)
        ax.bar(x_pos + bw / 2, profit, width=bw,
               color=["#59A14F" if v >= 0 else "#E15759" for v in profit],
               alpha=0.85, edgecolor="#ffffff30", linewidth=0.5,
               label="Profit Contribution %", zorder=3)
        for xi, (av, pv, ratio) in enumerate(zip(alloc, profit, ratios)):
            if av == 0:
                continue
            ax.text(xi, max(av, pv, 0) + 0.5, f"×{ratio:.2f}", ha="center", va="bottom",
                    fontsize=9, color="#C99A00" if ratio >= 1 else "#FF9DA7",
                    fontweight="bold")
        ax.axhline(0, color="#666688", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sectors_seen, rotation=22, ha="right", color=CTX, fontsize=10)
        ax.set_ylabel("Normalised share  (%)", color=CTX, fontsize=11)
        ax.set_title(dim_label(dim1, d1_val) if dim1 else "All Data",
                     color=FG, fontsize=12, fontweight="bold", pad=8)
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    leg = [mpatches.Patch(color="#4E79A7", label="Allocation %"),
           mpatches.Patch(color="#59A14F", label="Profit Contribution %  (red = loss)")]
    fig.legend(handles=leg, loc="lower center", ncol=2, facecolor=PANEL_BG,
               labelcolor=FG, framealpha=0.9, edgecolor=SEP, bbox_to_anchor=(0.5, 0))
    suptitle = "Sector Profit Contribution vs Allocation"
    if dim1:
        suptitle += f"  ·  split by {dim1}"
    fig.suptitle(suptitle, color=FG, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    return fig


# ── Statistics summary table ──────────────────────────────────────────────────

def render_statistics(df, metric="ROI %", group_dim="Model", split_dim=None):
    split_vals = dim_values(df, split_dim) if split_dim else [None]
    group_vals = dim_values(df, group_dim)
    all_blocks = []
    for split_val in split_vals:
        sub = mask(df, split_dim, split_val) if split_dim else df
        rows_stat = []
        for gv in group_vals:
            vals = mask(sub, group_dim, gv)[metric].dropna().tolist()
            if len(vals) < 2:
                continue
            arr = np.array(vals, dtype=float)
            mean, std = float(arr.mean()), float(arr.std(ddof=1))
            se = std / np.sqrt(len(arr))
            try:
                from scipy import stats as _sp
                ci_lo, ci_hi = _sp.t.interval(0.95, df=len(arr) - 1, loc=mean, scale=se)
            except Exception:
                ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
            cv = (std / abs(mean) * 100) if mean != 0 else float("nan")
            rows_stat.append({"label": lbl(gv), "n": len(arr), "mean": mean, "std": std,
                              "se": se, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "cv": cv})
        all_blocks.append((split_val, rows_stat))

    n_blocks = len(all_blocks)
    max_rows = max((len(s) for _, s in all_blocks), default=1)
    fig_h = max(4.0, 1.8 * n_blocks + 0.55 * max_rows * n_blocks)
    fig, axes_arr = _hl_new_fig(n_blocks, 1, figsize=(13, fig_h), squeeze=False)
    headers = [group_dim, "n", "Mean", "Std Dev", "Std Error",
               "95% CI Low", "95% CI High", "CV %"]
    for bi, (split_val, rows_stat) in enumerate(all_blocks):
        ax = axes_arr[bi][0]
        ax.set_facecolor(CBKG)
        ax.axis("off")
        ax.set_title(f"{dim_label(split_dim, split_val) if split_dim else 'All Data'}"
                     f"  ·  {metric} grouped by {group_dim}",
                     color=FG, fontsize=12, fontweight="bold", pad=10, loc="left")
        if not rows_stat:
            ax.text(0.5, 0.5, "Insufficient data (need n ≥ 2)", ha="center", va="center",
                    color=CTX, transform=ax.transAxes)
            continue
        valid_cvs = [r["cv"] for r in rows_stat if not np.isnan(r["cv"])]
        max_cv = max(valid_cvs) if valid_cvs else 1.0
        cell_text, cell_colors = [], []
        for r in rows_stat:
            cv_val = r["cv"]
            if np.isnan(cv_val) or max_cv == 0:
                cv_bg = CAXES
            else:
                ratio = 1.0 - min(cv_val, max_cv) / max_cv
                cv_bg = (0.10, 0.30 + 0.40 * ratio, 0.15)
            mean_bg = (0.10, 0.28, 0.15) if r["mean"] >= 0 else (0.28, 0.10, 0.10)
            cell_text.append([r["label"], str(r["n"]), f"{r['mean']:+.2f}%",
                              f"{r['std']:.2f}%", f"{r['se']:.2f}%", f"{r['ci_lo']:+.2f}%",
                              f"{r['ci_hi']:+.2f}%",
                              f"{cv_val:.1f}%" if not np.isnan(cv_val) else "—"])
            cell_colors.append([CAXES, CAXES, mean_bg, CAXES, CAXES, CAXES, CAXES, cv_bg])
        tbl = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.auto_set_column_width(list(range(len(headers))))
        tbl.scale(1, 1.6)
        for j in range(len(headers)):
            c = tbl[0, j]
            c.set_facecolor(ACCENT)
            c.set_text_props(color="white", fontweight="bold")
        for i in range(len(rows_stat)):
            for j in range(len(headers)):
                tbl[i + 1, j].set_text_props(color="white" if j in (2, 7) else FG)
    fig.suptitle(f"Statistical Summary  ·  {metric}  ·  grouped by {group_dim}"
                 + (f"  (split by {split_dim})" if split_dim else ""),
                 color=FG, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ── Bubble chart  (NEW) ───────────────────────────────────────────────────────

def _bubble_axis_value(axis_spec, sub_df, filter_value):
    """Resolve a bubble axis value.  ``axis_spec`` is either a metric column name
    (→ mean of that column) or a filter-size alias (→ parsed stage size)."""
    key = str(axis_spec).strip().lower().replace(" ", "").replace("_", "")
    if key in _FILTER_AXIS_ALIASES:
        initial, final = parse_filter_sizes(filter_value)
        return initial if _FILTER_AXIS_ALIASES[key] == "initial" else final
    series = sub_df[axis_spec].dropna().astype(float)
    return float(series.mean()) if len(series) else None


def render_bubble(df, x="final", y="initial", size="ROI %", size_agg="Mean",
                  group_dim="Filter", color_dim=None, jitter=0.10, stack=False,
                  title=None, color_map=None, note=None):
    """Bubble chart.  Default encodes the two-stage filter grid:
        x = Stage-Two (final) size, y = Stage-One (initial) size, bubble = mean ROI.
    Any axis may instead be a metric name; ``color_dim`` adds a categorical series.
    With ``stack=True`` and a ``color_dim``, the per-series bubbles for each cell
    are drawn concentrically (largest behind) instead of jittered apart, so the
    series can be compared in place — e.g. one nested stack of models per config.
    """
    color_vals = dim_values(df, color_dim) if color_dim else [None]
    group_vals = dim_values(df, group_dim)
    palette = {v: PLOT_COLORS[i % len(PLOT_COLORS)] for i, v in enumerate(color_vals)}

    points = []   # (x, y, size_val, color_val, group_label)
    skipped = []
    for cv in color_vals:
        csub = mask(df, color_dim, cv) if color_dim else df
        for gv in group_vals:
            sub = mask(csub, group_dim, gv)
            if sub.empty:
                continue
            xv = _bubble_axis_value(x, sub, gv)
            yv = _bubble_axis_value(y, sub, gv)
            if xv is None or yv is None:
                skipped.append(lbl(gv))
                continue
            if str(size).strip().lower() in ("count", "n"):
                sv = float(len(sub))
            else:
                sv = agg_apply(sub[size].dropna(), size_agg)
            if sv is None or np.isnan(sv):
                continue
            points.append((xv, yv, sv, cv, lbl(gv)))

    fig, ax = _hl_new_fig(figsize=(10, 6.8))
    _hl_style_ax(ax)
    if not points:
        ax.text(0.5, 0.5, "No bubbles to plot (check axis/group settings)",
                ha="center", va="center", color=CTX, transform=ax.transAxes)
        return fig

    sizes = np.array([p[2] for p in points], dtype=float)
    s_min, s_max = float(np.nanmin(sizes)), float(np.nanmax(sizes))

    def _area(v):
        if s_max == s_min:
            return 900.0
        return 200.0 + (v - s_min) / (s_max - s_min) * 2600.0

    pct = "%" if "%" in str(size) else ""   # append to value labels for percent metrics

    # Filter-size axes are discrete and unevenly spaced (3, 5, 10, 15, ...).  On a
    # linear axis that leaves large empty bands, so map each distinct value to an
    # evenly spaced ordinal position and keep the real sizes as tick labels.
    is_filter_axis = str(x).strip().lower().replace(" ", "").replace("_", "") in _FILTER_AXIS_ALIASES
    is_filter_y = str(y).strip().lower().replace(" ", "").replace("_", "") in _FILTER_AXIS_ALIASES
    x_uniq = sorted({p[0] for p in points})
    y_uniq = sorted({p[1] for p in points})
    x_at = {v: i for i, v in enumerate(x_uniq)}
    y_at = {v: i for i, v in enumerate(y_uniq)}
    xpos = (lambda v: x_at[v]) if is_filter_axis else (lambda v: v)
    ypos = (lambda v: y_at[v]) if is_filter_y else (lambda v: v)

    color_idx = {cv: i for i, cv in enumerate(color_vals)}
    if stack and color_dim:
        # Concentric stack: one nested cluster per cell, largest bubble behind.
        from collections import defaultdict
        cells = defaultdict(list)
        for (xv, yv, sv, cv, glabel) in points:
            cells[(xpos(xv), ypos(yv))].append((sv, cv, glabel))
        for (px, py), items in cells.items():
            items.sort(key=lambda t: -t[0])          # largest drawn behind
            for (sv, cv, glabel) in items:
                ax.scatter(px, py, s=_area(sv), alpha=1.0, zorder=3,
                           color=palette[cv], edgecolors="#ffffff", linewidths=1.3)
            r_max = (_area(items[0][0]) / np.pi) ** 0.5
            n_it = len(items)
            for k, (sv, cv, glabel) in enumerate(items):   # color-coded value column
                y_off = (n_it - 1) / 2.0 * 11.0 - k * 11.0
                ax.annotate(f"{sv:.0f}{pct}", (px, py), textcoords="offset points",
                            xytext=(r_max + 7, y_off), ha="left", va="center",
                            fontsize=8, fontweight="bold", color=palette[cv], zorder=7)
            ax.annotate(items[0][2], (px, py), textcoords="offset points",
                        xytext=(0, r_max + 4), ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color=FG, zorder=6)
    else:
        for (xv, yv, sv, cv, glabel) in points:
            px, py = xpos(xv), ypos(yv)
            # tiny horizontal jitter so overlapping categorical series stay readable
            jx = px + (color_idx[cv] - (len(color_vals) - 1) / 2) * jitter if color_dim else px
            bcolor = color_map.get(glabel, palette[cv]) if color_map else palette[cv]
            ax.scatter(jx, py, s=_area(sv), alpha=0.7, zorder=3,
                       color=bcolor, edgecolors="#0F172A", linewidths=0.8)
            ax.annotate(f"{glabel}\n{sv:.1f}{pct}", (jx, py), ha="center", va="center",
                        fontsize=7.5, color=FG, zorder=4)

    _top_pad = 0.4 if (stack and color_dim) else 0.0
    if is_filter_axis:
        ax.set_xticks(range(len(x_uniq)))
        ax.set_xticklabels([f"{int(v)}" for v in x_uniq])
        ax.set_xlim(-0.7, len(x_uniq) - 0.3)
    if is_filter_y:
        ax.set_yticks(range(len(y_uniq)))
        ax.set_yticklabels([f"{int(v)}" for v in y_uniq])
        ax.set_ylim(-0.7, len(y_uniq) - 0.3 + _top_pad)

    xlabel = "Stage-Two output size (final)" if is_filter_axis else f"Mean {x}"
    ylabel = "Stage-One pool size (initial)" if is_filter_y else f"Mean {y}"
    size_label = ("number of runs" if str(size).strip().lower() in ("count", "n")
                  else f"{size_agg.lower()} {size}")
    ax.set_xlabel(xlabel, color=CTX, fontsize=11)
    ax.set_ylabel(ylabel, color=CTX, fontsize=11)
    title_extra = (f"stacked by {color_dim}" if (stack and color_dim)
                   else f"one bubble per {group_dim}")
    ax.set_title(title or f"Bubble — area ∝ {size_label}  ·  {title_extra}",
                 color=FG, fontsize=13, fontweight="bold", loc="left", pad=10)

    if note:   # e.g. an unfiltered-baseline reference that has no grid position
        ax.text(0.985, 0.985, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=CTX, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec=SEP, alpha=0.9))

    # Size legend (three representative bubbles).  Legend markers use their own
    # capped scale so the largest reference bubble stays inside the legend frame
    # (matplotlib sizes legend rows by font height, not marker size), while
    # handleheight/labelspacing give each row enough room to contain the marker.
    legend_vals = sorted({s_min, (s_min + s_max) / 2, s_max})

    def _legend_area(v):
        if s_max == s_min:
            return 90.0
        return 40.0 + (v - s_min) / (s_max - s_min) * 190.0

    size_handles = [plt.scatter([], [], s=_legend_area(v), color="#94A3B8",
                                edgecolors="#0F172A", alpha=0.55,
                                label=f"{v:.1f}") for v in legend_vals]
    leg1 = ax.legend(handles=size_handles, title=size_label, loc="upper left",
                     frameon=True, fontsize=9, labelspacing=1.1, borderpad=0.9,
                     handletextpad=1.2, handleheight=2.0,
                     facecolor=PANEL_BG, labelcolor=FG, edgecolor=SEP)
    leg1.get_title().set_color(FG)
    leg1.get_title().set_fontsize(9)
    ax.add_artist(leg1)
    if color_dim:
        col_handles = [mpatches.Patch(color=palette[cv], label=lbl(cv)) for cv in color_vals]
        ax.legend(handles=col_handles, title=color_dim, loc="upper right",
                  facecolor=PANEL_BG, labelcolor=FG, edgecolor=SEP)
    if skipped:
        ax.text(0.99, 0.01, "excluded (no X→Y sizes): " + ", ".join(sorted(set(skipped))),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=CTX, alpha=0.8)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Headless CLI
# ──────────────────────────────────────────────────────────────────────────────

_CHART_DISPATCH = {
    "bar": "render_bar", "box": "render_box", "pie": "render_pie",
    "profit": "render_profit", "stats": "render_statistics", "bubble": "render_bubble",
}
_DEFAULT_OUT = {
    "bar": "chart_bar.png", "box": "chart_box.png", "pie": "chart_pie.png",
    "profit": "chart_profit.png", "stats": "chart_stats.png", "bubble": "chart_bubble.png",
}


def _apply_cli_filters(df, filter_specs, period):
    if period is not None:
        df = df[df["Period (mo)"] == period].copy()
    for spec in filter_specs or []:
        if "=" not in spec:
            continue
        dim, vals = spec.split("=", 1)
        dim = dim.strip()
        if dim not in df.columns:
            continue
        keep = {v.strip() for v in vals.split(",")}
        df = df[df[dim].apply(lambda x: lbl(x) in keep)]
    return df


def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Investment Strategy Chart Visualizer — GUI by default, "
                    "or headless chart export via --headless/--chart.")
    p.add_argument("--export-paper-figures", action="store_true",
                   help="Regenerate the fixed set of paper figures into results/figures/.")
    p.add_argument("--headless", action="store_true",
                   help="Render a single chart without launching the GUI.")
    p.add_argument("--chart", choices=list(_CHART_DISPATCH),
                   help="Chart type to render in headless mode.")
    p.add_argument("--metric", default="ROI %")
    p.add_argument("--agg", default="Mean")
    p.add_argument("--dim1"); p.add_argument("--dim2"); p.add_argument("--dim3"); p.add_argument("--dim4")
    p.add_argument("--group"); p.add_argument("--split")
    p.add_argument("--plot-style", default="Box Plot",
                   choices=["Box Plot", "Violin", "Box + Violin"])
    p.add_argument("--weight", default="Sum of allocations",
                   choices=["Sum of allocations", "Average per row"])
    # bubble axes
    p.add_argument("--x", default="final"); p.add_argument("--y", default="initial")
    p.add_argument("--size", default="ROI %"); p.add_argument("--color")
    p.add_argument("--stack", action="store_true",
                   help="Bubble: stack the colour series concentrically per cell "
                        "(largest behind) instead of jittering them apart.")
    p.add_argument("--title", help="Bubble: override the chart title.")
    # data subsetting / styling
    p.add_argument("--period", type=int, default=None)
    p.add_argument("--filter", action="append", default=[],
                   help="Dim=val[,val...]  (repeatable), e.g. --filter Model=gpt-5.1")
    p.add_argument("--theme", choices=["light", "dark"], default="light")
    p.add_argument("--out", default=None)
    p.add_argument("--dpi", type=int, default=200)
    return p


def run_headless(args):
    hl_set_theme(dark=(args.theme == "dark"))
    df = _apply_cli_filters(load_data(), args.filter, args.period)
    if df.empty:
        print("No rows match the current filters.", file=sys.stderr)
        return 1

    chart = args.chart
    if chart == "bar":
        fig = render_bar(df, args.metric, args.agg,
                         args.dim1, args.dim2 or "Filter", args.dim3 or "Model", args.dim4)
    elif chart == "box":
        fig = render_box(df, args.metric, args.plot_style,
                         args.dim1 or "Temperature", args.dim2 or "Model")
    elif chart == "pie":
        fig = render_pie(df, args.dim1 or "Model", args.weight)
    elif chart == "profit":
        fig = render_profit(df, args.dim1 or "Model")
    elif chart == "stats":
        fig = render_statistics(df, args.metric, args.group or "Model", args.split)
    elif chart == "bubble":
        fig = render_bubble(df, x=args.x, y=args.y, size=args.size,
                            size_agg=args.agg, group_dim=args.group or "Filter",
                            color_dim=args.color, stack=args.stack, title=args.title)
    else:
        print("No --chart specified.", file=sys.stderr)
        return 2

    out = args.out or _DEFAULT_OUT[chart]
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(out)
    return 0


if __name__ == "__main__":
    if "--export-paper-figures" in sys.argv:
        export_paper_figures()
    elif "--headless" in sys.argv or "--chart" in sys.argv:
        _args = _build_arg_parser().parse_args()
        sys.exit(run_headless(_args))
    else:
        app = App()
        app.mainloop()

# end of chart_visualizer.py
