"""
Investment Strategy Results Viewer

Browse, filter, and inspect all investment strategy result JSON files.
Supports single-run, sliding-window, and legacy BASE formats.
"""

import tkinter as tk
from tkinter import ttk
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Any


class ResultEntry:
    """Parsed metadata for a single result file (or one period within a sliding-window file)."""

    def __init__(self):
        self.filepath: str = ""
        self.filename: str = ""
        self.run_type: str = ""          # single / sliding_window / legacy
        self.model: str = "N/A"
        self.temperature: str = "N/A"
        self.sector: str = "N/A"
        self.timestamp: str = "N/A"
        self.analysis_window: str = "N/A"  # "start - end"
        self.portfolio_return: str = "N/A"
        self.alpha: str = "N/A"
        self.period_label: str = ""       # e.g. "Period 3" for sliding window sub-entries
        self.raw_data: Dict = {}


def discover_files(results_dir: str) -> List[str]:
    """List all .json files in the results/ folder (non-recursive)."""
    if not os.path.isdir(results_dir):
        return []
    return sorted(
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".json")
    )


def _parse_timestamp_from_filename(filename: str) -> str:
    """Try to extract a datetime from filenames like *_20260208_123259.json."""
    m = re.search(r"(\d{8})_(\d{6})\.json$", filename)
    if m:
        try:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    # SESSION format: SESSION_2025-10-05-11-02-38_...
    m2 = re.search(r"SESSION_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", filename)
    if m2:
        try:
            dt = datetime.strptime(m2.group(1), "%Y-%m-%d-%H-%M-%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return "N/A"


def _parse_sector_from_filename(filename: str) -> str:
    """Extract sector keyword from filename prefix (e.g. 'technology')."""
    base = os.path.basename(filename)
    # sector is the first word before _investment or _sliding
    m = re.match(r"^([a-zA-Z]+)_", base)
    return m.group(1) if m else "N/A"


def _fmt(val, suffix="%") -> str:
    if isinstance(val, (int, float)):
        return f"{val:.2f}{suffix}"
    return str(val) if val else "N/A"


def parse_file(filepath: str) -> List[ResultEntry]:
    """Parse a JSON result file into one or more ResultEntry objects."""
    entries: List[ResultEntry] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return entries

    filename = os.path.basename(filepath)
    fallback_ts = _parse_timestamp_from_filename(filename)
    fallback_sector = _parse_sector_from_filename(filename)

    # --- New-format sliding window (has metadata + results array) ---
    if isinstance(data, dict) and "metadata" in data and "results" in data:
        meta = data["metadata"]
        for i, period in enumerate(data["results"], 1):
            e = ResultEntry()
            e.filepath = filepath
            e.filename = filename
            e.run_type = "sliding_window"
            e.model = meta.get("model", "N/A")
            e.temperature = str(meta.get("temperature", "N/A"))
            e.sector = meta.get("sector", fallback_sector)
            e.timestamp = meta.get("timestamp", fallback_ts)
            if e.timestamp != "N/A" and "T" in str(e.timestamp):
                try:
                    e.timestamp = datetime.fromisoformat(str(e.timestamp)).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
            ps = period.get("period_start", "")
            pe = period.get("period_end", "")
            e.analysis_window = f"{ps} to {pe}" if ps else "N/A"
            e.period_label = f"Period {i}"
            ra = period.get("return_analysis", {})
            e.portfolio_return = _fmt(ra.get("main_return"))
            bench = ra.get("benchmark_comparison", {})
            e.alpha = _fmt(bench.get("alpha"))
            e.raw_data = period
            # Propagate parent-level warning into each period so the detail panel can show it
            if meta.get("training_data_warning"):
                e.raw_data.setdefault("metadata", {})["training_data_warning"] = meta["training_data_warning"]
            entries.append(e)
        return entries

    # --- Old-format sliding window (bare JSON array) ---
    if isinstance(data, list):
        for i, period in enumerate(data, 1):
            e = ResultEntry()
            e.filepath = filepath
            e.filename = filename
            e.run_type = "sliding_window"
            e.sector = fallback_sector
            e.timestamp = fallback_ts
            ps = period.get("period_start", "")
            pe = period.get("period_end", "")
            e.analysis_window = f"{ps} to {pe}" if ps else "N/A"
            e.period_label = f"Period {i}"
            ra = period.get("return_analysis", {})
            e.portfolio_return = _fmt(ra.get("main_return"))
            bench = ra.get("benchmark_comparison", {})
            e.alpha = _fmt(bench.get("alpha"))
            e.raw_data = period
            entries.append(e)
        return entries

    # --- Single dict result ---
    e = ResultEntry()
    e.filepath = filepath
    e.filename = filename
    e.raw_data = data

    meta = data.get("metadata", {})
    if meta:
        e.run_type = meta.get("run_type", "single")
        e.model = meta.get("model", "N/A")
        e.temperature = str(meta.get("temperature", "N/A"))
        e.sector = meta.get("sector", fallback_sector)
        ts = meta.get("timestamp", fallback_ts)
        if ts and "T" in str(ts):
            try:
                ts = datetime.fromisoformat(str(ts)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
        e.timestamp = ts
        asd = meta.get("analysis_start_date", "")
        aed = meta.get("analysis_end_date", "")
        e.analysis_window = f"{asd} to {aed}" if asd else "N/A"
    else:
        # Distinguish old single-run (has return_analysis at top) vs legacy BASE (has analysis_data wrapper)
        if "analysis_data" in data:
            e.run_type = "legacy"
        else:
            e.run_type = "single"
        e.sector = fallback_sector
        e.timestamp = fallback_ts
        ad = data.get("analysis_data", {})
        asum = ad.get("analysis_summary", {}) if ad else data.get("analysis_summary", {})
        ap = asum.get("analysis_period", "")
        if ap:
            e.analysis_window = ap
        elif asum.get("implementation_date"):
            e.analysis_window = f"ending {asum['implementation_date']}"

    ra = data.get("return_analysis", {})
    e.portfolio_return = _fmt(ra.get("main_return"))
    bench = ra.get("benchmark_comparison", {})
    e.alpha = _fmt(bench.get("alpha"))
    entries.append(e)
    return entries


class ResultsViewerApp:
    COLUMNS = ("filename", "type", "period", "timestamp", "model", "sector", "window", "return", "alpha")
    HEADERS = ("File", "Type", "Period", "Timestamp", "Model", "Sector", "Analysis Window", "Return", "Alpha")
    COL_WIDTHS = (220, 90, 70, 145, 100, 90, 180, 80, 80)

    def __init__(self, root: tk.Tk, project_dir: str):
        self.root = root
        self.project_dir = project_dir
        self.all_entries: List[ResultEntry] = []
        self.filtered_entries: List[ResultEntry] = []
        self._sort_col = "timestamp"
        self._sort_reverse = True

        self.root.title("Investment Strategy Results Viewer")
        self.root.geometry("1300x850")
        self.root.minsize(900, 600)

        self._build_filters()
        self._build_table()
        self._build_detail_panel()
        self._load_all()

    # ------------------------------------------------------------------ UI
    def _build_filters(self):
        frame = ttk.LabelFrame(self.root, text="Filters", padding=8)
        frame.pack(fill="x", padx=8, pady=(8, 0))

        # Row of filters
        ttk.Label(frame, text="Sector:").grid(row=0, column=0, padx=(0, 4))
        self.sector_var = tk.StringVar()
        self.sector_entry = ttk.Entry(frame, textvariable=self.sector_var, width=14)
        self.sector_entry.grid(row=0, column=1, padx=(0, 12))

        ttk.Label(frame, text="Type:").grid(row=0, column=2, padx=(0, 4))
        self.type_var = tk.StringVar(value="All")
        self.type_combo = ttk.Combobox(frame, textvariable=self.type_var, width=14,
                                       values=["All", "single", "sliding_window", "legacy"], state="readonly")
        self.type_combo.grid(row=0, column=3, padx=(0, 12))

        ttk.Label(frame, text="Model:").grid(row=0, column=4, padx=(0, 4))
        self.model_var = tk.StringVar(value="All")
        self.model_combo = ttk.Combobox(frame, textvariable=self.model_var, width=14, state="readonly")
        self.model_combo.grid(row=0, column=5, padx=(0, 12))

        ttk.Label(frame, text="From:").grid(row=0, column=6, padx=(0, 4))
        self.from_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.from_var, width=12).grid(row=0, column=7, padx=(0, 12))

        ttk.Label(frame, text="To:").grid(row=0, column=8, padx=(0, 4))
        self.to_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.to_var, width=12).grid(row=0, column=9, padx=(0, 12))

        ttk.Button(frame, text="Apply", command=self._apply_filters).grid(row=0, column=10, padx=(0, 4))
        ttk.Button(frame, text="Reset", command=self._reset_filters).grid(row=0, column=11, padx=(0, 4))
        ttk.Button(frame, text="Refresh", command=self._load_all).grid(row=0, column=12)

        self.status_var = tk.StringVar(value="Loading...")
        ttk.Label(frame, textvariable=self.status_var).grid(row=0, column=13, padx=(16, 0))

    def _build_table(self):
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True, padx=8, pady=4)

        self.tree = ttk.Treeview(container, columns=self.COLUMNS, show="headings", selectmode="browse")
        for col, hdr, w in zip(self.COLUMNS, self.HEADERS, self.COL_WIDTHS):
            self.tree.heading(col, text=hdr, command=lambda c=col: self._sort_by(c))
            self.tree.column(col, width=w, minwidth=50)

        vsb = ttk.Scrollbar(container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

    def _build_detail_panel(self):
        frame = ttk.LabelFrame(self.root, text="Details", padding=8)
        frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.detail_text = tk.Text(frame, wrap="word", height=16, font=("Consolas", 10))
        dsb = ttk.Scrollbar(frame, orient="vertical", command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=dsb.set)
        self.detail_text.pack(side="left", fill="both", expand=True)
        dsb.pack(side="right", fill="y")

    # ----------------------------------------------------------- data load
    def _load_all(self):
        filepaths = discover_files(self.project_dir)
        self.all_entries = []
        for fp in filepaths:
            self.all_entries.extend(parse_file(fp))

        # Populate model combo with discovered models
        models = sorted({e.model for e in self.all_entries if e.model != "N/A"})
        self.model_combo["values"] = ["All"] + models

        self._apply_filters()

    # ------------------------------------------------------------ filters
    def _apply_filters(self):
        entries = self.all_entries
        sector_q = self.sector_var.get().strip().lower()
        type_q = self.type_var.get()
        model_q = self.model_var.get()
        from_q = self.from_var.get().strip()
        to_q = self.to_var.get().strip()

        if sector_q:
            entries = [e for e in entries if sector_q in e.sector.lower()]
        if type_q != "All":
            entries = [e for e in entries if e.run_type == type_q]
        if model_q != "All":
            entries = [e for e in entries if e.model == model_q]
        if from_q:
            entries = [e for e in entries if e.timestamp >= from_q]
        if to_q:
            entries = [e for e in entries if e.timestamp <= to_q + "z"]

        self.filtered_entries = entries
        self._refresh_table()
        self.status_var.set(f"{len(entries)} result(s) shown ({len(self.all_entries)} total)")

    def _reset_filters(self):
        self.sector_var.set("")
        self.type_var.set("All")
        self.model_var.set("All")
        self.from_var.set("")
        self.to_var.set("")
        self._apply_filters()

    # ------------------------------------------------------------- table
    def _refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        for i, e in enumerate(self.filtered_entries):
            self.tree.insert("", "end", iid=str(i), values=(
                e.filename,
                e.run_type,
                e.period_label,
                e.timestamp,
                e.model,
                e.sector,
                e.analysis_window,
                e.portfolio_return,
                e.alpha,
            ))

    def _sort_by(self, col):
        reverse = self._sort_reverse if col == self._sort_col else False
        self._sort_col = col
        self._sort_reverse = not reverse

        idx = self.COLUMNS.index(col)

        def sort_key(entry):
            val = (entry.filename, entry.run_type, entry.period_label, entry.timestamp,
                   entry.model, entry.sector, entry.analysis_window,
                   entry.portfolio_return, entry.alpha)[idx]
            # Try numeric sort for return/alpha columns
            if col in ("return", "alpha"):
                try:
                    return float(val.replace("%", ""))
                except (ValueError, AttributeError):
                    return float("-inf")
            return str(val).lower()

        self.filtered_entries.sort(key=sort_key, reverse=self._sort_reverse)
        self._refresh_table()

    # ---------------------------------------------------------- details
    def _on_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        entry = self.filtered_entries[int(sel[0])]
        self._show_detail(entry)

    def _show_detail(self, entry: ResultEntry):
        self.detail_text.delete("1.0", "end")
        data = entry.raw_data
        lines: List[str] = []

        lines.append(f"FILE: {entry.filepath}")
        lines.append(f"TYPE: {entry.run_type}  |  MODEL: {entry.model}  |  TEMP: {entry.temperature}")
        lines.append(f"SECTOR: {entry.sector}  |  TIMESTAMP: {entry.timestamp}")
        lines.append(f"ANALYSIS WINDOW: {entry.analysis_window}")

        # Training-data overlap warning (from metadata or parent sliding-window metadata)
        meta = data.get("metadata", {})
        warning = meta.get("training_data_warning")
        if warning:
            lines.append("")
            lines.append("!" * 80)
            lines.append(f"WARNING: {warning}")
            lines.append("!" * 80)

        lines.append("=" * 80)

        # Strategy text
        strategy = data.get("investment_strategy", "")
        if isinstance(strategy, dict):
            strategy = json.dumps(strategy, indent=2)
        if strategy:
            lines.append("\nSTRATEGY:")
            lines.append("-" * 40)
            lines.append(str(strategy))

        # Company recommendations
        ad = data.get("analysis_data", {})
        companies = (data.get("final_recommendations") or
                     data.get("recommended_companies") or
                     (ad.get("recommended_companies") if ad else []) or
                     [])
        if companies:
            lines.append(f"\nRECOMMENDED COMPANIES ({len(companies)}):")
            lines.append("-" * 40)
            lines.append(f"{'#':<4}{'Ticker':<8}{'Company':<30}{'Alloc%':<8}{'Score':<8}{'Growth':<12}{'Volatility':<12}")
            lines.append("-" * 82)
            for i, c in enumerate(companies, 1):
                if not isinstance(c, dict):
                    continue
                ticker = c.get("ticker", "N/A")
                name = c.get("company_name", c.get("name", "N/A"))
                alloc = c.get("final_allocation", c.get("recommended_allocation", c.get("allocation", "N/A")))
                score = c.get("overall_score", c.get("confidence_score", "N/A"))
                growth = c.get("growth_assessment", "N/A")
                vol = c.get("volatility_assessment", "N/A")
                lines.append(f"{i:<4}{ticker:<8}{name:<30}{str(alloc):<8}{str(score):<8}{growth:<12}{vol:<12}")

        # Return analysis
        ra = data.get("return_analysis", {})
        if ra and not ra.get("error"):
            lines.append("\nRETURN ANALYSIS:")
            lines.append("-" * 40)
            lines.append(f"  Portfolio Return:  {_fmt(ra.get('main_return'))}")
            lines.append(f"  Total ROI:         {_fmt(ra.get('total_roi'))}")
            fv = ra.get("total_final_value", ra.get("final_portfolio_value"))
            if isinstance(fv, (int, float)):
                lines.append(f"  Final Value:       ${fv:,.2f}")

            # Individual returns
            ind = ra.get("individual_returns", {})
            if ind:
                lines.append("\n  Individual Stock Returns:")
                for ticker, ret in ind.items():
                    lines.append(f"    {ticker:<8} {_fmt(ret)}")

            # Benchmark
            bench = ra.get("benchmark_comparison", {})
            if bench:
                lines.append(f"\n  Benchmark (Market): {_fmt(bench.get('market_return'))}")
                lines.append(f"  Alpha:              {_fmt(bench.get('alpha'))}")
                lines.append(f"  Outperformed:       {bench.get('outperformed', 'N/A')}")

        # Portfolio summary
        ps = data.get("portfolio_summary", {})
        if ps:
            lines.append("\nPORTFOLIO SUMMARY:")
            lines.append("-" * 40)
            lines.append(f"  Expected Return: {ps.get('expected_return', ps.get('expected_annual_return', 'N/A'))}")
            lines.append(f"  Risk Level:      {ps.get('risk_level', 'N/A')}")
            lines.append(f"  Focus:           {ps.get('strategy_focus', 'N/A')}")

        # Investment parameters
        ip = data.get("investment_parameters", {})
        if ip:
            inv = ip.get("initial_investment", ip.get("investment_amount"))
            if isinstance(inv, (int, float)):
                lines.append(f"\nINVESTMENT: ${inv:,.2f}")

        self.detail_text.insert("1.0", "\n".join(lines))


def main():
    root = tk.Tk()

    # Point at the dedicated results/ folder (sibling of SCCUR_TESTS/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(project_dir, "results")

    ResultsViewerApp(root, results_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
