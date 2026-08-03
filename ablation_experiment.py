#!/usr/bin/env python3
"""
ablation_experiment.py — controlled component ablation for reviewer Major #10.

Rebuilds the filtered-vs-single-pass comparison around ONE fixed information set
per (model, window, temperature, candidate set X, final portfolio size N), so
that only the *structure* differs across arms — not the documents, the candidate
set, or the number of holdings. This is the airtight version of the paper's
central claim (structure, not context length, drives the gain).

One shared document set is built per cell: the window's news + reports for a
broad candidate pool (BROAD_POOL, default 30). The single-pass arm chooses from
the WHOLE pool; the staged arms use the top-X shortlist of it — so A1 is a genuine
from-scratch baseline, not a re-run over the pipeline's own shortlist. All arms end
at exactly N holdings.

SIX arms per cell. A1 draws from the broad pool; the rest operate on the FIXED
top-X shortlist so they share one candidate set and documents (the reviewer's
requirement). All end at N holdings except A6, which holds all X by design.

  A1  single_pool     — news + reports for the WHOLE broad pool in ONE prompt, pick
                        N from scratch. Carries MORE context than A2; if it still
                        loses, the gain is not context length.
  A2  sequential_cut  — news shortlists the top X; reports narrow X -> N. Treatment.
  A3  sequential_nocut— the X shortlist's reports shown, N selected in one step (no
                        staged cut).
  A4  reports_scrambled— like A2 but the X reports are shuffled (or withheld).
  A5  single_matched  — reviewer-literal single pass: ONE prompt with news + the SAME
                        X-shortlist reports (budget-matched to A2), pick N.
  A6  hold_all        — reviewer-literal "eliminate nobody": show the X reports, keep
                        ALL X (final size = X, not N).

Clean contrasts, all over the SAME fixed X shortlist unless noted:
  A5 vs A2  isolates ordering/packaging (one prompt vs staged, same documents).
  A2 vs A6  isolates the elimination step (cut X->N vs keep all X).
  A2 vs A4  isolates whether the reports carry real signal.
  A1 vs A2  broad-pool single pass vs the funnel — the context-length test.
Per-arm input/output token counts are logged so the context-length confound the
reviewer raised is measured directly.

This does NOT replace the paper's main runs; it is a targeted control. Scope it
small (flagship model, a few windows, the key configs) — see GRID at the bottom.

Run from the project root (needs OPENAI_API_KEY / the same env the main runner uses):

    python ablation_experiment.py

Outputs one JSON per cell under results/ablation/ plus a combined
results/ablation_summary.csv.

Reuses InvestmentStrategyGenerator's own helpers so the news processing, report
fetching, per-stock summaries, and return computation are byte-for-byte the same
as the main pipeline.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import random
import getpass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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

# Windows consoles default to a legacy codepage (often ASCII/cp1252); news and
# report text routinely contain smart quotes (’ “ ” –), so the pipeline's prints
# would crash with UnicodeEncodeError. Force UTF-8 (and never crash on a stray
# character) before any pipeline code runs.
for _stream in ("stdout", "stderr"):
    try:
        getattr(sys, _stream).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# API KEY.  You should NOT need to touch this file to supply the key. It is read
# (in order) from: the OPENAI_API_KEY env var; the key you already saved in the
# permutation runner (.permutation_runner_config.json); this script's own
# .ablation_config.json (the GUI saves your entry there); and finally a hidden
# prompt at startup. Because the key lives in those config files, re-downloading
# an updated version of THIS script never wipes it. (You may still hardcode one
# below if you insist, but then don't commit it.)
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("OPENAI_API_KEY", "")  # optional hardcode; leave empty to use env/config/prompt instead

from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "pipeline"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ENGINE))

import sector_config  # noqa: E402
from investment_strategy_generator import (  # noqa: E402
    InvestmentStrategyGenerator,
    _oai_create_with_retry,
)

OUT_DIR = ROOT / "results" / "ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNNER_CONFIG = ROOT / ".permutation_runner_config.json"      # shared with the runner
ABLATION_CONFIG = ROOT / ".ablation_config.json"               # this tool's own store


def _load_saved_key() -> str:
    for p in (ABLATION_CONFIG, RUNNER_CONFIG):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            k = (d.get("openai_api_key") or "").strip()
            if k:
                return k
        except Exception:
            pass
    return ""


def _save_key(key: str) -> None:
    if not key:
        return
    try:
        d = {}
        if ABLATION_CONFIG.exists():
            d = json.loads(ABLATION_CONFIG.read_text(encoding="utf-8"))
        d["openai_api_key"] = key
        ABLATION_CONFIG.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass


def _resolve_api_key() -> str:
    key = (API_KEY.strip() or os.environ.get("OPENAI_API_KEY", "").strip()
           or _load_saved_key())
    if not key:
        try:
            key = getpass.getpass("Enter your OpenAI API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            key = ""
    return key


def _json_default(o):
    """Convert numpy scalars (np.bool_, np.int64, np.float64, …) and other
    non-native objects so json.dumps never chokes on pandas/numpy return values."""
    if hasattr(o, "item"):        # numpy scalars expose .item() -> native python
        try:
            return o.item()
        except Exception:
            pass
    if isinstance(o, (set, frozenset)):
        return list(o)
    return str(o)

# Arm 3 operationalization. The reviewer's "eliminates nobody" is in slight tension
# with "keep the final number of holdings identical across all four": holding
# everyone makes final size = X. Default keeps N fixed (single selection, no staged
# cut). Flip to True for the literal hold-all-X arm and report it separately.
HOLD_ALL = False
# Size of the broad candidate pool the single-pass arm (A1) chooses from. A1 sees
# the whole pool at once; the staged arms (A2/A3/A4) use the top-X shortlist of it,
# so A1 is a genuine from-scratch single-pass rather than a re-run over X.
BROAD_POOL = 30
# Arm 4 mode: "withheld" (reports removed, news rationale only) or "shuffled"
# (each ticker shown another ticker's report). Shuffled is the stronger control.
A4_MODE = "shuffled"
A4_SEED = 20260720  # fixed so the shuffle is reproducible (no Math.random equivalent)


class AblationGenerator(InvestmentStrategyGenerator):
    """Subclass that exposes the shared front-half (candidate set + report bundle)
    once, then runs four final-selection prompts over it."""

    # ---- token accounting -------------------------------------------------
    def _call(self, model: str, prompt: str, tok: Dict[str, int],
              max_tokens: int = 3000) -> str:
        resp = _oai_create_with_retry(
            self.openai_client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **self._temp_kwargs(model),
            max_completion_tokens=max_tokens,
        )
        u = getattr(resp, "usage", None)
        if u is not None:
            tok["prompt"] += int(getattr(u, "prompt_tokens", 0) or 0)
            tok["completion"] += int(getattr(u, "completion_tokens", 0) or 0)
        return resp.choices[0].message.content or ""

    # ---- shared preamble: build the fixed information set ONCE -------------
    def build_shared_bundle(self, keyword: str, news_path: Optional[str],
                            news_start: str, news_end: str, X: int,
                            model: str, tok: Dict[str, int]) -> Optional[dict]:
        """Build ONE document set per cell: the window's news plus reports for a
        broad candidate pool. A1 (single-pass) sees the whole pool; the staged
        arms use the top-X shortlist of it, so A1 is a real from-scratch baseline
        rather than a re-run over the same shortlist the pipeline produced."""
        macro = self._fetch_macro_context(news_end) or ""
        news_df = self._load_news_data(news_path, news_start, news_end, keyword)
        if news_df is None or len(news_df) == 0:
            return None
        articles_text = self._process_news_data(news_df, keyword)

        # Stage One (once): rank a BROAD pool of U >= X candidates from news only.
        U = max(BROAD_POOL, X)
        stage_one_prompt = f"""
You are an expert investment analyst. From the news below, identify and RANK the
{U} publicly-traded companies in the {keyword} sector best supported by the
coverage (rank 1 = strongest). Base this only on the news; do not use financial
reports yet.

{macro}

NEWS:
{articles_text}

Return ONLY JSON:
{{"companies_for_analysis": [
   {{"rank": 1, "ticker": "AAPL", "company_name": "Apple Inc.", "rationale": "why from news"}}
]}}
"""
        content = self._call(model, stage_one_prompt, tok, max_tokens=3500)
        parsed = self._load_json_response(content, "four-arm stage one", model,
                                          max_completion_tokens=3500)
        cands = (parsed or {}).get("companies_for_analysis", []) or []
        cands = sorted(cands, key=lambda c: c.get("rank", 10**9))[:U]
        if len(cands) < max(2, X // 2):
            return None

        # Fetch reports + per-stock summaries for the whole pool (reuses the main
        # pipeline's helpers so summaries are identical to production).
        financial_data = self._fetch_financial_data_for_tickers(
            [c["ticker"] for c in cands], max_tickers=None)
        per_stock: Dict[str, dict] = {}
        ordered: List[str] = []
        for c in cands:
            t = c["ticker"]
            try:
                h1, h3, h5 = self._load_temporal_price_histories(t, news_end)
                if h1 is None or h1.empty:
                    continue
                perf = self._calculate_performance_metrics(h1, h3, h5)
                info = self._fetch_ticker_info(t)
                fin = None
                for fd in financial_data:
                    if fd.get("ticker") == t:
                        fin = self._filter_financial_data_as_of(fd, news_end)
                        break
                summary = self._generate_stock_summary(t, {
                    "name": c["company_name"], "performance_data": perf,
                    "financial_data": fin, "news_rationale": c["rationale"],
                    "info": info})
                per_stock[t] = {"company_name": c["company_name"],
                                "rationale": c["rationale"], "summary": summary}
                ordered.append(t)
            except Exception as e:  # noqa: BLE001
                print(f"   [bundle] {t}: {e}")
        if len(per_stock) < 2:
            return None
        # Top-X shortlist (by news rank) for the staged arms.
        shortlist = [t for t in ordered if t in per_stock][:X]
        return {"macro": macro, "articles_text": articles_text,
                "per_stock": per_stock, "pool": ordered, "shortlist": shortlist}

    # ---- report block builders (identical formatting to the main pipeline) --
    @staticmethod
    def _subset(per_stock: Dict[str, dict], tickers: List[str]) -> Dict[str, dict]:
        return {t: per_stock[t] for t in tickers if t in per_stock}

    @staticmethod
    def _reports_block(per_stock: Dict[str, dict],
                       mode: str = "full", seed: int = A4_SEED) -> str:
        items = list(per_stock.items())
        if mode == "shuffled":
            rng = random.Random(seed)
            summaries = [v["summary"] for _, v in items]
            rng.shuffle(summaries)
            items = [(t, {**v, "summary": s}) for (t, v), s in zip(items, summaries)]
        out = "COMPREHENSIVE COMPANY ANALYSIS:\n" + "=" * 50 + "\n\n"
        for t, v in items:
            out += f"{v['company_name']} ({t}):\n"
            out += f"News Rationale: {v['rationale']}\n"
            if mode != "withheld":
                out += v["summary"] + "\n"
            else:
                out += "(financial report withheld for this control)\n"
            out += "-" * 30 + "\n\n"
        return out

    @staticmethod
    def _final_schema(N: int) -> str:
        return (f'{{"final_recommendations": [\n'
                f'  {{"ticker": "TICKER", "company_name": "Name", '
                f'"final_allocation": 0.0, "allocation_rationale": "why"}}\n'
                f']}}  (exactly {N} holdings, allocations sum to 100)')

    def _select_prompt(self, keyword: str, macro: str, news_or_reports: str,
                       N: int, instruction: str) -> str:
        return f"""
You are an expert investment analyst constructing a long-only {keyword} portfolio.

{macro}

{news_or_reports}

{instruction}
Allocate percentages across the selected companies (must sum to 100%).
Return ONLY JSON:
{self._final_schema(N)}
"""

    # ---- the four arms ----------------------------------------------------
    def arm_single(self, kw, b, N, model, tok):
        # TRUE single-pass: full news + reports for the WHOLE pool, choose N from
        # scratch in one prompt. No pre-cut shortlist, no staged funnel.
        block = ("NEWS:\n" + b["articles_text"] + "\n\n"
                 + self._reports_block(b["per_stock"], "full"))
        instr = (f"Consider all {len(b['per_stock'])} candidate companies below. "
                 f"Using the news and their financial reports together, select the "
                 f"{N} best for the final portfolio in a single step.")
        return self._run_final(kw, b, block, N, instr, model, tok)

    def arm_sequential_cut(self, kw, b, N, model, tok):
        # Two-stage treatment: news already produced the top-X shortlist; reports
        # narrow that shortlist X -> N.
        sub = self._subset(b["per_stock"], b["shortlist"])
        block = self._reports_block(sub, "full")
        instr = (f"A first, news-only stage shortlisted these {len(sub)} companies "
                 f"from a larger pool. Using their financial reports, narrow the "
                 f"shortlist down to exactly {N} for the final portfolio.")
        return self._run_final(kw, b, block, N, instr, model, tok)

    def arm_sequential_nocut(self, kw, b, N, model, tok):
        sub = self._subset(b["per_stock"], b["shortlist"])
        block = self._reports_block(sub, "full")
        if HOLD_ALL:
            n_hold = len(sub)
            instr = (f"Rank all {n_hold} shortlisted companies using their reports, "
                     f"then hold ALL of them in the final portfolio (no elimination).")
            return self._run_final(kw, b, block, n_hold, instr, model, tok)
        instr = (f"Rank all {len(sub)} shortlisted companies using their reports, "
                 f"then in a single step select exactly {N} for the final portfolio "
                 f"(no intermediate pool-cut).")
        return self._run_final(kw, b, block, N, instr, model, tok)

    def arm_reports_scrambled(self, kw, b, N, model, tok):
        sub = self._subset(b["per_stock"], b["shortlist"])
        block = self._reports_block(sub, A4_MODE)
        instr = (f"Using the (control) report block for these {len(sub)} shortlisted "
                 f"companies, select exactly {N} for the final portfolio.")
        return self._run_final(kw, b, block, N, instr, model, tok)

    def arm_single_matched(self, kw, b, N, model, tok):
        # Reviewer-literal single-pass (R2 + R4a): ONE prompt holding exactly the
        # FIXED candidate set's documents — news + the X-shortlist's reports — and
        # nothing broader. Budget-matched to A2 at the document level (same news +
        # same X reports), so A5 vs A2 isolates ordering/packaging alone: one prompt
        # vs the staged news->reports path, over an identical shortlist.
        sub = self._subset(b["per_stock"], b["shortlist"])
        block = ("NEWS:\n" + b["articles_text"] + "\n\n"
                 + self._reports_block(sub, "full"))
        instr = (f"Consider exactly these {len(sub)} shortlisted companies. Using the "
                 f"news and their financial reports together in this single prompt, "
                 f"select the {N} best for the final portfolio.")
        return self._run_final(kw, b, block, N, instr, model, tok)

    def arm_hold_all(self, kw, b, N, model, tok):
        # Reviewer-literal "eliminates nobody" (R4c): show the X shortlist's reports
        # but keep ALL of them — final size = X, not N. The holdings count differs by
        # design (this is the reviewer's own arm; their "eliminate nobody" and "keep
        # holdings identical" cannot both be literal). A2 vs A6 isolates the
        # elimination step: cut X->N versus no cut at all.
        sub = self._subset(b["per_stock"], b["shortlist"])
        n_hold = len(sub)
        block = self._reports_block(sub, "full")
        instr = (f"Rank all {n_hold} shortlisted companies using their financial "
                 f"reports, then hold ALL {n_hold} of them in the final portfolio "
                 f"(eliminate nobody).")
        return self._run_final(kw, b, block, n_hold, instr, model, tok)

    def _run_final(self, kw, b, block, N, instr, model, tok):
        prompt = self._select_prompt(kw, b["macro"], block, N, instr)
        content = self._call(model, prompt, tok, max_tokens=4000)
        parsed = self._load_json_response(content, "four-arm final", model,
                                          max_completion_tokens=4000)
        recs = (parsed or {}).get("final_recommendations", []) or []
        return {"final_recommendations": recs}

    # ---- run one matched cell (all four arms) -----------------------------
    def run_cell(self, *, keyword, news_path, news_start, news_end,
                 analysis_start, analysis_end, X, N, temperature, model,
                 investment_amount=10000.0) -> dict:
        self.temperature = temperature
        vmodel = self._validate_model_for_date_range(analysis_start)
        shared_tok = {"prompt": 0, "completion": 0}
        bundle = self.build_shared_bundle(keyword, news_path, news_start,
                                          news_end, X, vmodel, shared_tok)
        if bundle is None:
            return {"error": "bundle_failed", "model": vmodel, "X": X, "N": N,
                    "analysis_start": analysis_start}

        arms = {
            "A1_single_pool": self.arm_single,             # broad-pool single pass
            "A2_sequential_cut": self.arm_sequential_cut,  # treatment (X -> N)
            "A3_sequential_nocut": self.arm_sequential_nocut,
            "A4_reports_scrambled": self.arm_reports_scrambled,
            "A5_single_matched": self.arm_single_matched,  # budget-matched single pass
            "A6_hold_all": self.arm_hold_all,              # eliminate nobody (holds X)
        }
        results = {}
        for name, fn in arms.items():
            tok = {"prompt": 0, "completion": 0}
            try:
                strat = fn(keyword, bundle, N, vmodel, tok)
                ret = self._calculate_comprehensive_returns(
                    strat, analysis_start, analysis_end,
                    investment_amount, 0.0, ["OVERALL"])
                results[name] = {
                    "final_recommendations": strat["final_recommendations"],
                    "return_analysis": ret,
                    # arm-only tokens PLUS the shared stage-one/report tokens, so
                    # each arm's total reflects the full information it consumed
                    "tokens": {"arm_prompt": tok["prompt"],
                               "arm_completion": tok["completion"],
                               "shared_prompt": shared_tok["prompt"],
                               "shared_completion": shared_tok["completion"],
                               "total": tok["prompt"] + tok["completion"]
                                        + shared_tok["prompt"] + shared_tok["completion"]},
                }
            except Exception as e:  # noqa: BLE001
                results[name] = {"error": str(e)}
        return {
            "metadata": {"model": vmodel, "keyword": keyword, "X": X, "N": N,
                         "temperature": temperature, "hold_all": HOLD_ALL,
                         "a4_mode": A4_MODE, "news_start": news_start,
                         "news_end": news_end, "analysis_start": analysis_start,
                         "analysis_end": analysis_end,
                         "pool": bundle["pool"], "shortlist": bundle["shortlist"]},
            "arms": results,
        }


# ---- grid + driver --------------------------------------------------------
# Training cutoffs (see revision: GPT-5.1 = Sep 30 2024, windows start Oct 2024;
# GPT-4o / GPT-4o-mini = Oct 2023, windows start Jun 2024).
TRAINING_CUTOFFS = {
    "gpt-4o": datetime(2023, 10, 1),
    "gpt-4o-mini": datetime(2023, 10, 1),
    "gpt-5.1": datetime(2024, 10, 1),
}
MODELS = ["gpt-5.1", "gpt-4o", "gpt-4o-mini"]
TEMPS = [0.3, 0.4, 0.5]
CONFIG_PAIRS = [(20, 10), (30, 5), (10, 5), (20, 5), (30, 15), (10, 3), (5, 3)]
# Candidate window starts (12-month), Jun 2024 .. Jun 2025.
ALL_STARTS = [datetime(2024, 6, 1) + relativedelta(months=i) for i in range(13)]
KEYWORD = (sector_config.active_sector()
           if hasattr(sector_config, "active_sector") else "technology")
NEWS_PATH = str(ROOT / "merged_news_data.csv")

SUMMARY_CSV = ROOT / "results" / "ablation_summary.csv"
CSV_FIELDS = ["model", "config", "window", "run", "arm", "roi", "total_tokens", "n_holdings"]


def _skip_reason(model: str, start: datetime) -> Optional[str]:
    cut = TRAINING_CUTOFFS.get(model)
    if cut and start < cut:
        return f"before {model} cutoff {cut:%Y-%m}"
    if start + relativedelta(months=12) > datetime.now():
        return "window not complete"
    return None


def build_cells(models, temps, configs, starts):
    """Enumerate runnable cells (skipping cutoff/incomplete windows)."""
    cells = []
    cid = 0
    for model in models:
        for temp in temps:
            for (X, N) in configs:
                for start in starts:
                    cid += 1
                    sk = _skip_reason(model, start)
                    cells.append({
                        "id": cid, "model": model, "temp": temp, "X": X, "N": N,
                        "start": start, "skip": sk,
                        "status": "Skipped" if sk else "Pending",
                        "runs": [], "attempts": 0, "runs_target": 1,
                        "note": sk or ""})
    return cells


def write_csv_rows(cell, res, run_idx):
    """Append one CSV row per arm for a single completed run (call from ONE thread
    only — the GUI does this on the main thread; the CLI is single-threaded)."""
    rows = []
    for arm, r in (res.get("arms", {}) or {}).items():
        roi = (r.get("return_analysis") or {}).get("total_roi")
        tok = (r.get("tokens") or {}).get("total")
        rows.append({"model": cell["model"], "config": f"{cell['X']}->{cell['N']}",
                     "window": f"{cell['start']:%Y-%m}", "run": run_idx + 1, "arm": arm,
                     "roi": roi, "total_tokens": tok,
                     "n_holdings": len(r.get("final_recommendations", []) or [])})
    if not rows:
        return
    write_header = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def run_one_cell(api_key, cell, run_idx=0):
    """Execute all arms for one cell ONCE and write a per-run JSON. Thread-safe:
    each call builds its own generator and writes a uniquely-named file. Does NOT
    write the CSV (the caller does that from a single thread). Returns the result."""
    start = cell["start"]
    a_start = start.strftime("%Y-%m-%d")
    a_end = (start + relativedelta(months=12)).strftime("%Y-%m-%d")
    n_start = (start - relativedelta(months=24)).strftime("%Y-%m-%d")
    gen = AblationGenerator(api_key_openai=api_key, temperature=cell["temp"],
                           model=cell["model"], initial_candidates=cell["X"],
                           final_portfolio=cell["N"])
    result = gen.run_cell(keyword=KEYWORD, news_path=NEWS_PATH,
                          news_start=n_start, news_end=a_start,
                          analysis_start=a_start, analysis_end=a_end,
                          X=cell["X"], N=cell["N"], temperature=cell["temp"],
                          model=cell["model"])
    tag = f"{cell['model'].replace('.','')}_{cell['X']}to{cell['N']}_{start:%Y%m}_run{run_idx + 1}"
    (OUT_DIR / f"arm_{tag}.json").write_text(
        json.dumps(result, indent=1, default=_json_default), encoding="utf-8")
    return result


# ============================ CLI mode =====================================
def run_cli():
    api_key = _resolve_api_key()
    if not api_key:
        print("No API key provided."); sys.exit(1)
    cells = build_cells(["gpt-5.1"], [0.3], [(20, 10)], ALL_STARTS)
    for c in cells:
        if c["skip"]:
            print(f"skip {c['id']}: {c['skip']}"); continue
        print(f"\n=== {c['model']} {c['X']}->{c['N']} {c['start']:%Y-%m} ===")
        res = run_one_cell(api_key, c, 0)
        write_csv_rows(c, res, 0)
        for arm, r in res.get("arms", {}).items():
            roi = (r.get("return_analysis") or {}).get("total_roi")
            tok = (r.get("tokens") or {}).get("total")
            print(f"   {arm:22s} ROI={roi}  tokens={tok}")
    print(f"\nSummary -> {SUMMARY_CSV}")


# ============================ GUI mode =====================================
def run_gui():
    import threading
    import queue as _queue
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox

    ROW_BG = {"Pending": "", "Skipped": "#fff3cd", "Running": "#cce5ff",
              "Done": "#d4edda", "Error": "#f8d7da"}

    class Worker(threading.Thread):
        """Runs a list of (cell, run_idx) tasks across a thread pool. Each task is
        one full pass of all arms for one cell. Results stream back on the queue."""
        def __init__(self, tasks, api_key, q, stop_evt, n_threads):
            super().__init__(daemon=True)
            self.tasks = tasks
            self.api_key, self.q, self.stop = api_key, q, stop_evt
            self.n_threads = max(1, int(n_threads))

        def _one(self, cell, run_idx):
            if self.stop.is_set():
                return {"error": "stopped"}
            try:
                return run_one_cell(self.api_key, cell, run_idx)
            except Exception as e:  # noqa: BLE001
                return {"error": str(e)[:200]}

        def run(self):
            from concurrent.futures import ThreadPoolExecutor, as_completed
            for cid in {c["id"] for c, _ in self.tasks}:
                self.q.put({"kind": "status", "id": cid, "status": "Running",
                            "note": "queued…"})
            with ThreadPoolExecutor(max_workers=self.n_threads) as ex:
                futs = {ex.submit(self._one, c, r): (c, r) for (c, r) in self.tasks}
                for fut in as_completed(futs):
                    c, r = futs[fut]
                    self.q.put({"kind": "run", "id": c["id"], "run_idx": r,
                                "res": fut.result()})
            self.q.put({"kind": "done"})

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Four-Arm Controlled Experiment (Major #10)")
            self.geometry("1180x760")
            self._q = _queue.Queue()
            self._worker = None
            self._stop = threading.Event()
            self._cells = []
            self._iid = {}
            self._results = {}   # sig -> {status, runs, attempts, runs_target, note}
            self._model_vars = {m: tk.BooleanVar(value=(m == "gpt-5.1")) for m in MODELS}
            self._temp_vars = {t: tk.BooleanVar(value=(t == 0.3)) for t in TEMPS}
            self._cfg_vars = {p: tk.BooleanVar(value=(p == (20, 10))) for p in CONFIG_PAIRS}
            _ds = [d.strftime("%Y-%m") for d in ALL_STARTS]
            self._date_from = tk.StringVar(value=_ds[0])
            self._date_to = tk.StringVar(value=_ds[-1])
            self._holdall = tk.BooleanVar(value=HOLD_ALL)
            self._a4mode = tk.StringVar(value=A4_MODE)
            self._key_var = tk.StringVar(value=(os.environ.get("OPENAI_API_KEY", "")
                                                or API_KEY or _load_saved_key()))
            self._build_ui()
            self._rebuild_cells()
            self.after(250, self._drain)

        def _build_ui(self):
            bar = ttk.Frame(self, padding=(8, 6)); bar.pack(fill="x")
            self._btn_start = ttk.Button(bar, text="▶  Start", command=self._on_start)
            self._btn_stop = ttk.Button(bar, text="⏹  Stop", command=self._on_stop, state="disabled")
            self._btn_start.pack(side="left", padx=2); self._btn_stop.pack(side="left", padx=2)
            ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
            self._stats = tk.StringVar()
            ttk.Label(bar, textvariable=self._stats).pack(side="left", padx=6)
            ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
            ttk.Label(bar, text="Threads:").pack(side="left")
            self._threads_var = tk.IntVar(value=1)
            ttk.Spinbox(bar, from_=1, to=8, width=3,
                        textvariable=self._threads_var).pack(side="left", padx=(2, 8))
            ttk.Label(bar, text="Runs/cell:").pack(side="left")
            self._runs_var = tk.IntVar(value=1)
            ttk.Spinbox(bar, from_=1, to=20, width=3,
                        textvariable=self._runs_var).pack(side="left", padx=2)
            ttk.Label(bar, text="OpenAI API key:").pack(side="right", padx=(10, 2))
            ttk.Entry(bar, textvariable=self._key_var, width=46, show="•").pack(side="right")

            f = ttk.LabelFrame(self, text="Cells  (applied at Start)", padding=(8, 4))
            f.pack(fill="x", padx=6, pady=(0, 4))
            r1 = ttk.Frame(f); r1.pack(fill="x", pady=2)
            ttk.Label(r1, text="Models:").pack(side="left")
            for m in MODELS:
                ttk.Checkbutton(r1, text=m, variable=self._model_vars[m],
                                command=self._rebuild_cells).pack(side="left", padx=3)
            ttk.Separator(r1, orient="vertical").pack(side="left", fill="y", padx=8)
            ttk.Label(r1, text="Temps:").pack(side="left")
            for t in TEMPS:
                ttk.Checkbutton(r1, text=str(t), variable=self._temp_vars[t],
                                command=self._rebuild_cells).pack(side="left", padx=3)
            r2 = ttk.Frame(f); r2.pack(fill="x", pady=2)
            ttk.Label(r2, text="Configs (X→N):").pack(side="left")
            for p in CONFIG_PAIRS:
                ttk.Checkbutton(r2, text=f"{p[0]}→{p[1]}", variable=self._cfg_vars[p],
                                command=self._rebuild_cells).pack(side="left", padx=3)
            r3 = ttk.Frame(f); r3.pack(fill="x", pady=2)
            ttk.Label(r3, text="Window start:").pack(side="left")
            _ds = [d.strftime("%Y-%m") for d in ALL_STARTS]
            ttk.Combobox(r3, textvariable=self._date_from, values=_ds, width=9,
                         state="readonly").pack(side="left", padx=2)
            ttk.Label(r3, text="to").pack(side="left")
            ttk.Combobox(r3, textvariable=self._date_to, values=_ds, width=9,
                         state="readonly").pack(side="left", padx=2)
            for w in (r3.winfo_children()[1], r3.winfo_children()[3]):
                w.bind("<<ComboboxSelected>>", lambda e: self._rebuild_cells())
            ttk.Separator(r3, orient="vertical").pack(side="left", fill="y", padx=8)
            ttk.Checkbutton(r3, text="Arm 3 = hold all X (no final cut)",
                            variable=self._holdall).pack(side="left", padx=3)
            ttk.Label(r3, text="Arm 4:").pack(side="left", padx=(8, 2))
            ttk.Combobox(r3, textvariable=self._a4mode, values=["shuffled", "withheld"],
                         width=9, state="readonly").pack(side="left")

            pbf = ttk.Frame(self, padding=(8, 2)); pbf.pack(fill="x")
            self._pb = tk.DoubleVar(value=0.0)
            ttk.Progressbar(pbf, variable=self._pb, maximum=100).pack(fill="x", side="left", expand=True)

            pw = ttk.PanedWindow(self, orient="vertical"); pw.pack(fill="both", expand=True, padx=6, pady=4)
            tbl = ttk.Frame(pw); pw.add(tbl, weight=4)
            COLS = ("#", "Model", "Config", "Window",
                    "A1 pool", "A2 cut", "A3 nocut", "A4 scram",
                    "A5 matched", "A6 holdall", "Status", "Note")
            W = {"#": 38, "Model": 88, "Config": 60, "Window": 70, "A1 pool": 66,
                 "A2 cut": 62, "A3 nocut": 68, "A4 scram": 66, "A5 matched": 76,
                 "A6 holdall": 74, "Status": 66, "Note": 240}
            self._tree = ttk.Treeview(tbl, columns=COLS, show="headings", selectmode="browse")
            for c in COLS:
                self._tree.heading(c, text=c)
                self._tree.column(c, width=W.get(c, 80),
                                  anchor="w" if c == "Note" else "center",
                                  stretch=(c == "Note"))
            ys = ttk.Scrollbar(tbl, orient="vertical", command=self._tree.yview)
            self._tree.configure(yscrollcommand=ys.set)
            ys.pack(side="right", fill="y"); self._tree.pack(fill="both", expand=True)
            for st, bg in ROW_BG.items():
                if bg:
                    self._tree.tag_configure(st, background=bg)
            logf = ttk.LabelFrame(pw, text="Log", padding=4); pw.add(logf, weight=1)
            self._log = scrolledtext.ScrolledText(logf, height=8, state="disabled",
                                                  font=("Courier", 9), wrap="word")
            self._log.pack(fill="both", expand=True)

        # -- helpers --
        def _logln(self, s):
            self._log.configure(state="normal"); self._log.insert("end", s + "\n")
            self._log.see("end"); self._log.configure(state="disabled")

        def _selected(self):
            models = [m for m in MODELS if self._model_vars[m].get()]
            temps = [t for t in TEMPS if self._temp_vars[t].get()]
            cfgs = [p for p in CONFIG_PAIRS if self._cfg_vars[p].get()]
            fr, to = self._date_from.get(), self._date_to.get()
            starts = [d for d in ALL_STARTS if fr <= d.strftime("%Y-%m") <= to]
            return models, temps, cfgs, starts

        @staticmethod
        def _sig(c):
            return (c["model"], c["temp"], c["X"], c["N"], c["start"].strftime("%Y-%m"))

        def _rebuild_cells(self):
            models, temps, cfgs, starts = self._selected()
            self._cells = build_cells(models, temps, cfgs, starts)
            for i in self._tree.get_children():
                self._tree.delete(i)
            self._iid = {}
            for c in self._cells:
                saved = self._results.get(self._sig(c))
                if saved:  # preserve prior results across filter changes / restart
                    c["runs"] = saved["runs"]; c["attempts"] = saved["attempts"]
                    c["runs_target"] = saved["runs_target"]
                    c["status"] = saved["status"]; c["note"] = saved["note"]
                self._iid[c["id"]] = self._tree.insert("", "end",
                        values=self._row_vals(c), tags=(c["status"],))
            self._refresh_stats()

        def _row_vals(self, c):
            def roi(a):
                vals = [(r.get("arms", {}).get(a, {}).get("return_analysis") or {}).get("total_roi")
                        for r in c.get("runs", [])]
                vals = [v for v in vals if isinstance(v, (int, float))]
                return f"{sum(vals)/len(vals):.1f}" if vals else ""
            return (c["id"], c["model"], f"{c['X']}→{c['N']}", f"{c['start']:%Y-%m}",
                    roi("A1_single_pool"), roi("A2_sequential_cut"),
                    roi("A3_sequential_nocut"), roi("A4_reports_scrambled"),
                    roi("A5_single_matched"), roi("A6_hold_all"),
                    c["status"], c["note"])

        def _refresh_stats(self):
            cells = [c for c in self._cells if not c["skip"]]
            tot = len(self._cells)
            done_cells = sum(1 for c in cells if c["status"] == "Done")
            runs_done = sum(len(c.get("runs", [])) for c in cells)
            runs_target = sum(max(c.get("runs_target", 1), len(c.get("runs", [])))
                              for c in cells)
            skip = sum(1 for c in self._cells if c["skip"])
            self._stats.set(f"Total {tot}  Runnable {len(cells)}  "
                            f"Cells done {done_cells}  Runs {runs_done}/{runs_target}  "
                            f"Skipped {skip}")
            self._pb.set(runs_done / runs_target * 100 if runs_target else 0)

        def _set_row(self, c):
            self._tree.item(self._iid[c["id"]], values=self._row_vals(c),
                            tags=(c["status"],))
            self._tree.see(self._iid[c["id"]])

        # -- run control --
        def _on_start(self):
            key = self._key_var.get().strip()
            if not key:
                messagebox.showwarning("API key", "Enter your OpenAI API key first.")
                return
            _save_key(key)  # persist so re-downloading the script never wipes it
            global HOLD_ALL, A4_MODE
            HOLD_ALL = bool(self._holdall.get()); A4_MODE = self._a4mode.get()
            self._rebuild_cells()  # restores prior results; does NOT wipe finished cells
            target = max(1, int(self._runs_var.get()))
            n_threads = max(1, int(self._threads_var.get()))
            # Build the task list = only the runs still MISSING per cell (resume).
            tasks = []
            for c in self._cells:
                if c["skip"]:
                    continue
                c["runs_target"] = max(target, len(c.get("runs", [])))
                for r in range(len(c.get("runs", [])), target):
                    tasks.append((c, r))
            if not tasks:
                messagebox.showinfo("Nothing to run",
                                    "All selected cells already have the requested "
                                    "number of runs. Increase Runs/cell to add more.")
                return
            arms_per = 6
            n_calls = len(tasks) * (arms_per + 1)  # +1 stage-one per run
            if not messagebox.askyesno(
                    "Confirm run",
                    f"{len(tasks)} run(s) still needed × {arms_per} arms "
                    f"(+1 stage-one each) = {n_calls} live OpenAI calls, "
                    f"{n_threads} thread(s).\n\nFinished cells are kept; only missing "
                    f"runs will execute.\n\nProceed?"):
                return
            self._stop.clear()
            self._btn_start.config(state="disabled"); self._btn_stop.config(state="normal")
            self._logln(f"Starting {len(tasks)} run(s) on {n_threads} thread(s) "
                        f"({n_calls} calls)…")
            self._worker = Worker(tasks, key, self._q, self._stop, n_threads)
            self._worker.start()

        def _on_stop(self):
            self._stop.set(); self._logln("Stop requested — finishing current cell…")
            self._btn_stop.config(state="disabled")

        def _cell_by_id(self, cid):
            return next((c for c in self._cells if c["id"] == cid), None)

        def _persist(self, c):
            self._results[self._sig(c)] = {
                "status": c["status"], "runs": c.get("runs", []),
                "attempts": c.get("attempts", 0),
                "runs_target": c.get("runs_target", 1), "note": c.get("note", "")}

        def _drain(self):
            try:
                while True:
                    msg = self._q.get_nowait()
                    k = msg.get("kind")
                    if k in ("status", "run") and self._cell_by_id(msg["id"]) is None:
                        continue
                    if k == "status":
                        c = self._cell_by_id(msg["id"])
                        if c["status"] != "Done":  # never downgrade a finished cell
                            c["status"] = msg["status"]
                            if msg.get("note"):
                                c["note"] = msg["note"]
                            self._persist(c); self._set_row(c)
                    elif k == "run":
                        c = self._cell_by_id(msg["id"])
                        res = msg["res"]; c["attempts"] = c.get("attempts", 0) + 1
                        if isinstance(res, dict) and res.get("arms"):
                            c.setdefault("runs", []).append(res)
                            write_csv_rows(c, res, msg["run_idx"])  # main thread = safe
                        else:
                            err = (res or {}).get("error", "unknown")
                            self._logln(f"[{c['id']}] run {msg['run_idx']+1} failed: {err}")
                        got = len(c.get("runs", []))
                        tgt = c.get("runs_target", 1)
                        if got >= tgt:
                            c["status"] = "Done"
                        elif c["attempts"] >= tgt and got == 0:
                            c["status"] = "Error"
                        else:
                            c["status"] = "Running"
                        c["note"] = f"{got}/{tgt} run(s)"
                        self._persist(c); self._set_row(c)
                        if got:
                            self._logln(f"[{c['id']}] {c['model']} {c['X']}→{c['N']} "
                                        f"{c['start']:%Y-%m}  run {msg['run_idx']+1} "
                                        f"({got}/{tgt})")
                    elif k == "done":
                        self._btn_start.config(state="normal")
                        self._btn_stop.config(state="disabled")
                        self._logln(f"Finished. Summary → {SUMMARY_CSV}")
                    self._refresh_stats()
            except _queue.Empty:
                pass
            self.after(250, self._drain)

    App().mainloop()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli()
    else:
        run_gui()
