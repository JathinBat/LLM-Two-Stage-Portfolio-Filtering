#!/usr/bin/env python3
"""
reinvestment_runner.py — Periodic-reinvestment / dynamic re-selection runner
============================================================================

Extends the two-stage LLM filtering pipeline from a single buy-and-hold decision
to a *sequence* of rebalance decisions across a holding window. At each rebalance
point the model is re-run with (a) news available up to that date and (b) the
realized performance of the stocks it previously held, and the full current
portfolio value is redeployed into a fresh selection (full reselection).

This is the "dynamic re-selection" tier from Paper_Split_Handoff.md (Paper B,
Candidate 1 — selection vs. management).

Two feedback injection modes (run them separately to compare):
  • "reflection_layer"  — prior-holdings performance is added as a reflection layer
                          BEFORE Stage One (lands in additional_context, the
                          company-identification prompt). The model reviews how its
                          picks did, then re-selects.
  • "financial_report"  — prior-holdings performance is attached to the Stage-Two
                          financial/final-selection step (lands in final_step_context,
                          alongside fundamentals).

Rebalance schedule is fully configurable (no default): specify exactly one of
--segments N, --interval-months M, or --dates d1,d2,...

Run modes:
  • --dry-run   uses a deterministic MockBackend (NO OpenAI calls) to validate the
                orchestration: segment dates, reinvestment compounding, and the
                feedback payloads for both modes.
  • (live)      uses RealBackend, which calls InvestmentStrategyGenerator and WILL
                make OpenAI calls. Requires OPENAI_API_KEY in the environment
                (the runner never reads keys hard-coded in source).

Usage examples
--------------
    # validate everything, no API calls
    python reinvestment_runner.py --dry-run --sector technology \\
        --start 2024-07-01 --end 2025-07-01 --segments 4 --mode both

    # live quarterly run, reflection-layer feedback (needs OPENAI_API_KEY)
    SECTOR=financials OPENAI_API_KEY=sk-... python reinvestment_runner.py \\
        --sector financials --start 2024-07-01 --end 2025-07-01 \\
        --interval-months 3 --mode reflection_layer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

from dateutil.relativedelta import relativedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "SCCUR_TESTS"))

import sector_config  # noqa: E402  (sector registry; honours SECTOR env var)

FEEDBACK_MODES = ("reflection_layer", "financial_report")
NEWS_LOOKBACK_YEARS = 2          # how far back of news to feed at each rebalance


# --------------------------------------------------------------------------- #
#  Rebalance schedule                                                         #
# --------------------------------------------------------------------------- #
def _d(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _s(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def build_segments(start: str, end: str, *,
                   n_segments: Optional[int] = None,
                   interval_months: Optional[int] = None,
                   dates: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Split [start, end] into consecutive holding segments. Exactly one of
    n_segments / interval_months / dates must be supplied (no default cadence).

    Returns a list of {"index","start","end","rebalance_date"} dicts. The
    rebalance_date is each segment's start — the as-of date the model decides on.
    """
    chosen = [x is not None for x in (n_segments, interval_months, dates)]
    if sum(chosen) != 1:
        raise ValueError("Specify exactly one of n_segments, interval_months, or dates.")

    s_dt, e_dt = _d(start), _d(end)
    if e_dt <= s_dt:
        raise ValueError("end must be after start.")

    bounds: List[datetime] = [s_dt]
    if dates is not None:
        for ds in dates:
            d = _d(ds)
            if not (s_dt < d < e_dt):
                raise ValueError(f"rebalance date {ds} must fall strictly inside the window.")
            bounds.append(d)
        bounds.append(e_dt)
    elif n_segments is not None:
        if n_segments < 1:
            raise ValueError("n_segments must be >= 1.")
        total_days = (e_dt - s_dt).days
        step = total_days / n_segments
        bounds = [s_dt + relativedelta(days=round(step * i)) for i in range(n_segments)]
        bounds.append(e_dt)
    else:  # interval_months
        if interval_months < 1:
            raise ValueError("interval_months must be >= 1.")
        cur = s_dt
        while True:
            nxt = cur + relativedelta(months=interval_months)
            if nxt >= e_dt:
                break
            bounds.append(nxt)
            cur = nxt
        bounds.append(e_dt)

    bounds = sorted(set(bounds))
    segs = []
    for i in range(len(bounds) - 1):
        segs.append({
            "index": i,
            "start": _s(bounds[i]),
            "end": _s(bounds[i + 1]),
            "rebalance_date": _s(bounds[i]),
        })
    return segs


# --------------------------------------------------------------------------- #
#  Performance feedback                                                        #
# --------------------------------------------------------------------------- #
def build_feedback_text(prior_holdings: List[Dict[str, Any]],
                        individual_returns: Dict[str, float],
                        prior_segment: Dict[str, str]) -> str:
    """
    Format how the previous segment's holdings performed, for injection into the
    next decision. Only realized (past, as-of) returns — no look-ahead.
    """
    if not prior_holdings:
        return ""
    lines = [
        f"Your portfolio from {prior_segment['start']} to {prior_segment['end']} "
        f"produced these realized results (use them to inform — not anchor — the "
        f"next selection; past performance is not guaranteed to persist):"
    ]
    for h in sorted(prior_holdings, key=lambda x: individual_returns.get(x["ticker"], 0.0),
                    reverse=True):
        t = h["ticker"]
        r = individual_returns.get(t)
        rtxt = f"{r:+.1f}%" if isinstance(r, (int, float)) else "n/a"
        lines.append(f"  - {t}: held at {h['allocation']:.1f}% weight, realized {rtxt}")
    winners = [t for t in individual_returns if individual_returns[t] > 0]
    losers = [t for t in individual_returns if individual_returns[t] <= 0]
    lines.append(f"Summary: {len(winners)} winners, {len(losers)} laggards. "
                 f"Reassess whether each thesis still holds before reselecting.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Backends                                                                    #
# --------------------------------------------------------------------------- #
class StrategyBackend:
    """Interface: run one segment, return a normalized result dict."""

    def run_segment(self, *, sector: str, capital: float, news_start: str,
                    news_end: str, analysis_start: str, analysis_end: str,
                    feedback_text: str, feedback_mode: str) -> Dict[str, Any]:
        raise NotImplementedError


def _normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Map a generate_complete_strategy result to the runner's normalized shape."""
    recs = result.get("final_recommendations", []) or []
    holdings = [{"ticker": r.get("ticker", ""),
                 "allocation": float(r.get("final_allocation", 0) or 0)}
                for r in recs if r.get("ticker")]
    ra = result.get("return_analysis", {}) or {}
    return {
        "holdings": holdings,
        "segment_return_pct": ra.get("main_return", 0.0),
        "individual_returns": ra.get("individual_returns", {}) or {},
        "benchmark": ra.get("benchmark_comparison", {}) or {},
    }


class RealBackend(StrategyBackend):
    """Wraps InvestmentStrategyGenerator. Makes real OpenAI calls."""

    def __init__(self, model: str = "gpt-5.1"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "RealBackend needs OPENAI_API_KEY in the environment. "
                "Refusing to read keys hard-coded in source. "
                "Use --dry-run to validate without any API calls.")
        from investment_strategy_generator import InvestmentStrategyGenerator
        self.model = model
        self.gen = InvestmentStrategyGenerator(
            api_key_openai=api_key,
            nyt_api_key=os.environ.get("NYT_API_KEY"),
            model=model,
        )

    def run_segment(self, *, sector, capital, news_start, news_end,
                    analysis_start, analysis_end, feedback_text, feedback_mode):
        kwargs = dict(
            user_input_keyword=sector,
            investment_amount=capital,
            news_start_date=news_start,
            news_end_date=news_end,
            analysis_start_date=analysis_start,
            analysis_end_date=analysis_end,
            news_data_path=sector_config.news_file(ROOT),
        )
        if feedback_text:
            if feedback_mode == "reflection_layer":
                kwargs["additional_context"] = (
                    "PORTFOLIO REVIEW (reflection layer):\n" + feedback_text)
            elif feedback_mode == "financial_report":
                kwargs["final_step_context"] = feedback_text
            else:
                raise ValueError(f"unknown feedback_mode {feedback_mode}")
        result = self.gen.generate_complete_strategy(**kwargs)
        if result.get("error"):
            raise RuntimeError(f"generator error: {result['error']}")
        return _normalize_result(result)


class MockBackend(StrategyBackend):
    """
    Deterministic backend for dry-run validation. No API calls. Produces plausible
    holdings from the active sector universe and pseudo-random-but-seeded returns,
    and RECORDS the feedback text + mode it received so the wiring can be asserted.
    """

    def __init__(self, final_portfolio: int = 5, seed: int = 7):
        self.final_portfolio = final_portfolio
        self.seed = seed
        self.received: List[Dict[str, Any]] = []   # audit log of injected feedback

    def run_segment(self, *, sector, capital, news_start, news_end,
                    analysis_start, analysis_end, feedback_text, feedback_mode):
        import hashlib
        self.received.append({
            "rebalance_date": news_end,
            "feedback_mode": feedback_mode,
            "feedback_injected": bool(feedback_text),
            "feedback_preview": feedback_text[:160],
        })
        universe = sector_config.universe()
        # Deterministic pick: rotate through the universe by segment date.
        h = int(hashlib.md5(f"{self.seed}{news_end}".encode()).hexdigest(), 16)
        n = self.final_portfolio
        picks = [universe[(h + i * 7) % len(universe)] for i in range(n)]
        picks = list(dict.fromkeys(picks))               # dedupe, keep order
        while len(picks) < n:                            # backfill if collisions
            cand = universe[(h + len(picks) * 13) % len(universe)]
            if cand not in picks:
                picks.append(cand)
        alloc = round(100.0 / len(picks), 2)
        holdings = [{"ticker": t, "allocation": alloc} for t in picks]
        # Deterministic per-stock returns in [-12%, +20%]
        individual = {}
        for t in picks:
            hv = int(hashlib.md5(f"{news_end}{t}".encode()).hexdigest(), 16)
            individual[t] = round(-12 + (hv % 3200) / 100.0, 2)   # -12 .. +20
        seg_return = round(sum(individual[t] * alloc / 100.0 for t in picks), 2)
        return {
            "holdings": holdings,
            "segment_return_pct": seg_return,
            "individual_returns": individual,
            "benchmark": {"benchmark_return": round((h % 1500) / 100.0 - 2, 2)},
        }


# --------------------------------------------------------------------------- #
#  Orchestration                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class SegmentRecord:
    index: int
    rebalance_date: str
    start: str
    end: str
    capital_in: float
    holdings: List[Dict[str, Any]]
    segment_return_pct: float
    capital_out: float
    individual_returns: Dict[str, float]
    feedback_mode: str
    feedback_injected: bool
    benchmark: Dict[str, Any] = field(default_factory=dict)


def run_reinvestment(backend: StrategyBackend, *, sector: str, start_capital: float,
                     segments: List[Dict[str, str]], feedback_mode: str) -> Dict[str, Any]:
    """Full-reselection reinvestment loop across the segment schedule."""
    if feedback_mode not in FEEDBACK_MODES:
        raise ValueError(f"feedback_mode must be one of {FEEDBACK_MODES}")

    records: List[SegmentRecord] = []
    capital = float(start_capital)
    prior_holdings: List[Dict[str, Any]] = []
    prior_returns: Dict[str, float] = {}
    prior_segment: Optional[Dict[str, str]] = None

    for seg in segments:
        feedback_text = ""
        if prior_segment is not None:
            feedback_text = build_feedback_text(prior_holdings, prior_returns, prior_segment)

        news_end = seg["rebalance_date"]
        news_start = _s(_d(news_end) - relativedelta(years=NEWS_LOOKBACK_YEARS))

        res = backend.run_segment(
            sector=sector, capital=capital,
            news_start=news_start, news_end=news_end,
            analysis_start=seg["start"], analysis_end=seg["end"],
            feedback_text=feedback_text, feedback_mode=feedback_mode,
        )

        seg_return = float(res["segment_return_pct"] or 0.0)
        capital_out = capital * (1 + seg_return / 100.0)   # full reselection + reinvest
        records.append(SegmentRecord(
            index=seg["index"], rebalance_date=news_end,
            start=seg["start"], end=seg["end"],
            capital_in=round(capital, 2), holdings=res["holdings"],
            segment_return_pct=round(seg_return, 2),
            capital_out=round(capital_out, 2),
            individual_returns=res["individual_returns"],
            feedback_mode=feedback_mode,
            feedback_injected=bool(feedback_text),
            benchmark=res.get("benchmark", {}),
        ))

        capital = capital_out
        prior_holdings = res["holdings"]
        prior_returns = res["individual_returns"]
        prior_segment = seg

    total_roi = (capital - start_capital) / start_capital * 100.0
    return {
        "sector": sector,
        "feedback_mode": feedback_mode,
        "start_capital": start_capital,
        "final_value": round(capital, 2),
        "total_roi_pct": round(total_roi, 2),
        "num_rebalances": len(segments) - 1,
        "segments": [asdict(r) for r in records],
    }


# --------------------------------------------------------------------------- #
#  Reporting                                                                   #
# --------------------------------------------------------------------------- #
def print_report(run: Dict[str, Any]) -> None:
    print(f"\n{'='*72}")
    print(f" REINVESTMENT RUN — sector={run['sector']}  mode={run['feedback_mode']}")
    print(f"{'='*72}")
    print(f" Start capital : ${run['start_capital']:,.2f}")
    print(f" Rebalances    : {run['num_rebalances']} "
          f"({len(run['segments'])} holding segments)")
    print(f"{'-'*72}")
    print(f" {'seg':>3} {'rebalance':>11} {'window':>23} {'ret%':>7} "
          f"{'value out':>13} {'fb':>3}")
    for s in run["segments"]:
        win = f"{s['start']}->{s['end']}"
        fb = "Y" if s["feedback_injected"] else "-"
        print(f" {s['index']:>3} {s['rebalance_date']:>11} {win:>23} "
              f"{s['segment_return_pct']:>7.2f} ${s['capital_out']:>11,.2f} {fb:>3}")
    print(f"{'-'*72}")
    print(f" FINAL VALUE   : ${run['final_value']:,.2f}")
    print(f" TOTAL ROI     : {run['total_roi_pct']:+.2f}%")
    print(f"{'='*72}")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Periodic-reinvestment / dynamic re-selection runner")
    p.add_argument("--sector", default=sector_config.active_sector(),
                   help="sector keyword (default: SECTOR env / technology)")
    p.add_argument("--capital", type=float, default=100000.0)
    p.add_argument("--start", required=True, help="window start YYYY-MM-DD")
    p.add_argument("--end", required=True, help="window end YYYY-MM-DD")
    sched = p.add_mutually_exclusive_group(required=True)
    sched.add_argument("--segments", type=int, help="number of equal holding segments")
    sched.add_argument("--interval-months", type=int, help="months between rebalances")
    sched.add_argument("--dates", help="comma-separated interior rebalance dates")
    p.add_argument("--mode", choices=list(FEEDBACK_MODES) + ["both"], default="both")
    p.add_argument("--model", default="gpt-5.1")
    p.add_argument("--dry-run", action="store_true",
                   help="use the mock backend; NO OpenAI calls")
    p.add_argument("--final-portfolio", type=int, default=5,
                   help="holdings per segment (mock backend only)")
    p.add_argument("--out", default=None, help="write results JSON to this path")
    args = p.parse_args(argv)

    # Align sector_config with the requested sector (drives universe/benchmark/news).
    os.environ["SECTOR"] = args.sector

    dates = [d.strip() for d in args.dates.split(",")] if args.dates else None
    segments = build_segments(
        args.start, args.end,
        n_segments=args.segments, interval_months=args.interval_months, dates=dates,
    )
    print(f" Schedule: {len(segments)} segments, "
          f"{len(segments)-1} rebalances, {args.start} -> {args.end}")

    backend: StrategyBackend
    if args.dry_run:
        backend = MockBackend(final_portfolio=args.final_portfolio)
        print(" Backend : MockBackend (dry-run, no API calls)")
    else:
        backend = RealBackend(model=args.model)
        print(f" Backend : RealBackend (live, model={args.model})")

    modes = list(FEEDBACK_MODES) if args.mode == "both" else [args.mode]
    runs = {}
    for m in modes:
        run = run_reinvestment(backend, sector=args.sector,
                               start_capital=args.capital,
                               segments=segments, feedback_mode=m)
        runs[m] = run
        print_report(run)

    if len(runs) > 1:
        print(f"\n{'#'*72}\n MODE COMPARISON (total ROI)\n{'#'*72}")
        for m, r in runs.items():
            print(f"  {m:18s}: {r['total_roi_pct']:+.2f}%   final ${r['final_value']:,.2f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"schedule": segments, "runs": runs}, f, indent=2)
        print(f"\n Wrote results -> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
