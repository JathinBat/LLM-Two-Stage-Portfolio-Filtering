"""LLM relevance screen over the tech news corpus — Paper B only.

Why this exists
---------------
The manuscript's "News Article Relevance Screening Process" describes a
constrained LLM screen whose output is "stored in a curated database and reused
downstream." That screen was run once (1,382 articles -> 110 retained, 8.0%,
archived under news_collection_archive/gpt_filtered/) and validated against a
180-row blind sample for NHSJS Rec #5 — but its output was never wired into the
strategy pipeline. `merged_news_data.csv` contains all 1,272 screen-rejected
articles alongside the 110 that passed.

What the pipeline does instead is cruder: `_process_news_data` keeps only
articles whose headline or snippet literally contains the sector keyword
("technology"), which retains ~3% of the corpus and discards obviously relevant
market news ("Tech Stocks Sink as Investors Fret About Growth" is dropped;
"Ford Lost Money in the Third Quarter" is kept).

This script runs the real screen so Paper B can consume a properly curated
corpus. **Paper A is deliberately left alone** — it keeps reading
merged_news_data.csv through the existing path, so its published numbers stand.

The screening prompt is copied verbatim from
`BASE/comprehensive_news_collector.py` so the Rec #5 validation numbers
(precision 100%, 8.0% population pass rate) describe this screen too.

Outputs (neither filename contains "news"/"article"/"finance"/"consolidated",
so the Paper A workspace walk in `_load_news_data` can never pick them up):
  * tech_screened_corpus.csv  — passed articles only, in the corpus schema
  * screen_decisions_tech.csv — every article + pass/reject/error, the audit
                                trail for the paper's screening ledger

Usage
-----
    python screen_news_relevance.py --dry-run     # batches + cost estimate
    python screen_news_relevance.py --limit 120   # small live test
    python screen_news_relevance.py               # screen everything
    python screen_news_relevance.py --report      # summarize existing outputs

Resumable: decisions cache to `screen_cache_tech.json` keyed by article URL, so
re-running after a news backfill screens only the new articles.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import pandas as pd

import env_loader  # noqa: F401  (populates os.environ from .env)

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "merged_news_data.csv")
SCREENED = os.path.join(ROOT, "tech_screened_corpus.csv")
DECISIONS = os.path.join(ROOT, "screen_decisions_tech.csv")
CACHE = os.path.join(ROOT, "screen_cache_tech.json")

COLUMNS = ["headline", "pub_date", "snippet", "web_url", "source", "section",
           "news_desk", "industry", "search_keyword", "fetch_date"]

BATCH_SIZE = 30          # matches the original screening run
MODEL = "gpt-4o-mini"    # as validated for Rec #5

# Verbatim from BASE/comprehensive_news_collector.py so the validated precision
# figure describes this screen. Do not reword without re-running the validation.
SYSTEM_PROMPT = """You are an EXTREMELY STRICT financial news curator for SERIOUS STOCK INVESTMENT ANALYSIS. Only include articles that directly impact investment decisions.

STRICT CRITERIA - INCLUDE ONLY IF:
1. DIRECT STOCK/MARKET IMPACT: Article directly mentions stock prices, market movements, earnings, or financial performance
2. MAJOR COMPANY NEWS: Significant corporate developments (mergers, acquisitions, leadership changes, product launches) for publicly traded companies
3. ECONOMIC INDICATORS: Federal Reserve decisions, inflation data, GDP reports, unemployment that directly affect markets
4. SECTOR ANALYSIS: Tech sector growth, financial sector regulations, industry trends affecting multiple stocks
5. STARTUP/IPO NEWS: Only if discussing funding rounds, IPOs, or acquisition potential
6. FINANCIAL TECHNOLOGY: Only fintech innovations that could disrupt existing financial institutions

ABSOLUTELY EXCLUDE:
- General tech news without financial/business angle
- Politics unless directly market-moving (like trade policy)
- Opinion pieces or analysis without concrete data
- Entertainment, sports, lifestyle, health, crime, weather
- General business news without stock market relevance
- Articles about individuals unless they're major company executives
- Regulatory news unless it directly affects specific industries/stocks
- General economic news without clear market implications
- Any article that doesn't help make specific investment decisions

BE EXTREMELY SELECTIVE. Only 20-40% of articles should pass this filter.

Return ONLY the numbers (indices) of relevant articles, separated by commas.
Example: 1,3,5,7,12"""


# --------------------------------------------------------------------------- #
#  Corpus assembly                                                             #
# --------------------------------------------------------------------------- #
def load_inputs(extra: List[str]) -> pd.DataFrame:
    """Corpus + any backfill files, deduped. Backfills are picked up by glob."""
    import glob

    paths = [CORPUS] + sorted(glob.glob(os.path.join(ROOT, "tech_news_backfill_*.csv")))
    paths += [p for p in extra if p not in paths]

    frames = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  (skipping missing {os.path.basename(p)})")
            continue
        df = pd.read_csv(p)
        missing = [c for c in COLUMNS if c not in df.columns]
        for c in missing:
            df[c] = ""
        frames.append(df[COLUMNS])
        print(f"  {os.path.basename(p):48s} {len(df):6d} rows")

    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    before = len(out)
    out = out.drop_duplicates(subset=["web_url"], keep="first")
    out = out.drop_duplicates(subset=["headline"], keep="first")
    print(f"  combined: {before} -> {len(out)} unique articles")
    return out.reset_index(drop=True)


def article_key(row) -> str:
    url = str(row.get("web_url", "") or "").strip()
    return url if url and url.lower() != "nan" else f"hl::{str(row.get('headline', ''))[:200]}"


# --------------------------------------------------------------------------- #
#  Screening                                                                   #
# --------------------------------------------------------------------------- #
def load_cache() -> Dict[str, str]:
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, str]) -> None:
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def screen_batch(client, batch: pd.DataFrame, on_error: str) -> Dict[int, str]:
    """{row position -> 'pass'|'reject'|'error'} for one batch."""
    lines = []
    for pos, (_, row) in enumerate(batch.iterrows(), start=1):
        hl = str(row.get("headline", ""))[:100]
        sn = str(row.get("snippet", ""))[:150]
        lines.append(f"{pos}. {hl} - {sn}")
    user = "Filter these articles for STRICT investment relevance:\n\n" + "\n".join(lines)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
                temperature=0.1,
                max_tokens=200,
            )
            text = (resp.choices[0].message.content or "").strip()
            picked = {int(x) for x in text.replace(",", " ").split() if x.isdigit()}
            return {pos: ("pass" if pos in picked else "reject")
                    for pos in range(1, len(batch) + 1)}
        except Exception as e:                                   # noqa: BLE001
            wait = 2 ** attempt
            print(f"      attempt {attempt + 1}/3 failed ({e}); retrying in {wait}s")
            time.sleep(wait)

    # The original collector failed OPEN here (kept every article on error), which
    # is how unscreened material reached the corpus. Default to dropping instead,
    # and always record the outcome so the count is visible rather than silent.
    verdict = "pass" if on_error == "keep" else "error"
    print(f"      batch failed after 3 attempts -> recorded as '{verdict}'")
    return {pos: verdict for pos in range(1, len(batch) + 1)}


def report() -> int:
    if not os.path.exists(DECISIONS):
        print(f"No decisions file yet ({os.path.basename(DECISIONS)}). Run the screen first.")
        return 1
    dec = pd.read_csv(DECISIONS)
    counts = dec["decision"].value_counts().to_dict()
    total = len(dec)
    passed = counts.get("pass", 0)
    print(f"decisions : {total}")
    for k, v in sorted(counts.items()):
        print(f"  {k:7s}: {v:6d} ({v / total * 100:5.1f}%)")
    print(f"\npass rate : {passed / total * 100:.1f}%  "
          f"(manuscript's representative run: 8.0%)")
    if os.path.exists(SCREENED):
        s = pd.read_csv(SCREENED)
        d = pd.to_datetime(s["pub_date"], errors="coerce")
        print(f"screened corpus: {len(s)} rows, {d.min():%Y-%m-%d} -> {d.max():%Y-%m-%d}")
        print("\npassed articles per quarter:")
        for period, n in d.groupby(d.dt.to_period("Q")).size().items():
            print(f"  {period}: {n:5d}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="screen only the first N articles (testing)")
    ap.add_argument("--dry-run", action="store_true", help="batch/cost estimate; no API calls")
    ap.add_argument("--report", action="store_true", help="summarize existing outputs and exit")
    ap.add_argument("--on-error", choices=["drop", "keep"], default="drop",
                    help="what to do with a batch that fails 3 attempts (default: drop)")
    ap.add_argument("--inputs", default="", help="comma-separated extra input CSVs")
    args = ap.parse_args()

    if args.report:
        return report()

    print("Loading inputs...")
    extra = [p.strip() for p in args.inputs.split(",") if p.strip()]
    df = load_inputs(extra)
    if df.empty:
        print("No articles to screen.")
        return 1
    if args.limit:
        df = df.head(args.limit)
        print(f"  --limit: screening only the first {len(df)} articles")

    cache = load_cache()
    # Only pass/reject count as settled. Articles cached as "error" (a batch that
    # exhausted its retries) must be re-attempted on the next run, or a transient
    # outage would silently freeze them out of the corpus forever.
    settled = {k for k, v in cache.items() if v in ("pass", "reject")}
    n_retry = sum(1 for v in cache.values() if v == "error")
    todo = df[~df.apply(article_key, axis=1).isin(settled)]
    if n_retry:
        print(f"  ({n_retry} previously errored articles will be retried)")
    n_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n{len(df)} articles, {len(df) - len(todo)} already cached, "
          f"{len(todo)} to screen -> {n_batches} batches of {BATCH_SIZE}")
    # ~1.5k input + ~0.1k output tokens per batch on gpt-4o-mini.
    print(f"  rough cost: ~{n_batches * 1.6 / 1000:.2f}M tokens on {MODEL} (cents, not dollars)")

    if args.dry_run:
        print("\n--dry-run: no API calls made.")
        return 0

    if len(todo):
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("\nOPENAI_API_KEY is not set. Put it in .env and re-run.")
            return 1
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo.iloc[i:i + BATCH_SIZE]
            bnum = i // BATCH_SIZE + 1
            verdicts = screen_batch(client, batch, args.on_error)
            n_pass = sum(1 for v in verdicts.values() if v == "pass")
            for pos, (_, row) in enumerate(batch.iterrows(), start=1):
                cache[article_key(row)] = verdicts[pos]
            save_cache(cache)          # checkpoint every batch — safe to interrupt
            print(f"  batch {bnum}/{n_batches}: {n_pass}/{len(batch)} passed")
            time.sleep(1)              # gentle on rate limits

    # ---- write outputs ---------------------------------------------------- #
    keys = df.apply(article_key, axis=1)
    df = df.assign(decision=[cache.get(k, "unscreened") for k in keys])
    df.to_csv(DECISIONS, index=False)

    passed = df[df["decision"] == "pass"][COLUMNS].copy()
    passed = passed.sort_values("pub_date")
    passed.to_csv(SCREENED, index=False)

    counts = df["decision"].value_counts().to_dict()
    print(f"\nWrote {os.path.basename(DECISIONS)} ({len(df)} rows) and "
          f"{os.path.basename(SCREENED)} ({len(passed)} rows)")
    for k, v in sorted(counts.items()):
        print(f"  {k:11s}: {v:6d} ({v / len(df) * 100:5.1f}%)")
    if counts.get("error"):
        print(f"  NOTE: {counts['error']} articles hit repeated API errors and were "
              f"excluded from the screened corpus. Re-running this command retries "
              f"exactly those (settled pass/reject decisions are not re-screened).")
    print(f"\nPass rate {len(passed) / len(df) * 100:.1f}% vs the manuscript's "
          f"representative 8.0%.")
    print("Paper A is untouched — it still reads merged_news_data.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
