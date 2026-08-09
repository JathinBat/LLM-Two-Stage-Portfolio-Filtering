"""Backfill the technology news corpus from the NY Times Article Search API.

Why this exists
---------------
`merged_news_data.csv` (the tech sector's news file) ends **2025-11-12**. The
Paper B windows run 12-month holds from 2024-10 .. 2025-06 starts, so rebalance
decisions extend to 2026-06 — past the end of the corpus. Across the 90 existing
runs, 66 of 432 decision dates (15.3%) were made with no news newer than
2025-11, and staleness rises with cadence (5.6% at semiannual -> 19.4% at
monthly). That confounds the "rebalance frequency does not help" finding with
"the news ran out," so the corpus needs extending before the feedback ON/OFF
experiment runs at monthly cadence.

What it does
------------
Fetches NYT articles month by month (so density is even rather than
front-loaded by the API's newest-first sort), dedupes by URL, and writes a CSV
in exactly the `merged_news_data.csv` schema.

Output modes (never destructive by default):
  * default  -> writes a NEW file `tech_news_backfill_<start>_<end>.csv`.
                The tech loader walks the workspace for any CSV whose name
                contains "news", so this file is picked up automatically with
                no edit to the existing corpus.
  * --merge  -> additionally appends into merged_news_data.csv, after copying
                the original to merged_news_data.backup_<timestamp>.csv.

Usage
-----
    python backfill_tech_news.py --report            # coverage only, no API calls
    python backfill_tech_news.py --dry-run           # show the fetch plan + cost
    python backfill_tech_news.py                     # fetch 2025-11-01 -> today
    python backfill_tech_news.py --merge             # fetch, then fold into the corpus
    python backfill_tech_news.py --start 2025-11-01 --end 2026-07-01

Resumable: every month is checkpointed under `news_backfill_checkpoints/`. Re-run
the same command after an interruption and completed months are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

import env_loader  # noqa: F401  (populates os.environ from .env)

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "merged_news_data.csv")
CKPT_DIR = os.path.join(ROOT, "news_backfill_checkpoints")
NYT_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

# The corpus schema. Column order is preserved on write so the backfill file is
# a drop-in sibling of merged_news_data.csv.
COLUMNS = ["headline", "pub_date", "snippet", "web_url", "source", "section",
           "news_desk", "industry", "search_keyword", "fetch_date"]

# Curated tech/finance keywords. Deliberately narrower than
# comprehensive_news_collector.py's ~60 — that list was built for a one-off bulk
# pull, and at month granularity 60 keywords x 9 months would run for many hours
# against a 5-req/min rate limit for heavily redundant articles.
KEYWORDS = [
    "technology", "tech stocks", "artificial intelligence", "semiconductors",
    "software", "cloud computing", "cybersecurity", "data center",
    "chipmaker", "electric vehicles", "earnings", "stock market",
    "Nasdaq", "investors", "Federal Reserve", "interest rates",
    "merger", "IPO", "quantum computing", "robotics",
]

# NYT's public tier is ~5 requests/minute; 12s between calls stays under it.
# Their docs also ask for a daily cap, which a backfill of this size respects.
DEFAULT_DELAY = 12.0
DEFAULT_MAX_PAGES = 4          # 4 pages x 10 = up to 40 articles/keyword/month


# --------------------------------------------------------------------------- #
#  Coverage reporting                                                          #
# --------------------------------------------------------------------------- #
def load_corpus_dates(path: str = CORPUS) -> Optional[pd.Series]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=["pub_date"])
    dates = pd.to_datetime(df["pub_date"], errors="coerce", utc=True)
    return dates.dt.tz_localize(None).dropna()


def report_coverage() -> None:
    dates = load_corpus_dates()
    if dates is None or dates.empty:
        print(f"No usable corpus at {CORPUS}")
        return
    print(f"corpus : {CORPUS}")
    print(f"articles with usable dates: {len(dates)}")
    print(f"range  : {dates.min():%Y-%m-%d} -> {dates.max():%Y-%m-%d}")
    print("\narticles per month (last 24 months of the corpus):")
    monthly = dates.groupby(dates.dt.to_period("M")).size().tail(24)
    for period, n in monthly.items():
        print(f"  {period}: {n:5d}  {'#' * min(50, n // 12)}")
    gap_days = (pd.Timestamp.today().normalize() - dates.max().normalize()).days
    print(f"\ngap between corpus end and today: {gap_days} days")


# --------------------------------------------------------------------------- #
#  Fetching                                                                    #
# --------------------------------------------------------------------------- #
def month_spans(start: str, end: str) -> List[tuple]:
    """[(YYYY-MM-DD begin, YYYY-MM-DD end), ...] one entry per calendar month."""
    cur = pd.Timestamp(start).normalize().replace(day=1)
    last = pd.Timestamp(end).normalize()
    spans = []
    while cur <= last:
        nxt = cur + relativedelta(months=1)
        month_end = pd.Timestamp(nxt) - pd.DateOffset(days=1)
        spans.append((cur.strftime("%Y-%m-%d"),
                      min(month_end, last).strftime("%Y-%m-%d")))
        cur = nxt
    return spans


def _fetch_page(api_key: str, keyword: str, begin: str, end: str, page: int,
                delay: float) -> Optional[List[Dict]]:
    """One NYT page. None means 'stop paging this keyword/month'."""
    params = {
        "q": keyword,
        "begin_date": begin.replace("-", ""),
        "end_date": end.replace("-", ""),
        "sort": "newest",
        "page": page,
        "api-key": api_key,
    }
    for attempt in range(4):
        try:
            resp = requests.get(NYT_URL, params=params, timeout=30)
        except Exception as e:                                   # noqa: BLE001
            wait = delay * (attempt + 1)
            print(f"      network error ({e}); retry in {wait:.0f}s")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp.json().get("response", {}).get("docs", [])
        if resp.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"      rate limited; sleeping {wait}s")
            time.sleep(wait)
            continue
        if resp.status_code == 401:
            print("      401 Unauthorized — check NYT_API_KEY in .env")
            return None
        print(f"      HTTP {resp.status_code}; giving up on this page")
        return None
    return None


def fetch_month(api_key: str, begin: str, end: str, keywords: List[str],
                max_pages: int, delay: float) -> List[Dict]:
    rows: List[Dict] = []
    fetched_at = datetime.now().isoformat()
    for kw in keywords:
        print(f"    '{kw}'", end="", flush=True)
        got = 0
        for page in range(max_pages):
            docs = _fetch_page(api_key, kw, begin, end, page, delay)
            time.sleep(delay)
            if not docs:
                break
            for d in docs:
                rows.append({
                    "headline": (d.get("headline") or {}).get("main", ""),
                    "pub_date": d.get("pub_date", ""),
                    "snippet": d.get("snippet", "") or d.get("abstract", ""),
                    "web_url": d.get("web_url", ""),
                    "source": "NY Times",
                    "section": d.get("section_name", ""),
                    "news_desk": d.get("news_desk", ""),
                    "industry": "technology",
                    "search_keyword": kw,
                    "fetch_date": fetched_at,
                })
            got += len(docs)
            if len(docs) < 10:      # short page = end of results
                break
        print(f" -> {got}")
    return rows


def normalize(rows: List[Dict]) -> pd.DataFrame:
    """Corpus-shaped frame: naive 'YYYY-MM-DD HH:MM:SS' pub_date, deduped by URL."""
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df["pub_date"], errors="coerce", utc=True)
    df = df[ts.notna()].copy()
    df["pub_date"] = ts[ts.notna()].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.drop_duplicates(subset=["web_url"], keep="first")
    df = df.drop_duplicates(subset=["headline"], keep="first")
    return df.sort_values("pub_date")[COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Merge                                                                       #
# --------------------------------------------------------------------------- #
def merge_into_corpus(new_df: pd.DataFrame) -> None:
    """Append to merged_news_data.csv, backing the original up first.

    Per repo rules nothing is deleted: the pre-merge corpus is copied to a
    timestamped backup that stays on disk.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(ROOT, f"merged_news_data.backup_{stamp}.csv")
    shutil.copy2(CORPUS, backup)
    print(f"  backed up existing corpus -> {os.path.basename(backup)}")

    old = pd.read_csv(CORPUS)
    before = len(old)
    combined = pd.concat([old, new_df], ignore_index=True, sort=False)
    if "web_url" in combined.columns:
        combined = combined.drop_duplicates(subset=["web_url"], keep="first")
    combined = combined.drop_duplicates(subset=["headline"], keep="first")
    combined.to_csv(CORPUS, index=False)
    print(f"  corpus {before} -> {len(combined)} rows (+{len(combined) - before} new)")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2025-11-01",
                    help="first month to fetch (default: 2025-11-01, where the corpus ends)")
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"),
                    help="last date to fetch (default: today)")
    ap.add_argument("--merge", action="store_true",
                    help="also append into merged_news_data.csv (backs it up first)")
    ap.add_argument("--report", action="store_true",
                    help="print corpus coverage and exit; makes no API calls")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the fetch plan and request count; makes no API calls")
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                    help=f"pages per keyword per month (default {DEFAULT_MAX_PAGES}, 10 articles/page)")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                    help=f"seconds between API calls (default {DEFAULT_DELAY}; NYT allows ~5/min)")
    ap.add_argument("--keywords", default="",
                    help="comma-separated override for the default keyword set")
    args = ap.parse_args()

    if args.report:
        report_coverage()
        return 0

    keywords = ([k.strip() for k in args.keywords.split(",") if k.strip()]
                if args.keywords else KEYWORDS)
    spans = month_spans(args.start, args.end)
    if not spans:
        print(f"No months between {args.start} and {args.end}.")
        return 1

    max_calls = len(spans) * len(keywords) * args.max_pages
    print(f"Backfill plan: {spans[0][0]} -> {spans[-1][1]}")
    print(f"  {len(spans)} months x {len(keywords)} keywords x <= {args.max_pages} pages")
    print(f"  <= {max_calls} API calls, <= {max_calls * args.delay / 3600:.1f}h at {args.delay:g}s/call")
    print(f"  (early-exit on short pages usually cuts this well below the ceiling)")

    if args.dry_run:
        print("\n--dry-run: no API calls made. Months to fetch:")
        for b, e in spans:
            print(f"  {b} .. {e}")
        return 0

    api_key = os.environ.get("NYT_API_KEY", "").strip()
    if not api_key:
        print("\nNYT_API_KEY is not set. Put it in .env (see .env.example) and re-run.")
        return 1

    os.makedirs(CKPT_DIR, exist_ok=True)
    all_rows: List[Dict] = []
    for i, (begin, end) in enumerate(spans, 1):
        ckpt = os.path.join(CKPT_DIR, f"tech_{begin[:7]}.json")
        if os.path.exists(ckpt):
            with open(ckpt, encoding="utf-8") as f:
                rows = json.load(f)
            print(f"[{i}/{len(spans)}] {begin[:7]}: checkpoint hit ({len(rows)} rows) — skipping")
            all_rows.extend(rows)
            continue

        print(f"[{i}/{len(spans)}] {begin} .. {end}")
        rows = fetch_month(api_key, begin, end, keywords, args.max_pages, args.delay)
        with open(ckpt, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        print(f"    month total: {len(rows)} rows -> {os.path.basename(ckpt)}")
        all_rows.extend(rows)

    df = normalize(all_rows)
    if df.empty:
        print("\nNothing fetched — no file written.")
        return 1

    out = os.path.join(ROOT, f"tech_news_backfill_{args.start}_{args.end}.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} unique articles -> {os.path.basename(out)}")
    print(f"  range: {df.pub_date.min()[:10]} -> {df.pub_date.max()[:10]}")
    per_month = pd.to_datetime(df.pub_date).dt.to_period("M").value_counts().sort_index()
    for period, n in per_month.items():
        print(f"    {period}: {n:5d}")

    if args.merge:
        print("\nMerging into the corpus...")
        merge_into_corpus(df)
    else:
        print("\nNot merged. The tech loader picks up any workspace CSV whose name")
        print("contains 'news', so this file is already live for new runs.")
        print("Use --merge to fold it into merged_news_data.csv instead.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
