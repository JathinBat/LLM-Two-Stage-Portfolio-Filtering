"""Graded 1-10 investment-relevance scoring over the tech corpus — Paper B only.

Why this exists
---------------
The binary screen (`screen_news_relevance.py`) gives one corpus at one operating
point: 13.5% pass, take it or leave it. Comparing "screened" against "unscreened"
therefore does not isolate the relevance filter at all, because the two paths use
DIFFERENT selection mechanisms:

    screened=True   tech_screened_corpus.csv, and _process_news_data is called
                    with an EMPTY keyword, so no substring filter runs
    screened=False  merged_news_data.csv + backfills, and _process_news_data IS
                    called with sector="technology" -- a literal substring match

Measured over a 24-month lookback ending 2024-10-01, those two sets overlap on
only 13% of the screened corpus: 636 screened articles never reach the unscreened
path, and 402 substring-matched articles never reach the screened one. So the
-6.31pp corpus effect (p=0.039) measured on that contrast is "two different
article selectors", not "relevance filtering on/off".

Scoring every article 1-10 on ONE scale fixes that. Threshold is then the only
variable, it is tunable after the fact at zero marginal cost, and the same
mechanism produces every corpus -- so a threshold sweep is a clean dose-response
rather than an apples-to-oranges comparison.

Outputs (no filename contains news/article/finance/consolidated, so Paper A's
workspace walk in `_load_news_data` can never pick them up and its published
numbers stand):
  * tech_relevance_scores.csv — EVERY article + its 1-10 score. The artifact.
  * tech_scored_ge<N>.csv     — corpus at threshold N, written by --emit

Usage
-----
    python score_news_relevance.py --dry-run       # batches + cost estimate
    python score_news_relevance.py --limit 60      # small live test
    python score_news_relevance.py                 # score everything
    python score_news_relevance.py --report        # distribution of existing scores
    python score_news_relevance.py --emit 7        # write tech_scored_ge7.csv
    python score_news_relevance.py --emit 5,6,7,8  # several at once

Resumable: scores cache to `score_cache_tech.json` keyed by article URL, so a
re-run after a backfill only scores the new articles.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import Dict, List

import pandas as pd

import env_loader  # noqa: F401  (populates os.environ from .env)

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "merged_news_data.csv")
SCORES = os.path.join(ROOT, "tech_relevance_scores.csv")
CACHE = os.path.join(ROOT, "score_cache_tech.json")

COLUMNS = ["headline", "pub_date", "snippet", "web_url", "source", "section",
           "news_desk", "industry", "search_keyword", "fetch_date"]

BATCH_SIZE = 30
MODEL = "gpt-4o-mini"      # same model as the validated binary screen

# Anchored rubric. Without explicit anchors a 1-10 request drifts toward 6-8 for
# everything and the scale carries no information; the bands below are what make
# a threshold sweep interpretable.
SYSTEM_PROMPT = """You rate news articles for their usefulness to a SERIOUS STOCK INVESTMENT decision in the technology sector.

Score EVERY article from 1 to 10 using these anchors:

10-9  DIRECT, MATERIAL, SPECIFIC. Earnings results or guidance, M&A, major
      product/chip launches, analyst upgrades/downgrades, regulatory action, or
      Fed/macro decisions — naming specific publicly traded companies, with
      concrete figures or a clear market consequence.
8-7   DIRECTLY RELEVANT. Named public company business developments, sector
      demand shifts, supply-chain or pricing news with obvious read-through to
      revenue or margins, even without hard numbers.
6-5   CONTEXTUAL. Industry trends, competitive dynamics, macro conditions or
      policy that affect the sector broadly but name no company and imply no
      specific trade.
4-3   TANGENTIAL. General technology coverage, product reviews, opinion or
      commentary, company news with no financial angle.
2-1   IRRELEVANT to investing. Entertainment, sports, lifestyle, crime, weather,
      human interest, politics with no market implication.

Judge the article on its own content. Do NOT inflate: a typical general-news
corpus should have most articles in the 1-4 range.

Return ONE line per article, exactly "<index>:<score>", nothing else.
Example:
1:8
2:3
3:10"""


def load_inputs(extra: List[str]) -> pd.DataFrame:
    paths = [CORPUS] + sorted(glob.glob(os.path.join(ROOT, "tech_news_backfill_*.csv")))
    paths += [p for p in extra if p not in paths]
    frames = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  (skipping missing {os.path.basename(p)})")
            continue
        df = pd.read_csv(p, low_memory=False)
        for c in COLUMNS:
            if c not in df.columns:
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


def load_cache() -> Dict[str, int]:
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return {k: int(v) for k, v in json.load(f).items()}
    return {}


def save_cache(c: Dict[str, int]) -> None:
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(c, f)


def score_batch(client, batch: pd.DataFrame) -> Dict[int, int]:
    """{row position (1-based) -> score 1..10}. Missing entries mean unparsed."""
    lines = []
    for pos, (_, row) in enumerate(batch.iterrows(), start=1):
        hl = str(row.get("headline", ""))[:120]
        sn = str(row.get("snippet", ""))[:180]
        lines.append(f"{pos}. {hl} - {sn}")
    user = "Score these articles 1-10 for investment relevance:\n\n" + "\n".join(lines)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
                temperature=0.1,
                max_tokens=16 * len(batch) + 64,
            )
            text = (resp.choices[0].message.content or "")
            out = {}
            for m in re.finditer(r"(\d+)\s*[:.\)]\s*(\d+)", text):
                pos, sc = int(m.group(1)), int(m.group(2))
                if 1 <= pos <= len(batch) and 1 <= sc <= 10:
                    out[pos] = sc
            if out:
                return out
            print("      unparseable reply; retrying")
        except Exception as e:                                    # noqa: BLE001
            wait = 2 ** attempt
            print(f"      attempt {attempt + 1}/3 failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    return {}


def emit(thresholds: List[int]) -> None:
    if not os.path.exists(SCORES):
        print(f"no {os.path.basename(SCORES)} yet — run the scorer first")
        return
    df = pd.read_csv(SCORES, low_memory=False)
    df = df[df.relevance_score.notna()]
    for n in thresholds:
        sub = df[df.relevance_score >= n]
        out = os.path.join(ROOT, f"tech_scored_ge{n}.csv")
        sub[COLUMNS].to_csv(out, index=False)
        print(f"  {os.path.basename(out):32s} {len(sub):6d} articles  "
              f"({100 * len(sub) / len(df):5.1f}% of scored)")


def report() -> None:
    if not os.path.exists(SCORES):
        print("no scores yet")
        return
    df = pd.read_csv(SCORES, low_memory=False)
    s = df.relevance_score.dropna()
    print(f"scored articles: {len(s)} of {len(df)}")
    print()
    print("  score  count    share   cumulative (>= score)")
    for n in range(10, 0, -1):
        c = int((s == n).sum())
        cum = int((s >= n).sum())
        print(f"   {n:2d}   {c:6d}  {100*c/len(s):6.2f}%   {cum:6d}  ({100*cum/len(s):5.1f}%)")
    print()
    print(f"  mean {s.mean():.2f}   median {s.median():.0f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--emit", default="")
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()

    if args.report:
        report()
        return 0
    if args.emit:
        emit([int(x) for x in args.emit.replace(",", " ").split()])
        return 0

    print("Loading corpus...")
    df = load_inputs(args.extra)
    if df.empty:
        print("no articles")
        return 1

    cache = load_cache()
    df["_key"] = df.apply(article_key, axis=1)
    todo = df[~df._key.isin(cache)]
    if args.limit:
        todo = todo.head(args.limit)

    nb = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n  total {len(df)}   cached {len(df) - len(todo[~todo._key.isin(cache)]) if False else len(cache)}"
          f"   to score {len(todo)}   batches {nb}  (model {MODEL})")
    if args.dry_run:
        print("  dry run — no API calls made")
        return 0

    if todo.empty:
        print("  nothing new to score")
    else:
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            print("OPENAI_API_KEY not set")
            return 1
        client = OpenAI(api_key=key)
        done = 0
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo.iloc[i:i + BATCH_SIZE]
            got = score_batch(client, batch)
            for pos, (_, row) in enumerate(batch.iterrows(), start=1):
                if pos in got:
                    cache[article_key(row)] = got[pos]
            done += len(batch)
            if (i // BATCH_SIZE) % 10 == 0 or done >= len(todo):
                save_cache(cache)
                print(f"    {done}/{len(todo)} scored  (cache {len(cache)})")
        save_cache(cache)

    df["relevance_score"] = df._key.map(cache)
    df.drop(columns=["_key"]).to_csv(SCORES, index=False)
    print(f"\nWrote {os.path.basename(SCORES)}  "
          f"({int(df.relevance_score.notna().sum())} scored of {len(df)})")
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
