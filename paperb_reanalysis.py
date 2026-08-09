#!/usr/bin/env python3
"""Paper B (URTC) reanalysis suite — addresses IEEE peer review §4.2-4.6.

Run from the ASDRP-LLM-Long-Term-Investment-Strategy repo root:
    python paperb_reanalysis.py window-stats     # §4.2: n=9 window-level tests + Holm
    python paperb_reanalysis.py bl-cells         # §4.6: why only 45/144 cells scorable
    python paperb_reanalysis.py selection-audit  # §4.3: was the model portfolio equal-weighted?

Written without access to the live repo: the loaders sniff column names and
fail loudly with the actual schema so mismatches are obvious, not silent.
"""
import sys, csv, glob, json, collections, statistics, itertools, os

def sniff(path, want):
    """Return {concept: actual_column} for the first matching alias, or die verbosely."""
    with open(path, newline='', encoding='utf-8-sig') as f:
        cols = next(csv.reader(f))
    out = {}
    for concept, aliases in want.items():
        hit = next((c for c in cols for a in aliases if a == c.lower().strip()), None)
        if hit is None:
            sys.exit(f"[{path}] no column for {concept!r}; have {cols}\n"
                     f"  -> edit the alias list in sniff() call for this concept")
        out[concept] = hit
    return out

def wilcoxon_exact(diffs):
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0: return 1.0
    ranks = {}
    for i, (a, _) in enumerate(sorted((abs(x), j) for j, x in enumerate(d))):
        ranks[_] = i + 1  # no tie handling; fine for continuous ROI diffs
    W = sum(r for j, r in ranks.items() if d[j] > 0)
    total = n * (n + 1) // 2
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def count(k, s):
        if s < 0: return 0
        if k == 0: return 1 if s == 0 else 0
        return count(k - 1, s) + count(k - 1, s - k)
    le = sum(count(n, s) for s in range(0, min(W, total - W) + 1))
    return min(1.0, 2 * le / (2 ** n))

def sign_test(wins, n):
    from math import comb
    k = min(wins, n - wins)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def holm(pvals):
    """pvals: list of (label, p). Returns [(label, p, p_holm)]."""
    m = len(pvals)
    srt = sorted(pvals, key=lambda x: x[1])
    out, running = [], 0.0
    for i, (lab, p) in enumerate(srt):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out.append((lab, p, running))
    return out

# ---------------------------------------------------------------- window-stats
def window_stats():
    # Candidate per-run sources, most likely first. Adjust if the real file differs.
    candidates = ["tier1a_per_run.csv", "paper_b/per_run.csv", "paper_b/runs.csv"]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        sys.exit(f"none of {candidates} found — point me at the per-run results file")
    col = sniff(path, {
        "window": ["window", "window_start", "start", "period_start"],
        "arm":    ["arm", "condition", "variant", "config", "label"],
        "roi":    ["roi", "return", "ret", "final_return", "roi_pct", "total_return"],
        "init":   ["init", "init_roi", "baseline_roi", "static_roi", "init_return"],
    })
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8-sig')))
    print(f"[{path}] {len(rows)} rows; arms: "
          f"{sorted(set(r[col['arm']] for r in rows))}")
    byarm = collections.defaultdict(lambda: collections.defaultdict(lambda: {'roi': [], 'init': []}))
    for r in rows:
        try:
            byarm[r[col['arm']]][r[col['window']]]['roi'].append(float(r[col['roi']]))
            byarm[r[col['arm']]][r[col['window']]]['init'].append(float(r[col['init']]))
        except ValueError:
            continue
    primary, exploratory = [], []
    print(f"\n{'ARM':<28}{'n win':>6}{'mean diff pp':>14}{'wins':>7}{'sign p':>9}{'wilcoxon p':>12}")
    for arm, wins_d in sorted(byarm.items()):
        diffs = [statistics.mean(v['roi']) - statistics.mean(v['init'])
                 for w, v in sorted(wins_d.items()) if v['roi'] and v['init']]
        n = len(diffs); w = sum(1 for d in diffs if d > 0)
        sp, wp = sign_test(w, n), wilcoxon_exact(diffs)
        print(f"{arm:<28}{n:>6}{statistics.mean(diffs):>14.2f}{f'{w}/{n}':>7}{sp:>9.3f}{wp:>12.3f}")
        (primary if 'init' in arm.lower() or True else exploratory).append((arm, wp))
    print("\nHolm-adjusted (treating all arm-vs-INIT tests as one family):")
    for lab, p, ph in holm(primary):
        print(f"  {lab:<28} p={p:.4f}  holm={ph:.4f}")
    print("\nNOTE: paper's primary hypotheses = arm-vs-INIT and equal-weight-vs-model-weight."
          "\nIf sizing contrasts live in a separate file, rerun with that file for family 2.")

# ------------------------------------------------------------------- bl-cells
def bl_cells():
    hits = glob.glob("**/*sizing*.csv", recursive=True) + glob.glob("**/*black*litterman*.csv", recursive=True)
    if not hits:
        sys.exit("no sizing/BL results csv found — check results/ layout; need per-cell "
                 "records of which sizing rules scored to tabulate the 45-cell subset")
    path = hits[0]
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8-sig')))
    print(f"[{path}] {len(rows)} rows; columns: {list(rows[0].keys())}")
    print("→ tabulate scorable-vs-failed cells by window and arm from these columns, "
          "then check failures are not concentrated (reviewer §4.6 selection-bias concern).")

# ------------------------------------------------------------- selection-audit
def selection_audit():
    path = "selection_null_periods.csv"
    if not os.path.exists(path):
        sys.exit(f"{path} not found")
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8-sig')))
    print(f"[{path}] {len(rows)} rows; columns: {list(rows[0].keys())}")
    wcols = [c for c in rows[0] if 'weight' in c.lower()]
    print("weight-related columns:", wcols or "NONE — check selection_null.py directly for "
          "whether the model portfolio return used conviction weights or equal weights")
    print("Reviewer §4.3 requires: model portfolio EQUAL-WEIGHTED, same holding count, "
          "same dates and price treatment as the 10,000 null draws. If conviction weights "
          "were used, rerun the null scoring with the model's names equal-weighted "
          "(pure price arithmetic; no API calls).")

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'window-stats'
    {'window-stats': window_stats, 'bl-cells': bl_cells, 'selection-audit': selection_audit}[cmd]()
