#!/usr/bin/env python3
"""
interrater_agreement.py — human-human agreement for the relevance screen

WHY THIS EXISTS
---------------
screen_validation_metrics.py measures the SCREEN against one person's labels.
That answers "does the screen agree with a human", but leaves open "how reliable
is the human standard the screen is being judged against". A reviewer who is
already sceptical can point out that the ground truth is a single author's
judgement with no reliability estimate attached. This script closes that gap by
scoring a second, independent rating of the SAME 180 articles.

The second rater is blind to the screen's decision AND to the first rater's
labels, and sees the rows in a different order (see
screen_validation_sample_rater2.xlsx).

RE-WEIGHTING
------------
The labelling sample is STRATIFIED: it over-samples screen-passed articles
(70 passed + 110 rejected) against a population pass rate of 8.0%. Raw agreement
over the 180 rows therefore over-weights the passed stratum. Every population
figure below is re-weighted by stratum; the per-stratum figures are unweighted
because they condition on a single stratum.

INPUTS
------
  screen_validation_sample.xlsx        sheet "Label"  -> rater 1
  screen_validation_sample_rater2.xlsx sheet "Label"  -> rater 2
  screen_validation_key.csv            id, llm_decision (pass/reject)

USAGE
-----
    python interrater_agreement.py
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

R1_NAME = "screen_validation_sample.xlsx"
# The frozen second rating is the authoritative one; its sha256 is recorded in
# rater2_FINAL.sha256. The working copy is tried only as a fallback, because it
# was overwritten several times while the rating was being collected.
R2_NAMES = ("screen_validation_sample_rater2_FINAL.xlsx",
            "screen_validation_sample_rater2.xlsx")
KEY_NAME = "screen_validation_key.csv"
LABEL_COL = "relevant? (1/0)"

POP_PASS = 110
POP_TOTAL = 1382


def _find(name):
    for base in (HERE, REPO_ROOT):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None


def _load_meta():
    global POP_PASS, POP_TOTAL
    p = os.path.join(HERE, "_meta.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        POP_PASS = int(m.get("pop_pass", POP_PASS))
        POP_TOTAL = int(m.get("pop_total", POP_TOTAL))


def _read_labels(path, who):
    try:
        import openpyxl
    except ImportError:
        sys.exit("openpyxl is required:  pip install openpyxl")
    wb = openpyxl.load_workbook(path, data_only=True)
    if "Label" not in wb.sheetnames:
        sys.exit("%s: no 'Label' sheet" % os.path.basename(path))
    ws = wb["Label"]
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(c).strip() if c is not None else "" for c in rows[0]]
    if "id" not in hdr or LABEL_COL not in hdr:
        sys.exit("%s: expected columns 'id' and %r, found %s"
                 % (os.path.basename(path), LABEL_COL, hdr))
    i_id, i_lab = hdr.index("id"), hdr.index(LABEL_COL)
    out, blank = {}, []
    for r in rows[1:]:
        if r is None or r[i_id] is None:
            continue
        rid = str(r[i_id]).strip()
        v = r[i_lab]
        if v is None or str(v).strip() == "":
            blank.append(rid)
            continue
        try:
            out[rid] = int(float(str(v).strip()))
        except ValueError:
            sys.exit("%s: row %s has non-numeric label %r" % (who, rid, v))
    if blank:
        sys.exit("%s has %d unlabelled row(s) (e.g. %s). Label every row first."
                 % (who, len(blank), ", ".join(blank[:6])))
    bad = {k: v for k, v in out.items() if v not in (0, 1)}
    if bad:
        sys.exit("%s: labels must be 0 or 1; got %s" % (who, list(bad.items())[:5]))
    return out


def _read_key():
    p = _find(KEY_NAME)
    if not p:
        sys.exit("missing %s" % KEY_NAME)
    key = {}
    with open(p, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rid = (row.get("id") or "").strip()
            if not rid:
                continue
            if "screen_pass" in row and str(row["screen_pass"]).strip() != "":
                key[rid] = int(float(row["screen_pass"]))
            else:
                d = (row.get("llm_decision") or "").strip().lower()
                if d not in ("pass", "reject"):
                    sys.exit("%s: unrecognised llm_decision %r for %s"
                             % (KEY_NAME, d, rid))
                key[rid] = 1 if d == "pass" else 0
    return key


def _kappa(a, b, ids, w=None):
    """Cohen's kappa; w maps id -> stratum weight (default 1)."""
    w = w or {}
    tot = sum(w.get(i, 1.0) for i in ids)
    if tot <= 0:
        return float("nan")
    obs = sum(w.get(i, 1.0) for i in ids if a[i] == b[i]) / tot
    pa1 = sum(w.get(i, 1.0) for i in ids if a[i] == 1) / tot
    pb1 = sum(w.get(i, 1.0) for i in ids if b[i] == 1) / tot
    exp = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return float("nan") if abs(1 - exp) < 1e-12 else (obs - exp) / (1 - exp)


def _wilson(k, n, z=1.959963985):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    _load_meta()
    p1 = _find(R1_NAME)
    p2 = next((q for q in (_find(n) for n in R2_NAMES) if q), None)
    if not p1:
        sys.exit("missing %s" % R1_NAME)
    if not p2:
        sys.exit("missing %s — generate it before scoring" % R2_NAMES[0])
    print('rater 2 file: %s' % os.path.basename(p2))

    r1 = _read_labels(p1, "rater 1")
    r2 = _read_labels(p2, "rater 2")
    key = _read_key()

    ids = sorted(set(r1) & set(r2) & set(key))
    if not ids:
        sys.exit("no overlapping ids across the two ratings and the key")
    only1, only2 = set(r1) - set(r2), set(r2) - set(r1)
    if only1 or only2:
        print("WARNING: %d id(s) only in rater 1, %d only in rater 2"
              % (len(only1), len(only2)))

    passed = [i for i in ids if key[i] == 1]
    rejected = [i for i in ids if key[i] == 0]
    pop_rate = POP_PASS / POP_TOTAL
    # stratum weights: each sampled article stands for this many population items
    w = {}
    for i in passed:
        w[i] = (pop_rate / len(passed)) if passed else 0.0
    for i in rejected:
        w[i] = ((1 - pop_rate) / len(rejected)) if rejected else 0.0

    agree = [i for i in ids if r1[i] == r2[i]]
    n11 = sum(1 for i in ids if r1[i] == 1 and r2[i] == 1)
    n10 = sum(1 for i in ids if r1[i] == 1 and r2[i] == 0)
    n01 = sum(1 for i in ids if r1[i] == 0 and r2[i] == 1)
    n00 = sum(1 for i in ids if r1[i] == 0 and r2[i] == 0)

    print("=" * 70)
    print("HUMAN-HUMAN AGREEMENT  (n = %d labelled articles)" % len(ids))
    print("=" * 70)
    print()
    print("                       rater 2: relevant   rater 2: not")
    print("  rater 1: relevant    %14d   %12d" % (n11, n10))
    print("  rater 1: not         %14d   %12d" % (n01, n00))
    print()

    raw = len(agree) / len(ids)
    lo, hi = _wilson(len(agree), len(ids))
    print("  raw agreement (unweighted)      : %.1f%%  (95%% CI [%.1f, %.1f])"
          % (100 * raw, 100 * lo, 100 * hi))
    print("  Cohen's kappa (unweighted)      : %+.2f" % _kappa(r1, r2, ids))
    print()
    print("  re-weighted to the %.1f%% population pass rate:" % (100 * pop_rate))
    wagree = sum(w[i] for i in agree)
    print("     agreement                    : %.1f%%" % (100 * wagree))
    print("     Cohen's kappa                : %+.2f" % _kappa(r1, r2, ids, w))
    print()
    print("  by stratum (unweighted, conditions on one stratum):")
    for lab, grp in (("screen PASSED  ", passed), ("screen REJECTED", rejected)):
        if not grp:
            continue
        a = sum(1 for i in grp if r1[i] == r2[i])
        k = _kappa(r1, r2, grp)
        # If either rater used a single label throughout a stratum there is no
        # variance to correct for, so kappa collapses to 0 (or is undefined)
        # however well the raters actually agree. Flag it rather than print a
        # bare 0.00 that reads as "no agreement".
        const = len({r1[i] for i in grp}) == 1 or len({r2[i] for i in grp}) == 1
        note = "   [kappa degenerate: one rater is constant here]" if const else ""
        print("     %s n=%3d   agreement %.1f%%   kappa %+.2f%s"
              % (lab, len(grp), 100 * a / len(grp), k, note))
    print()

    # how much would the screen's headline numbers move under rater 2?
    def prec(r):
        if not passed:
            return float("nan")
        return sum(r[i] for i in passed) / len(passed)

    def for_(r):
        if not rejected:
            return float("nan")
        return sum(r[i] for i in rejected) / len(rejected)

    print("  screen metrics under each rater (sensitivity of the headline):")
    for nm, r in (("rater 1", r1), ("rater 2", r2)):
        p, f = prec(r), for_(r)
        tp = pop_rate * p
        fn = (1 - pop_rate) * f
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        pl, ph = _wilson(int(round(p * len(passed))), len(passed))
        print("     %s  precision %.1f%% (95%% CI [%.1f, %.1f])   "
              "false-omission %.1f%%   re-weighted recall %.1f%%"
              % (nm, 100 * p, 100 * pl, 100 * ph, 100 * f, 100 * rec))
    print()
    kw = _kappa(r1, r2, ids, w)
    ku = _kappa(r1, r2, ids)
    print("  Manuscript-ready sentence:")
    print("     A second rater, blind to both the screen's decisions and the first")
    print("     rater's labels, independently labelled the same %d articles;" % len(ids))
    print("     the two raters agreed on %.1f%% of them (Cohen's kappa %.2f" % (100 * raw, ku))
    print("     unweighted, %.2f re-weighted to the population pass rate)." % kw)
    print("=" * 70)


if __name__ == "__main__":
    main()
