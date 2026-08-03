#!/usr/bin/env python3
"""
compile_results_to_excel.py
===========================
Scans every perm_*.json in results/ and writes the three permutation Excel
files (permutation_results_T0.3/0.4/0.5.xlsx) using exactly the same logic
as the permutation runner GUI.

Run from the project root:
    python compile_results_to_excel.py
"""

import glob
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from permutation_runner import (
    RESULTS_DIR,
    EXCEL_PATH_FMT,
    TEMPERATURES,
    ST_DONE,
    ST_SKIPPED,
    build_permutations,
    save_excel,
    _is_valid_result,
    _match_perm_by_metadata,
)

DEFAULT_JSON_SOURCE_DIRS = [
    os.path.join(ROOT, "results"),
    os.path.join(ROOT, "results_old"),
]
CONSOLIDATED_JSON_DIR = os.path.join(RESULTS_DIR, "consolidated_json")
ALL_PERMUTATIONS_JSON = os.path.join(RESULTS_DIR, "all_permutations_consolidated.json")


def _is_relative_to(path: str, parent: str) -> bool:
    try:
        return (
            os.path.commonpath([os.path.abspath(path), os.path.abspath(parent)])
            == os.path.abspath(parent)
        )
    except ValueError:
        return False


def _parse_result_timestamp(result: dict, fpath: str) -> datetime:
    """Best-effort timestamp used to choose the newest duplicate JSON run."""
    meta = result.get("metadata", {}) if isinstance(result, dict) else {}
    raw = meta.get("timestamp") or meta.get("run_timestamp")
    if raw:
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass

    match = re.search(r"_(\d{8}_\d{6})\.json$", os.path.basename(fpath))
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass

    try:
        return datetime.fromtimestamp(os.path.getmtime(fpath))
    except OSError:
        return datetime.min


def _result_priority(result: dict, fpath: str) -> tuple:
    """Sort key where higher means newer and more authoritative."""
    norm = os.path.normcase(os.path.abspath(fpath))
    in_primary_results = (
        _is_relative_to(fpath, RESULTS_DIR)
        and not _is_relative_to(fpath, os.path.join(ROOT, "results_old"))
    )
    return (_parse_result_timestamp(result, fpath), 1 if in_primary_results else 0, norm)


def _metadata_identity(result: dict, fallback_path: str) -> tuple:
    """
    Stable identity for a permutation result, independent of legacy perm_id shifts.
    Duplicate JSONs with the same identity are resolved by _result_priority.
    """
    meta = result.get("metadata", {}) if isinstance(result, dict) else {}
    try:
        return (
            str(meta.get("model", "")).strip(),
            round(float(meta.get("temperature", -1)), 6),
            int(meta.get("period_months", -1)),
            str(meta.get("analysis_start_date", ""))[:10],
            str(meta.get("analysis_end_date", ""))[:10],
            int(meta.get("filter_init", -99)),
            int(meta.get("filter_final", -99)),
        )
    except (TypeError, ValueError):
        return ("path", os.path.normcase(os.path.abspath(fallback_path)))


def iter_permutation_json_files(source_dirs: list[str] | None = None) -> list[str]:
    """Return all perm_*.json files from source directories, newest paths last."""
    source_dirs = source_dirs or DEFAULT_JSON_SOURCE_DIRS
    files: list[str] = []
    seen = set()
    for source_dir in source_dirs:
        source_path = os.path.abspath(source_dir)
        if not os.path.isdir(source_path):
            continue
        for fpath in glob.glob(os.path.join(source_path, "**", "perm_*.json"), recursive=True):
            if (
                _is_relative_to(fpath, CONSOLIDATED_JSON_DIR)
                and os.path.abspath(source_path) != os.path.abspath(CONSOLIDATED_JSON_DIR)
            ):
                continue
            norm = os.path.normcase(os.path.abspath(fpath))
            if norm not in seen:
                seen.add(norm)
                files.append(fpath)
    return sorted(files)


def load_consolidated_results(source_dirs: list[str] | None = None) -> tuple[dict[tuple, tuple[dict, str]], int, int]:
    """
    Load all valid permutation JSONs and keep one newest result per permutation identity.
    Returns (identity -> (result, source_path), files_read, valid_runs_seen).
    """
    chosen: dict[tuple, tuple[dict, str]] = {}
    files_read = valid_runs_seen = 0

    for fpath in iter_permutation_json_files(source_dirs):
        files_read += 1
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception as exc:
            print(f"  [warn] Could not read {os.path.basename(fpath)}: {exc}")
            continue

        if not _is_valid_result(result):
            continue

        valid_runs_seen += 1
        identity = _metadata_identity(result, fpath)
        prev = chosen.get(identity)
        if prev is None or _result_priority(result, fpath) > _result_priority(prev[0], prev[1]):
            chosen[identity] = (result, fpath)

    return chosen, files_read, valid_runs_seen


def consolidate_json_files(
    source_dirs: list[str] | None = None,
    output_dir: str | None = None,
) -> tuple[int, int, int, str]:
    """
    Copy the newest valid JSON for each permutation identity into one directory.
    Existing JSONs in the output directory are replaced.
    """
    output_dir = output_dir or CONSOLIDATED_JSON_DIR
    os.makedirs(output_dir, exist_ok=True)

    for old_file in glob.glob(os.path.join(output_dir, "perm_*.json")):
        os.remove(old_file)

    chosen, files_read, valid_runs_seen = load_consolidated_results(source_dirs)
    for result, source_path in chosen.values():
        dest = os.path.join(output_dir, os.path.basename(source_path))
        if os.path.exists(dest):
            stem, ext = os.path.splitext(os.path.basename(source_path))
            identity_text = repr(_metadata_identity(result, source_path)).encode("utf-8")
            identity_suffix = hashlib.sha1(identity_text).hexdigest()[:10]
            dest = os.path.join(output_dir, f"{stem}_{identity_suffix}{ext}")
        shutil.copy2(source_path, dest)

    return files_read, valid_runs_seen, len(chosen), output_dir


def _serialize_permutation(p: dict) -> dict:
    """Return the stable, JSON-friendly fields for one permutation row."""
    return {
        "perm_id": p["id"],
        "model": p["model"],
        "temperature": p["temp"],
        "period_months": p["period"],
        "analysis_start_date": p["start"].strftime("%Y-%m-%d"),
        "analysis_end_date": p["end"].strftime("%Y-%m-%d"),
        "filter_init": p["init_n"],
        "filter_final": p["fin_n"],
        "filter_label": _filter_label_for_export(p["init_n"], p["fin_n"]),
        "skip_reason": p.get("skip"),
    }


def _filter_label_for_export(init_n: int, fin_n: int) -> str:
    if (init_n, fin_n) == (0, 0):
        return "unfiltered"
    if (init_n, fin_n) == (-1, -1):
        return "ranked_final"
    return f"{init_n}->{fin_n}"


def write_all_permutations_json(
    source_dirs: list[str] | None = None,
    output_path: str | None = None,
) -> tuple[int, int, int, int, str]:
    """
    Write one JSON file containing the full current permutation space.
    Each permutation includes the newest consolidated result when available.
    """
    output_path = output_path or ALL_PERMUTATIONS_JSON
    perms = build_permutations()
    perm_by_id = {p["id"]: p for p in perms}
    chosen, files_read, valid_runs_seen = load_consolidated_results(source_dirs)

    latest_by_perm_id: dict[int, tuple[dict, str]] = {}
    unmatched_results = []
    for identity, (result, source_path) in chosen.items():
        p = _match_perm_by_metadata(result, perm_by_id, perms)
        if p is None:
            unmatched_results.append({
                "identity": list(identity),
                "source_file": os.path.relpath(source_path, ROOT),
            })
            continue
        latest_by_perm_id[p["id"]] = (result, source_path)

    rows = []
    for p in perms:
        row = _serialize_permutation(p)
        result_source = latest_by_perm_id.get(p["id"])
        row["has_result"] = result_source is not None
        if result_source:
            result, source_path = result_source
            row["source_file"] = os.path.relpath(source_path, ROOT)
            row["result_timestamp"] = _parse_result_timestamp(result, source_path).isoformat()
            row["result"] = result
        else:
            row["source_file"] = None
            row["result_timestamp"] = None
            row["result"] = None
        rows.append(row)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dirs": [os.path.relpath(d, ROOT) for d in (source_dirs or DEFAULT_JSON_SOURCE_DIRS)],
        "files_scanned": files_read,
        "valid_runs_seen": valid_runs_seen,
        "newest_unique_results": len(chosen),
        "permutation_space_count": len(perms),
        "matched_permutations": len(latest_by_perm_id),
        "unmatched_result_count": len(unmatched_results),
        "unmatched_results": unmatched_results,
        "permutations": rows,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return files_read, valid_runs_seen, len(chosen), len(latest_by_perm_id), output_path


def load_results_into_perms(
    perms: list[dict],
    source_dirs: list[str] | None = None,
    newest_per_permutation: bool = False,
) -> tuple[int, int]:
    """
    Scan perm_*.json files and populate each permutation's `results` list.
    Matches each file to its permutation by metadata content (model, temp,
    period, start, filter) so perm ID shifts from adding new filter configs
    don't corrupt the mapping.
    Returns (files_read, runs_loaded).
    """
    perm_by_id = {p["id"]: p for p in perms}

    files_read = runs_loaded = 0
    if newest_per_permutation:
        chosen, files_read, _valid_runs_seen = load_consolidated_results(source_dirs)
        result_sources = [(result, fpath) for result, fpath in chosen.values()]
    else:
        result_sources = []
        for fpath in iter_permutation_json_files(source_dirs or [RESULTS_DIR]):
            files_read += 1
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    result = json.load(f)
            except Exception as exc:
                print(f"  [warn] Could not read {os.path.basename(fpath)}: {exc}")
                continue
            result_sources.append((result, fpath))

    result_sources.sort(key=lambda item: _result_priority(item[0], item[1]))
    for result, fpath in result_sources:
        if not _is_valid_result(result):
            continue

        p = _match_perm_by_metadata(result, perm_by_id, perms)
        if p is None or p["status"] == ST_SKIPPED:
            continue

        p["results"].append(result)
        runs_loaded += 1

    for p in perms:
        if p["results"]:
            latest = p["results"][-1]
            p["status"] = ST_DONE
            p["result"] = latest
            p["elapsed_s"] = latest.get("metadata", {}).get("elapsed_s", 0)

    return files_read, runs_loaded


def main():
    args = sys.argv[1:]
    include_all_sources = "--all-json-sources" in args
    newest_per_permutation = "--latest-json-only" in args
    copy_consolidated_json = "--consolidate-json" in args
    write_all_json = "--write-all-json" in args
    source_dirs = (
        DEFAULT_JSON_SOURCE_DIRS
        if include_all_sources or newest_per_permutation or copy_consolidated_json or write_all_json
        else [RESULTS_DIR]
    )

    print("=" * 60)
    print("Permutation Results -> Excel Compiler")
    print("=" * 60)

    if copy_consolidated_json:
        files_read, valid_runs_seen, copied_count, out_dir = consolidate_json_files(source_dirs)
        print(
            f"\nConsolidated JSON files: {copied_count} newest unique run(s) "
            f"from {valid_runs_seen} valid run(s) / {files_read} file(s) -> {out_dir}"
        )

    if write_all_json:
        files_read, valid_runs_seen, unique_count, matched_count, out_path = write_all_permutations_json(source_dirs)
        print(
            f"\nAll-permutations JSON: {matched_count} matched permutation(s), "
            f"{unique_count} newest unique result(s) from {valid_runs_seen} valid run(s) / "
            f"{files_read} file(s) -> {out_path}"
        )

    print("\nBuilding permutation parameter space...")
    perms = build_permutations()
    print(f"  {len(perms)} permutations in parameter space")

    source_label = ", ".join(os.path.relpath(d, ROOT) for d in source_dirs)
    print(f"\nLoading JSON result files from {source_label}...")
    if newest_per_permutation:
        print("  Priority: newest valid JSON per permutation identity")
    files_read, runs_loaded = load_results_into_perms(
        perms,
        source_dirs=source_dirs,
        newest_per_permutation=newest_per_permutation,
    )
    print(f"  {files_read} files scanned, {runs_loaded} valid runs loaded")

    if runs_loaded == 0:
        print("\nNo valid results found — nothing to write.")
        return

    print()
    errors = []
    for temp in TEMPERATURES:
        path = EXCEL_PATH_FMT.format(temp=temp)
        count = sum(len(p["results"]) for p in perms if p["temp"] == temp)
        if count == 0:
            print(f"  T={temp}: no runs — skipping")
            continue
        print(f"  T={temp}: {count} run(s) -> {os.path.basename(path)} ...", end=" ", flush=True)
        err = save_excel(perms, path, temp_filter=temp)
        if err:
            print(f"FAILED ({err})")
            errors.append((temp, err))
        else:
            print("OK")

    print()
    if errors:
        print("Errors:")
        for temp, err in errors:
            print(f"  T={temp}: {err}")
        sys.exit(1)
    else:
        print("Done. Excel files written to results/")


if __name__ == "__main__":
    main()
