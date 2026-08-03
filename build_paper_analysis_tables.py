#!/usr/bin/env python3
"""Build paper-ready analysis tables from current permutation JSON results."""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats

from permutation_runner import (
    FILTER_CONFIGS,
    MODELS,
    PERIOD_MONTHS,
    RESULTS_DIR,
    START_DATES,
    TEMPERATURES,
    ST_SKIPPED,
    build_permutations,
    _filter_label,
    _invalid_result_reason,
    _is_valid_result,
    _iter_saved_result_json_files,
    _match_perm_by_metadata,
)


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = Path(RESULTS_DIR)
OUTPUT_XLSX = RESULTS_PATH / "paper_analysis_tables.xlsx"
OUTPUT_JSON = RESULTS_PATH / "paper_analysis_tables.json"


def _cv(series: pd.Series) -> float | None:
    mean = series.mean()
    sd = series.std(ddof=1)
    if pd.isna(mean) or abs(mean) < 1e-12 or pd.isna(sd):
        return None
    return sd / abs(mean)


def _ci95(series: pd.Series) -> tuple[float | None, float | None, float | None]:
    vals = series.dropna().astype(float)
    n = len(vals)
    if n == 0:
        return None, None, None
    mean = vals.mean()
    if n == 1:
        return mean, mean, 0.0
    se = vals.std(ddof=1) / math.sqrt(n)
    margin = stats.t.ppf(0.975, n - 1) * se
    return mean - margin, mean + margin, margin


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_run_dataframe() -> tuple[pd.DataFrame, dict]:
    perms = build_permutations()
    perm_by_id = {p["id"]: p for p in perms}
    rows = []
    files_read = invalid = unmatched = skipped = 0

    for fpath in _iter_saved_result_json_files():
        files_read += 1
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            invalid += 1
            continue
        if not _is_valid_result(result):
            invalid += 1
            continue
        p = _match_perm_by_metadata(result, perm_by_id, perms)
        if p is None:
            unmatched += 1
            continue
        if p["status"] == ST_SKIPPED:
            skipped += 1
            continue

        meta = result.get("metadata", {})
        ret = result.get("return_analysis", {})
        bench = ret.get("benchmark_comparison", {}) or {}
        roi = _safe_float(ret.get("total_roi"))
        market = _safe_float(bench.get("market_return"))
        alpha = _safe_float(bench.get("alpha"))
        if alpha is None and roi is not None and market is not None:
            alpha = roi - market

        rows.append({
            "Perm ID": p["id"],
            "Model": p["model"],
            "Temperature": p["temp"],
            "Period (mo)": p["period"],
            "Start": p["start"].strftime("%Y-%m"),
            "End": p["end"].strftime("%Y-%m"),
            "Window": f"{p['start']:%b %Y}-{p['end']:%b %Y}",
            "Filter": _filter_label(p["init_n"], p["fin_n"]),
            "ROI": roi,
            "S&P Return": market,
            "Alpha": alpha,
            "Beat S&P": bool(alpha is not None and alpha > 0),
            "Source File": str(Path(fpath).relative_to(ROOT)),
            "Timestamp": meta.get("timestamp"),
            "Elapsed Seconds": meta.get("elapsed_s"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["Temperature", "Model", "Start", "Period (mo)", "Filter", "Perm ID", "Timestamp"],
            kind="stable",
        ).reset_index(drop=True)
    diagnostics = {
        "files_read": files_read,
        "invalid_or_unreadable_files": invalid,
        "unmatched_valid_files": unmatched,
        "skipped_valid_files": skipped,
        "matched_valid_runs": len(df),
    }
    return df, diagnostics


def dataset_summary(df: pd.DataFrame, diagnostics: dict) -> pd.DataFrame:
    perms = build_permutations()
    valid_perms = [p for p in perms if p["status"] != ST_SKIPPED]
    metrics = [
        ("Total permutations generated", len(perms)),
        ("Total valid permutations after training-cutoff filtering", len(valid_perms)),
        ("Number of models", len(MODELS)),
        ("Number of temperatures", len(TEMPERATURES)),
        ("Number of evaluation windows", len({(p["start"], p["end"]) for p in valid_perms})),
        ("Number of holding periods", len(PERIOD_MONTHS)),
        ("Number of filtering configurations", len(FILTER_CONFIGS)),
        ("Total portfolio runs analyzed", len(df)),
        ("Matched unique permutations", df["Perm ID"].nunique() if not df.empty else 0),
        ("JSON files scanned", diagnostics["files_read"]),
        ("Invalid/unreadable JSON files skipped", diagnostics["invalid_or_unreadable_files"]),
        ("Unmatched valid JSON files skipped", diagnostics["unmatched_valid_files"]),
    ]
    return pd.DataFrame(metrics, columns=["Metric", "Value"])


def grouped_roi_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        roi = grp["ROI"].dropna().astype(float)
        ci_low, ci_high, ci_margin = _ci95(roi)
        q1 = roi.quantile(0.25) if len(roi) else None
        q3 = roi.quantile(0.75) if len(roi) else None
        row = dict(zip(group_cols, key))
        row.update({
            "Mean ROI": roi.mean() if len(roi) else None,
            "Median ROI": roi.median() if len(roi) else None,
            "SD": roi.std(ddof=1) if len(roi) > 1 else 0.0 if len(roi) == 1 else None,
            "CV": _cv(roi),
            "Min ROI": roi.min() if len(roi) else None,
            "Max ROI": roi.max() if len(roi) else None,
            "IQR": (q3 - q1) if len(roi) else None,
            "95% CI Low": ci_low,
            "95% CI High": ci_high,
            "95% CI +/-": ci_margin,
            "Sample Size": len(roi),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def configuration_vs_unfiltered(df: pd.DataFrame, by_model: bool = False) -> pd.DataFrame:
    context_cols = ["Model", "Temperature", "Period (mo)", "Start", "End"]
    if not by_model:
        output_group_cols = ["Filter"]
    else:
        output_group_cols = ["Model", "Filter"]

    means = (
        df.groupby(context_cols + ["Filter"], dropna=False)["ROI"]
        .mean()
        .reset_index()
    )
    base = means[means["Filter"] == "unfiltered"][context_cols + ["ROI"]].rename(
        columns={"ROI": "Unfiltered ROI"}
    )
    comp = means[means["Filter"] != "unfiltered"].merge(base, on=context_cols, how="inner")
    comp["ROI Difference"] = comp["ROI"] - comp["Unfiltered ROI"]
    comp["Beat Unfiltered"] = comp["ROI Difference"] > 0

    rows = []
    for key, grp in comp.groupby(output_group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(output_group_cols, key))
        row.update({
            "Win Rate vs Unfiltered": grp["Beat Unfiltered"].mean(),
            "Average ROI Difference": grp["ROI Difference"].mean(),
            "Median ROI Difference": grp["ROI Difference"].median(),
            "Number of Comparisons": len(grp),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        output_group_cols + ["Win Rate vs Unfiltered"],
        ascending=[True] * len(output_group_cols) + [False],
        kind="stable",
    )


def best_configuration_per_model(df: pd.DataFrame) -> pd.DataFrame:
    stats_df = grouped_roi_stats(df, ["Model", "Filter"])
    idx = stats_df.groupby("Model")["Mean ROI"].idxmax()
    return (
        stats_df.loc[idx, ["Model", "Filter", "Mean ROI", "Median ROI", "Sample Size"]]
        .rename(columns={"Filter": "Best Config"})
        .sort_values("Model", kind="stable")
    )


def temperature_analysis(df: pd.DataFrame) -> pd.DataFrame:
    temp_stats = grouped_roi_stats(df, ["Temperature"])
    cfg = grouped_roi_stats(df, ["Temperature", "Filter"])
    best = cfg.loc[cfg.groupby("Temperature")["Mean ROI"].idxmax(), ["Temperature", "Filter", "Mean ROI"]]
    best = best.rename(columns={"Filter": "Best Configuration", "Mean ROI": "Best Config Mean ROI"})
    return temp_stats.merge(best, on="Temperature", how="left")


def evaluation_window_analysis(df: pd.DataFrame) -> pd.DataFrame:
    cfg = grouped_roi_stats(df, ["Period (mo)", "Start", "End", "Window", "Filter"])
    best = cfg.loc[cfg.groupby(["Period (mo)", "Start", "End"])["Mean ROI"].idxmax()]
    worst = cfg.loc[cfg.groupby(["Period (mo)", "Start", "End"])["Mean ROI"].idxmin()]
    overall = grouped_roi_stats(df, ["Period (mo)", "Start", "End", "Window"])
    sp = (
        df.groupby(["Period (mo)", "Start", "End"], dropna=False)["S&P Return"]
        .mean()
        .reset_index()
        .rename(columns={"S&P Return": "S&P Return"})
    )
    return (
        overall.merge(sp, on=["Period (mo)", "Start", "End"], how="left")
        .merge(best[["Period (mo)", "Start", "End", "Filter", "Mean ROI"]].rename(
            columns={"Filter": "Best Config", "Mean ROI": "Best Config Mean ROI"}
        ), on=["Period (mo)", "Start", "End"], how="left")
        .merge(worst[["Period (mo)", "Start", "End", "Filter", "Mean ROI"]].rename(
            columns={"Filter": "Worst Config", "Mean ROI": "Worst Config Mean ROI"}
        ), on=["Period (mo)", "Start", "End"], how="left")
    )


def sp_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for filt, grp in df.groupby("Filter", dropna=False):
        alpha = grp["Alpha"].dropna().astype(float)
        rows.append({
            "Filter": filt,
            "Mean Excess Return": alpha.mean() if len(alpha) else None,
            "Beat S&P %": grp["Beat S&P"].mean() if len(grp) else None,
            "Average Alpha": alpha.mean() if len(alpha) else None,
            "Median Alpha": alpha.median() if len(alpha) else None,
            "Sample Size": len(alpha),
        })
    return pd.DataFrame(rows).sort_values("Average Alpha", ascending=False, kind="stable")


def statistical_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tests = []
    for label, col in [("Configurations", "Filter"), ("Models", "Model")]:
        groups = [g["ROI"].dropna().astype(float).values for _, g in df.groupby(col)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups)
            tests.append({"Test": f"ANOVA across {label}", "Statistic": f_stat, "p-value": p_value})

    def add_ttest(name: str, a: pd.Series, b: pd.Series):
        a = a.dropna().astype(float)
        b = b.dropna().astype(float)
        if len(a) > 1 and len(b) > 1:
            stat, p = stats.ttest_ind(a, b, equal_var=False)
            tests.append({"Test": name, "Statistic": stat, "p-value": p})

    config_stats = grouped_roi_stats(df, ["Filter"])
    best_config = config_stats.sort_values("Mean ROI", ascending=False).iloc[0]["Filter"]
    add_ttest(
        f"Welch t-test: {best_config} vs unfiltered",
        df[df["Filter"] == best_config]["ROI"],
        df[df["Filter"] == "unfiltered"]["ROI"],
    )
    add_ttest("Welch t-test: gpt-5.1 vs gpt-4o", df[df["Model"] == "gpt-5.1"]["ROI"], df[df["Model"] == "gpt-4o"]["ROI"])
    add_ttest("Welch t-test: gpt-5.1 vs gpt-4o-mini", df[df["Model"] == "gpt-5.1"]["ROI"], df[df["Model"] == "gpt-4o-mini"]["ROI"])

    tukey = tukey_hsd(df, "Filter", "ROI")
    return pd.DataFrame(tests), tukey


def tukey_hsd(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    groups = {
        str(name): grp[value_col].dropna().astype(float).values
        for name, grp in df.groupby(group_col, dropna=False)
        if len(grp[value_col].dropna()) > 1
    }
    k = len(groups)
    n_total = sum(len(v) for v in groups.values())
    df_error = n_total - k
    if k < 2 or df_error <= 0:
        return pd.DataFrame()
    ss_error = sum(((vals - vals.mean()) ** 2).sum() for vals in groups.values())
    mse = ss_error / df_error
    rows = []
    for a, b in combinations(groups, 2):
        vals_a = groups[a]
        vals_b = groups[b]
        diff = vals_a.mean() - vals_b.mean()
        se = math.sqrt(mse / 2 * (1 / len(vals_a) + 1 / len(vals_b)))
        q_stat = abs(diff) / se if se else None
        p_value = stats.studentized_range.sf(q_stat, k, df_error) if q_stat is not None else None
        rows.append({
            "Group A": a,
            "Group B": b,
            "Mean Difference": diff,
            "q statistic": q_stat,
            "p-value": p_value,
            "Reject at 0.05": bool(p_value is not None and p_value < 0.05),
        })
    return pd.DataFrame(rows).sort_values("p-value", kind="stable")


def headline_numbers(df: pd.DataFrame) -> pd.DataFrame:
    cfg = grouped_roi_stats(df, ["Filter"])
    model = grouped_roi_stats(df, ["Model"])
    vs_unfiltered = configuration_vs_unfiltered(df)
    reliability = grouped_roi_stats(df, ["Filter"])
    strong_threshold = cfg["Mean ROI"].quantile(0.75)
    strong = reliability[reliability["Mean ROI"] >= strong_threshold].sort_values("CV")

    best_cfg = cfg.sort_values("Mean ROI", ascending=False).iloc[0]
    unfiltered = cfg[cfg["Filter"] == "unfiltered"].iloc[0] if (cfg["Filter"] == "unfiltered").any() else None
    gpt51 = model[model["Model"] == "gpt-5.1"].iloc[0] if (model["Model"] == "gpt-5.1").any() else None
    beat = vs_unfiltered.sort_values("Win Rate vs Unfiltered", ascending=False).iloc[0]
    stable = strong.iloc[0] if len(strong) else reliability.sort_values("CV").iloc[0]
    return pd.DataFrame([
        {"Headline Metric": "Best mean ROI configuration", "Value": best_cfg["Filter"], "Statistic": best_cfg["Mean ROI"]},
        {"Headline Metric": "Unfiltered mean ROI", "Value": "unfiltered", "Statistic": None if unfiltered is None else unfiltered["Mean ROI"]},
        {"Headline Metric": "GPT-5.1 mean ROI", "Value": "gpt-5.1", "Statistic": None if gpt51 is None else gpt51["Mean ROI"]},
        {"Headline Metric": "Highest beat-rate vs unfiltered", "Value": beat["Filter"], "Statistic": beat["Win Rate vs Unfiltered"]},
        {"Headline Metric": "Lowest CV among strong-performing configurations", "Value": stable["Filter"], "Statistic": stable["CV"]},
    ])


def to_jsonable_table(df: pd.DataFrame) -> list[dict]:
    clean = df.where(pd.notnull(df), None)
    return clean.to_dict(orient="records")


def main() -> int:
    df, diagnostics = load_run_dataframe()
    tables = {
        "Dataset Summary": dataset_summary(df, diagnostics),
        "Overall Configuration Performance": grouped_roi_stats(df, ["Filter"]).sort_values("Mean ROI", ascending=False, kind="stable"),
        "Configuration vs Unfiltered": configuration_vs_unfiltered(df).sort_values("Win Rate vs Unfiltered", ascending=False, kind="stable"),
        "Model Performance": grouped_roi_stats(df, ["Model"]).sort_values("Mean ROI", ascending=False, kind="stable"),
        "Best Configuration Per Model": best_configuration_per_model(df),
        "Win Rates By Model": configuration_vs_unfiltered(df, by_model=True).sort_values(["Model", "Win Rate vs Unfiltered"], ascending=[True, False], kind="stable"),
        "Temperature Analysis": temperature_analysis(df),
        "Evaluation Window Analysis": evaluation_window_analysis(df),
        "S&P Comparison": sp_comparison(df),
        "Reliability Metrics": grouped_roi_stats(df, ["Filter"]).sort_values("CV", kind="stable"),
        "Headline Numbers": headline_numbers(df),
    }
    stats_tests, tukey = statistical_tests(df)
    tables["Statistical Tests"] = stats_tests
    tables["Tukey HSD Configs"] = tukey
    tables["Run Level Data"] = df

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            safe_name = sheet_name[:31]
            table.to_excel(writer, sheet_name=safe_name, index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                width = min(max(len(str(cell.value or "")) for cell in col) + 2, 80)
                ws.column_dimensions[col[0].column_letter].width = width

    payload = {
        "outputs": {
            "xlsx": str(OUTPUT_XLSX),
            "json": str(OUTPUT_JSON),
        },
        "diagnostics": diagnostics,
        "tables": {name: to_jsonable_table(table) for name, table in tables.items() if name != "Run Level Data"},
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"Wrote {OUTPUT_XLSX}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Analyzed {len(df)} matched valid portfolio run(s)")
    print(tables["Dataset Summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
