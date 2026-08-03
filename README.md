# Improving the Performance of Long-Term Stock Investments with a Two-Stage LLM Filtering Framework

Code, run outputs, and paper assets for the NHSJS submission.

## What this is

Most LLM investing systems put news and financial reports into a single prompt.
This project tests whether *sequencing* those sources changes results: news first
nominates a candidate pool (Stage One), then company financial reports narrow that
pool into a final portfolio (Stage Two). The notation `X→Y` means X candidates from
news, narrowed to Y holdings after report validation.

Across 2,759 analysed backtest runs spanning three OpenAI models, ten filtering
configurations, and rolling 12-month windows, moderate two-stage filtering beats an
unfiltered single-pass baseline given the same information.

## Layout

| path | contents |
|---|---|
| `permutation_runner.py` | the permutation grid runner (models × configurations × windows) |
| `pipeline/investment_strategy_generator.py` | the pipeline engine: screening, Stage One, Stage Two, baselines |
| `ablation_experiment.py` | the component ablation (single-prompt, budget-matched, no-elimination, shuffled-reports, hold-all) |
| `calculate_sharpe.py`, `risk_metrics.py` | risk reconstruction from tickers, weights and dates |
| `benchmark_stats.py` | benchmark comparisons: XLK, SMH, equal-weight universe, random matched portfolios |
| `revision_stats.py` | paired per-window statistics and the per-cell run breakdown |
| `build_paper_analysis_tables.py` | paper-ready summary tables |
| `sector_config.py` | sector universe, benchmark and news-source configuration |
| `temporal_compliance/` | look-ahead validation tests and results |
| `results/perm_*.json` | per-run outputs: selected tickers, weights, decision dates, returns |
| `results/ablation/` | ablation run outputs |
| `results/sharpe_analysis.xlsx` | per-run risk metrics |
| `results/logs/` | error reports for generations that returned no parseable portfolio |
| `ablation_results.csv` | parsed per-arm ablation results |
| `paper_assets/docs/NHSJS/` | manuscript, response to reviewers, supplementary information |
| `run_inventory.csv` | run accounting: every generated file traced to the analysed set |
| `pipeline_prompts.json` | the prompts used at every stage |

## Reproducibility

`paper_assets/docs/NHSJS/Supplementary Information.docx` contains the evaluation-window
list, full run accounting, per-cell run counts, per-configuration performance and risk
statistics, the ablation with paired tests, and all pipeline prompts.

Run accounting reconciles as follows: of the generated run files, 2,759 are analysed —
921 GPT-4o, 1,212 GPT-4o-mini, 626 GPT-5.1 — after excluding non-12-month holding
periods, exploratory model tags, and generations that returned no parseable portfolio.
No configuration is missing from any window for any model.

## Data

News articles come from the New York Times Article Search API; company financial reports
come from the Alpha Vantage company-report endpoints. Both corpora are licensed from
their providers and **are not redistributed here**, consistent with the manuscript's Data
and Code Availability statement. What is included: the prompts, the evaluation-window
list, the per-run outputs (tickers, weights, dates), and the analysis scripts — enough to
reconstruct every reported statistic. The retrieval queries and date conventions needed
to rebuild the corpora are documented in the manuscript's Methods.

Financial reports are admitted to a decision only if they were public strictly before
its decision date: availability is the later of the fiscal period end plus the statutory
filing deadline (45 days quarterly, 75 days annual) and any filing date supplied by the
source. This is enforced in `_filter_financial_records_as_of`.

## Configuration

No credentials are stored in this repository. Copy `.env.example` to `.env` and set:

```
OPENAI_API_KEY=...
ALPHA_VANTAGE_API_KEY=...
NYT_API_KEY=...
```

`env_loader` reads these at import time. Nothing reads a key from a config file or a
literal in source.

## Citation

Please cite the NHSJS article. See `paper_assets/docs/NHSJS/` for the manuscript.
