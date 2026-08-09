# URTC Paper B — Reinvestment & Portfolio Simulator

Code, run corpus, and analysis for "Where LLM Portfolio Value Is Created and Lost"
(IEEE MIT URTC submission), moved here from the ASDRP-LLM-Long-Term-Investment-Strategy
repository's dev branch.

- `portfolio_sim_runner.py` / `portfolio_sim_permutation_runner.py` — cash-aware simulator + grid runner
- `black_litterman.py`, `selection_null.py`, `rebalance_attribution.py`, `tier1a_baselines.py`,
  `analyze_feedback.py`, `paperb_benchmarks.py`, `paperb_reanalysis.py` (repo root) — analysis
- `results/portfolio_sim/<arm>/` — the 729-run corpus (per-run JSON with daily value series and risk metrics)
- `tech_*.csv`, `screen_*`/`score_*` — news corpus, relevance screens, and threshold variants
- `paper_b/` — experiment plans and figures; `SESSION_HANDOFF_PAPER_B.md` — analysis notes
- The manuscript and revision summary live at `paper_assets/docs/URTC/`
