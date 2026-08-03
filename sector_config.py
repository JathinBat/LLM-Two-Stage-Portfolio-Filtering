"""
sector_config.py — Sector registry for the two-stage LLM filtering pipeline.

Purpose
-------
The original pipeline was hard-coded to the technology sector in three places:
  * permutation_runner.py    -> KEYWORD = "technology", NEWS_DATA_PATH = merged_news_data.csv
  * investment_strategy_generator.py -> UNFILTERED_TICKER_UNIVERSE (tech list), benchmark "SPY"
  * the news loader globbed every news CSV in the workspace (so a non-tech run still
    ingested tech news).

This module makes the sector a single, explicit configuration so the pipeline can be
re-targeted (e.g. for the sector-generalization study) without touching analysis logic.

Usage
-----
Select the active sector with the SECTOR environment variable (default "technology"):

    SECTOR=financials  python permutation_runner.py
    SECTOR=healthcare  python permutation_runner.py
    # (unset) -> technology, identical to the original behaviour

The runner and generator read the active sector through the helper functions below.
Adding a new sector = adding one entry to SECTORS (universe + news file + benchmark).

NOTE — gating data: defining a universe does NOT supply news. Each non-tech sector
needs its own dated, pre-window news CSV (see news_path / news_dir and sectors/README.md).
Until that file exists, a sector run has no sector-specific news to screen.
"""

import os

# --------------------------------------------------------------------------- #
#  Technology — the existing universe (unchanged; kept here as the default)    #
# --------------------------------------------------------------------------- #
_TECH_TICKER_INFO = {
    "GOOGL": ("Mega-Cap Tech", "Alphabet"),   "MSFT": ("Mega-Cap Tech", "Microsoft"),
    "NVDA":  ("Semiconductors", "NVIDIA"),     "AAPL": ("Mega-Cap Tech", "Apple"),
    "TSM":   ("Semiconductors", "TSMC"),       "CSCO": ("Networking/Infra", "Cisco"),
    "AMD":   ("Semiconductors", "AMD"),        "TXN":  ("Semiconductors", "Texas Instruments"),
    "MCHP":  ("Semiconductors", "Microchip"),  "ASML": ("Semiconductors", "ASML"),
    "MU":    ("Semiconductors", "Micron"),     "META": ("Mega-Cap Tech", "Meta"),
    "AMZN":  ("Mega-Cap Tech", "Amazon"),      "CRM":  ("Enterprise SaaS", "Salesforce"),
    "AVGO":  ("Semiconductors", "Broadcom"),   "IBM":  ("Networking/Infra", "IBM"),
    "ZM":    ("Enterprise SaaS", "Zoom"),      "PLTR": ("Enterprise SaaS", "Palantir"),
    "QCOM":  ("Semiconductors", "Qualcomm"),   "TSLA": ("Mega-Cap Tech", "Tesla"),
    "INTC":  ("Semiconductors", "Intel"),      "LRCX": ("Semiconductors", "Lam Research"),
    "ORCL":  ("Enterprise SaaS", "Oracle"),    "ADBE": ("Enterprise SaaS", "Adobe"),
    "NOW":   ("Enterprise SaaS", "ServiceNow"),"SNPS": ("Semiconductors", "Synopsys"),
    "CDNS":  ("Semiconductors", "Cadence"),    "KLAC": ("Semiconductors", "KLA"),
    "AMAT":  ("Semiconductors", "Applied Materials"),
}

# --------------------------------------------------------------------------- #
#  Financials — news-dense, high-conviction sector                            #
# --------------------------------------------------------------------------- #
_FIN_TICKER_INFO = {
    "JPM":   ("Diversified Banks", "JPMorgan Chase"),   "BAC": ("Diversified Banks", "Bank of America"),
    "WFC":   ("Diversified Banks", "Wells Fargo"),      "C":   ("Diversified Banks", "Citigroup"),
    "USB":   ("Regional Banks", "U.S. Bancorp"),        "PNC": ("Regional Banks", "PNC Financial"),
    "TFC":   ("Regional Banks", "Truist Financial"),    "GS":  ("Investment Banking", "Goldman Sachs"),
    "MS":    ("Investment Banking", "Morgan Stanley"),  "SCHW":("Brokerage", "Charles Schwab"),
    "BLK":   ("Asset Management", "BlackRock"),         "BX":  ("Asset Management", "Blackstone"),
    "KKR":   ("Asset Management", "KKR"),               "BRK-B":("Insurance/Conglomerate", "Berkshire Hathaway"),
    "AIG":   ("Insurance", "American Intl. Group"),     "MET": ("Insurance", "MetLife"),
    "PRU":   ("Insurance", "Prudential Financial"),     "TRV": ("Insurance", "Travelers"),
    "CB":    ("Insurance", "Chubb"),                    "PGR": ("Insurance", "Progressive"),
    "AXP":   ("Consumer Finance", "American Express"),  "COF": ("Consumer Finance", "Capital One"),
    "V":     ("Payments", "Visa"),                      "MA":  ("Payments", "Mastercard"),
    "PYPL":  ("Payments/Fintech", "PayPal"),            "FIS": ("Payments", "Fidelity National Info"),
    "ICE":   ("Exchanges", "Intercontinental Exchange"),"CME": ("Exchanges", "CME Group"),
    "SPGI":  ("Financial Data", "S&P Global"),          "MCO": ("Financial Data", "Moody's"),
}

# --------------------------------------------------------------------------- #
#  Healthcare — second news-dense sector for the contrast study               #
# --------------------------------------------------------------------------- #
_HC_TICKER_INFO = {
    "LLY":   ("Pharmaceuticals", "Eli Lilly"),          "JNJ": ("Pharmaceuticals", "Johnson & Johnson"),
    "MRK":   ("Pharmaceuticals", "Merck"),              "PFE": ("Pharmaceuticals", "Pfizer"),
    "ABBV":  ("Pharmaceuticals", "AbbVie"),             "BMY": ("Pharmaceuticals", "Bristol Myers Squibb"),
    "AMGN":  ("Biotechnology", "Amgen"),                "GILD":("Biotechnology", "Gilead Sciences"),
    "VRTX":  ("Biotechnology", "Vertex"),               "REGN":("Biotechnology", "Regeneron"),
    "BIIB":  ("Biotechnology", "Biogen"),               "MRNA":("Biotechnology", "Moderna"),
    "ABT":   ("Medical Devices", "Abbott"),             "TMO": ("Life Sciences Tools", "Thermo Fisher"),
    "DHR":   ("Life Sciences Tools", "Danaher"),        "A":   ("Life Sciences Tools", "Agilent"),
    "MDT":   ("Medical Devices", "Medtronic"),          "SYK": ("Medical Devices", "Stryker"),
    "BSX":   ("Medical Devices", "Boston Scientific"),  "ISRG":("Medical Devices", "Intuitive Surgical"),
    "EW":    ("Medical Devices", "Edwards Lifesciences"),"ZBH":("Medical Devices", "Zimmer Biomet"),
    "UNH":   ("Managed Care", "UnitedHealth"),          "ELV": ("Managed Care", "Elevance Health"),
    "CI":    ("Managed Care", "Cigna"),                 "HUM": ("Managed Care", "Humana"),
    "CVS":   ("Healthcare Services", "CVS Health"),     "MCK": ("Healthcare Distribution", "McKesson"),
    "CAH":   ("Healthcare Distribution", "Cardinal Health"), "HCA": ("Hospitals", "HCA Healthcare"),
}

# --------------------------------------------------------------------------- #
#  Sector registry                                                            #
# --------------------------------------------------------------------------- #
#   keyword       -> Stage-One prompt sector keyword ("identify companies in the {keyword} sector")
#   ticker_info   -> {ticker: (sub_sector, company_name)}  (unfiltered universe + attribution)
#   benchmark     -> ticker used by _calculate_benchmark_comparison (default SPY = "beat the market")
#   sector_etf    -> optional sector ETF, for a "beat the sector" comparison if desired later
#   news_file     -> dated, pre-window news CSV for this sector (workspace-relative)
#   news_dir      -> optional folder of per-sector news CSVs (workspace-relative)
#   isolate_news  -> if True, the loader uses ONLY this sector's news file/dir (no workspace glob)
SECTORS = {
    "technology": {
        "keyword":      "technology",
        "ticker_info":  _TECH_TICKER_INFO,
        "benchmark":    "SPY",
        "sector_etf":   "XLK",
        # The LLM relevance screen's output — the curated, high-quality input set the
        # manuscript describes ("screened articles are stored in a curated database and
        # reused downstream"). 110 articles retained of 1,382 collected (8.0% pass rate),
        # validated for NHSJS Rec #5 at 100% precision (95% Wilson CI 94.8-100%).
        "news_file":    "news_collection_archive/gpt_filtered/"
                        "gpt_filtered_investment_news_20251115_042932.csv",
        "news_dir":     None,
        # MUST stay True. With isolate_news=False the loader walks the whole workspace
        # and ingests every CSV whose name contains news/article/finance/consolidated —
        # 34 files and ~33k rows here, including the unscreened corpora the screen was
        # meant to exclude. That silently defeats the screen.
        "isolate_news": True,
    },
    "financials": {
        "keyword":      "financials",
        "ticker_info":  _FIN_TICKER_INFO,
        "benchmark":    "SPY",
        "sector_etf":   "XLF",
        "news_file":    "sectors/financials/financials_news.csv",
        "news_dir":     "sectors/financials/news",
        "isolate_news": True,
    },
    "healthcare": {
        "keyword":      "healthcare",
        "ticker_info":  _HC_TICKER_INFO,
        "benchmark":    "SPY",
        "sector_etf":   "XLV",
        "news_file":    "sectors/healthcare/healthcare_news.csv",
        "news_dir":     "sectors/healthcare/news",
        "isolate_news": True,
    },
}

DEFAULT_SECTOR = "technology"


def active_sector() -> str:
    """Active sector from the SECTOR env var (default technology). Unknown -> default."""
    s = os.environ.get("SECTOR", DEFAULT_SECTOR).strip().lower()
    return s if s in SECTORS else DEFAULT_SECTOR


def _cfg() -> dict:
    return SECTORS[active_sector()]


def keyword() -> str:
    return _cfg()["keyword"]


def ticker_info() -> dict:
    """{ticker: (sub_sector, company_name)} for the active sector."""
    return dict(_cfg()["ticker_info"])


def universe() -> list:
    """Unfiltered ticker universe (list) for the active sector."""
    return list(_cfg()["ticker_info"].keys())


def benchmark() -> str:
    return _cfg().get("benchmark", "SPY")


def sector_etf() -> str:
    return _cfg().get("sector_etf", "")


def isolate_news() -> bool:
    return bool(_cfg().get("isolate_news", False))


def news_file(root: str) -> str:
    """Absolute path to the active sector's primary news CSV (may not exist yet)."""
    return os.path.join(root, _cfg()["news_file"]) if _cfg().get("news_file") else None


def news_dir(root: str):
    nd = _cfg().get("news_dir")
    return os.path.join(root, nd) if nd else None


if __name__ == "__main__":
    print(f"Active sector: {active_sector()}")
    for name, c in SECTORS.items():
        print(f"  {name:11s} | {len(c['ticker_info']):2d} tickers | benchmark {c['benchmark']} "
              f"| isolate_news={c['isolate_news']} | news={c['news_file']}")
