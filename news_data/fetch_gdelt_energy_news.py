"""
fetch_gdelt_energy_news.py
==========================
Production pipeline to:
  1. Extract high-signal US energy news from GDELT via BigQuery.
  2. Scrape full article text with newspaper3k.
  3. Apply minimal quality filters (length only).
  4. Serialise to Parquet.

Usage:
    python fetch_gdelt_energy_news.py
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from newspaper import Article, Config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID: str = "cs310-news-data-scraping"
SERVICE_ACCOUNT_KEY: Path = Path(__file__).resolve().parent / "cs310-news-data-scraping-f2278ab04a08.json"
OUTPUT_PATH: Path = Path(__file__).resolve().parent / "energy_news_signal_2015_2024.parquet"
MIN_ARTICLE_LENGTH: int = 300                 # characters
SCRAPE_TIMEOUT: int = 10                      # seconds per request
LOG_EVERY: int = 100                          # progress interval
LOG_FILE: Path = Path(__file__).resolve().parent / "pipeline.log"

# ---------------------------------------------------------------------------
# Logging  (console + persistent file)
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

log = logging.getLogger("gdelt_pipeline")
log.setLevel(logging.INFO)

# Console handler  (flush every line)
_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
log.addHandler(_console)

# File handler – force flush every line so `tail -f pipeline.log` works under nohup

class _FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes after every emit."""
    def emit(self, record):
        super().emit(record)
        self.flush()

_fh = _FlushFileHandler(LOG_FILE, mode="a", encoding="utf-8")
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
log.addHandler(_fh)

# ---------------------------------------------------------------------------
# BigQuery SQL
# ---------------------------------------------------------------------------
GDELT_QUERY: str = r"""
SELECT
  PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS publish_time,
  DocumentIdentifier AS url,
  ANY_VALUE(SourceCommonName) AS source,
  ANY_VALUE(V2Tone) AS tone_stats,

  -- === DEMAND SIDE ===

  -- D1: AI Data Centers
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'TECHNOLOGY|COMPUTER_PROCESSOR')
     AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'data-center|hyperscale|server-farm')
     AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'build|construct|invest|campus|gigawatt'), 1, 0)) AS D1_DataCenters,

  -- D2: AI Chips (STRICT: Boundaries \b)
  MAX(IF(
      (
        (REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'\b(nvidia|h100|mi300|gpu|ai-chip|tensor-core)\b')
         AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'\b(shortage|supply|demand|backlog|crunch)\b'))
        OR
        (REGEXP_CONTAINS(LOWER(AllNames), r'nvidia|tsmc|advanced micro devices')
         AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'\b(semiconductor|wafer|foundry|chip-plant)\b'))
      )
      AND NOT REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'stock|share-price|rating|market-close|investor|dividend'), 1, 0)) AS D2_Compute_Hw,

  -- D3: Grid Alerts
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENERGY_INFRASTRUCTURE_ERROR|CRISISLEX')
     AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'power-grid|electric-grid|utility-grid|ercot|pjm|caiso|miso|nyiso|iso-ne')
     AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'alert|emergency|outage|load-shed|blackout'), 1, 0)) AS D3_Grid_Alerts,

  -- D4: Weather Extremes (STRICT: Energy Context)
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'WB_2097_EXTREME_COLD|WB_2096_EXTREME_HEAT|NATURAL_DISASTER_HURRICANE')
     AND (
       REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'heatwave|polar-vortex|freeze-off|hurricane|bomb-cyclone|winter-storm')
       OR
       (REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'record') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'demand|load|consumption|peak-usage'))
     )
     -- MUST be Energy Related
     AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'power|grid|utility|outage|electricity|fuel|energy|gas|pipeline|refinery')
     AND REGEXP_CONTAINS(V2Locations, r'US#'), 1, 0)) AS D4_Weather_Extremes,

  -- === INFRASTRUCTURE ===

  -- G1: Gas Plant Adds
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENV_NATGAS|ENV_NATURALGAS|ENERGY_LIGHTING')
     AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'power-plant|combined-cycle|ccgt|gas-fired')
     AND (REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'new|project|approve|propose|construct|build|\bplan\b')
          OR REGEXP_CONTAINS(LOWER(AllNames), r'duke energy|dominion|nextera|southern company'))
     AND NOT REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'nuclear|solar|wind|hydro'), 1, 0)) AS G1_Gas_Plant_Adds,

  -- G2: Retirements
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENV_COAL|ENV_NUCLEARPOWER')
     AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'retire|close|decommission|shutdown|shutter'), 1, 0)) AS G2_Retirements,

  -- S1: Pipeline Capacity (STRICT: No Water/Irrigation)
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENV_NATGAS|ENV_NATURALGAS|WB_2299_PIPELINES|WB_1768_OIL_AND_GAS_PIPELINE')
     AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'pipeline|gas-transmission|midstream')
     AND (REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'expand|project|permit|open') OR REGEXP_CONTAINS(LOWER(AllNames), r'kinder morgan|williams|enbridge|tc energy|energy transfer'))
     AND NOT REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'water|sewage|leak|fire|carbon|co2|capture|irrigation|drainage|teacher'), 1, 0)) AS S1_Pipeline_Cap,

  -- S2: LNG Export
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENV_NATGAS|ENV_NATURALGAS')
     AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'lng|liquefied|export-terminal')
     AND (REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'export|cargo|shipment|fid') OR REGEXP_CONTAINS(LOWER(AllNames), r'cheniere|freeport|venture global|sempra'))
     AND NOT REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'holdings|etf|yield'), 1, 0)) AS S2_LNG_Infra,

  -- S3: Upstream Production
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENV_OIL|ENV_NATGAS|ENV_NATURALGAS')
     AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'permian|marcellus|haynesville|shale|fracking|rig-count|drilling')
     AND NOT REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'market-report|outlook-20'), 1, 0)) AS S3_Upstream,

  -- S4: Supply Shock
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENV_NATGAS|ENV_NATURALGAS|CRISISLEX')
     AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'freeze-off|force-majeure|leak|explosion|rupture|fire')
     AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'pipeline|midstream|processing-plant|lng|terminal|refinery')
     AND NOT REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'water|sewage|soy|food|market-report'), 1, 0)) AS S4_Supply_Shock,

  -- === REGIME ===

  -- R1: Policy
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENV_POLICY|ENV_CLIMATECHANGE|ENV_CARBONCAPTURE')
     AND (REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'ferc|doe|epa|white-house') OR REGEXP_CONTAINS(LOWER(AllNames), r'ferc|department of energy|environmental protection agency'))
     AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'rule|regulation|permit|veto|mandate|order|limit|standard|approval|restriction'), 1, 0)) AS R1_Policy,

  -- R2: Market Rules
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ECON_STOCKMARKET')
     AND (
       REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'capacity-market|pjm|pricing-rule|tariff|rate-case')
       OR
       (REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'auction') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'grid|power|electricity|capacity'))
     ), 1, 0)) AS R2_Market_Rules

FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE
  _PARTITIONDATE BETWEEN '2015-02-19' AND '2026-12-31'
  AND (REGEXP_CONTAINS(V2Locations, r'US#|CA#|MX#'))

GROUP BY 1, 2
HAVING
  (D1_DataCenters + D2_Compute_Hw + D3_Grid_Alerts + D4_Weather_Extremes +
   G1_Gas_Plant_Adds + G2_Retirements +
   S1_Pipeline_Cap + S2_LNG_Infra + S3_Upstream + S4_Supply_Shock +
   R1_Policy + R2_Market_Rules) > 0
"""

# ---------------------------------------------------------------------------
# Helper: BigQuery fetch
# ---------------------------------------------------------------------------

def fetch_from_bigquery(project_id: str) -> pd.DataFrame:
    """Execute the GDELT query and return the result as a DataFrame."""
    log.info("Initialising BigQuery client (project=%s) ...", project_id)
    credentials = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_KEY),
        scopes=["https://www.googleapis.com/auth/bigquery"],
    )
    client = bigquery.Client(project=project_id, credentials=credentials)

    log.info("Submitting query – this may take several minutes on 300 M+ rows ...")
    t0 = time.perf_counter()
    query_job = client.query(GDELT_QUERY)
    df = query_job.to_dataframe()
    elapsed = time.perf_counter() - t0

    log.info(
        "BigQuery returned %s rows  (%s columns)  in %.1f s",
        f"{len(df):,}",
        df.shape[1],
        elapsed,
    )
    return df


# ---------------------------------------------------------------------------
# Helper: article scraping
# ---------------------------------------------------------------------------

def _build_newspaper_config() -> Config:
    """Return a newspaper3k Config with a browser-like UA and timeout."""
    cfg = Config()
    cfg.browser_user_agent = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    cfg.request_timeout = SCRAPE_TIMEOUT
    cfg.fetch_images = False          # we only need text
    cfg.memoize_articles = False      # no on-disk cache
    return cfg


def scrape_article_text(url: str, config: Config) -> str | None:
    """
    Download and parse a single article.

    Returns the extracted full text, or None on any failure.
    """
    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
        text = article.text
        if text and text.strip():
            return text.strip()
        return None
    except Exception:
        return None


def scrape_all_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an `article_text` column by scraping every URL in *df*.

    Logs progress every LOG_EVERY rows with running success / failure counts.
    """
    config = _build_newspaper_config()
    total = len(df)
    texts: list[str | None] = []
    successes = 0
    failures = 0

    log.info("Beginning scrape of %s URLs ...", f"{total:,}")
    t0 = time.perf_counter()

    for idx, url in enumerate(df["url"], start=1):
        text = scrape_article_text(url, config)
        texts.append(text)

        if text is not None:
            successes += 1
        else:
            failures += 1

        if idx % LOG_EVERY == 0 or idx == total:
            elapsed = time.perf_counter() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            log.info(
                "Processed %s / %s  (✓ %s  ✗ %s)  %.1f art/s",
                f"{idx:,}",
                f"{total:,}",
                f"{successes:,}",
                f"{failures:,}",
                rate,
            )

    df = df.copy()
    df["article_text"] = texts
    return df


# ---------------------------------------------------------------------------
# Helper: minimal cleaning
# ---------------------------------------------------------------------------

def clean_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where scraping failed or text is too short.

    - Remove rows where article_text is None / empty / "ERROR".
    - Remove rows shorter than MIN_ARTICLE_LENGTH characters.
    """
    before = len(df)

    # Drop missing / error text
    df = df[df["article_text"].notna()].copy()
    df = df[~df["article_text"].isin(["", "ERROR"])].copy()

    # Drop short articles
    df = df[df["article_text"].str.len() >= MIN_ARTICLE_LENGTH].copy()

    after = len(df)
    log.info(
        "Filtering: %s → %s rows  (dropped %s)",
        f"{before:,}",
        f"{after:,}",
        f"{before - after:,}",
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("GDELT Energy-News Pipeline  –  start")
    log.info("=" * 60)

    # 1. BigQuery extraction
    df = fetch_from_bigquery(PROJECT_ID)

    # 2. Scrape article text
    df = scrape_all_articles(df)

    # 3. Minimal cleaning
    df = clean_and_filter(df)

    # 4. Serialise to Parquet
    df.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
    log.info("Saved %s rows → %s", f"{len(df):,}", OUTPUT_PATH)

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
