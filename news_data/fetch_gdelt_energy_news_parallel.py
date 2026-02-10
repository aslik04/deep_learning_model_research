#!/usr/bin/env python3
"""
fetch_gdelt_energy_news_parallel.py

Downloads ~700,000 energy-related article URLs from GDELT (2015-2026) via
BigQuery, scrapes their full text in parallel using newspaper3k, and saves
the result to a Parquet file.

Designed to run on a University HPC cluster (Slurm) with high-core nodes.
"""

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# ──────────────────────────── Configuration ────────────────────────────────
SERVICE_ACCOUNT_KEY = "cs310-news-data-scraping-5826b617ad1f.json"
RAW_URL_CACHE       = "gdelt_raw_urls.parquet"
OUTPUT_FILE         = "energy_news_signal_2015_2026.parquet"
MAX_WORKERS         = 100
MIN_TEXT_LENGTH      = 300
LOG_EVERY            = 1_000

# ──────────────────────────── Logging ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────── SQL Query ────────────────────────────────────
GDELT_QUERY = r"""
SELECT
  PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS publish_time,
  DocumentIdentifier AS url,
  ANY_VALUE(SourceCommonName) AS source,
  ANY_VALUE(V2Tone) AS tone_stats,

  -- === DEMAND SIDE ===
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'TECHNOLOGY|COMPUTER_PROCESSOR') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'data-center|hyperscale|server-farm') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'build|construct|invest|campus|gigawatt'), 1, 0)) AS D1_DataCenters,
  MAX(IF(((REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'\b(nvidia|h100|mi300|gpu|ai-chip|tensor-core)\b') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'\b(shortage|supply|demand|backlog|crunch)\b')) OR (REGEXP_CONTAINS(LOWER(AllNames), r'nvidia|tsmc|advanced micro devices') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'\b(semiconductor|wafer|foundry|chip-plant)\b'))) AND NOT REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'stock|share-price|rating|market-close|investor|dividend'), 1, 0)) AS D2_Compute_Hw,
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENERGY_INFRASTRUCTURE_ERROR|CRISISLEX') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'power-grid|electric-grid|utility-grid|ercot|pjm|caiso|miso|nyiso|iso-ne') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'alert|emergency|outage|load-shed|blackout'), 1, 0)) AS D3_Grid_Alerts,
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'WB_2097_EXTREME_COLD|WB_2096_EXTREME_HEAT|NATURAL_DISASTER_HURRICANE') AND (REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'heatwave|polar-vortex|freeze-off|hurricane|bomb-cyclone|winter-storm') OR (REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'record') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'demand|load|consumption|peak-usage'))) AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'power|grid|utility|outage|electricity|fuel|energy|gas|pipeline|refinery') AND REGEXP_CONTAINS(V2Locations, r'US#'), 1, 0)) AS D4_Weather_Extremes,

  -- === INFRASTRUCTURE ===
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENV_NATGAS|ENV_NATURALGAS|ENERGY_LIGHTING') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'power-plant|combined-cycle|ccgt|gas-fired') AND (REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'new|project|approve|propose|construct|build|\bplan\b') OR REGEXP_CONTAINS(LOWER(AllNames), r'duke energy|dominion|nextera|southern company')) AND NOT REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'nuclear|solar|wind|hydro'), 1, 0)) AS G1_Gas_Plant_Adds,
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENV_COAL|ENV_NUCLEARPOWER') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'retire|close|decommission|shutdown|shutter'), 1, 0)) AS G2_Retirements,
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENV_NATGAS|ENV_NATURALGAS|WB_2299_PIPELINES|WB_1768_OIL_AND_GAS_PIPELINE') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'pipeline|gas-transmission|midstream') AND (REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'expand|project|permit|open') OR REGEXP_CONTAINS(LOWER(AllNames), r'kinder morgan|williams|enbridge|tc energy|energy transfer')) AND NOT REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'water|sewage|leak|fire|carbon|co2|capture|irrigation|drainage|teacher'), 1, 0)) AS S1_Pipeline_Cap,
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENV_NATGAS|ENV_NATURALGAS') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'lng|liquefied|export-terminal') AND (REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'export|cargo|shipment|fid') OR REGEXP_CONTAINS(LOWER(AllNames), r'cheniere|freeport|venture global|sempra')) AND NOT REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'holdings|etf|yield'), 1, 0)) AS S2_LNG_Infra,
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENV_OIL|ENV_NATGAS|ENV_NATURALGAS') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'permian|marcellus|haynesville|shale|fracking|rig-count|drilling') AND NOT REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'market-report|outlook-20'), 1, 0)) AS S3_Upstream,
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENV_NATGAS|ENV_NATURALGAS|CRISISLEX') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'freeze-off|force-majeure|leak|explosion|rupture|fire') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'pipeline|midstream|processing-plant|lng|terminal|refinery') AND NOT REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'water|sewage|soy|food|market-report'), 1, 0)) AS S4_Supply_Shock,

  -- === REGIME ===
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ENV_POLICY|ENV_CLIMATECHANGE|ENV_CARBONCAPTURE') AND (REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'ferc|doe|epa|white-house') OR REGEXP_CONTAINS(LOWER(AllNames), r'ferc|department of energy|environmental protection agency')) AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'rule|regulation|permit|veto|mandate|order|limit|standard|approval|restriction'), 1, 0)) AS R1_Policy,
  MAX(IF(REGEXP_CONTAINS(V2Themes, r'ECON_STOCKMARKET') AND (REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'capacity-market|pjm|pricing-rule|tariff|rate-case') OR (REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'auction') AND REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'grid|power|electricity|capacity'))), 1, 0)) AS R2_Market_Rules

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


# ──────────────────────────── BigQuery ─────────────────────────────────────
def fetch_urls(key_path: str) -> pd.DataFrame:
    """Query GDELT via BigQuery or load cached Parquet."""
    if os.path.exists(RAW_URL_CACHE):
        log.info("Found cached URL file  ➜  loading %s", RAW_URL_CACHE)
        df = pd.read_parquet(RAW_URL_CACHE)
        log.info("Loaded %s rows from cache.", f"{len(df):,}")
        return df

    log.info("No cache found  ➜  querying BigQuery (this may take a few minutes)...")
    credentials = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/bigquery"],
    )
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    job_config = bigquery.QueryJobConfig(use_legacy_sql=False)
    query_job = client.query(GDELT_QUERY, job_config=job_config)
    df = query_job.to_dataframe()

    log.info("BigQuery returned %s rows.", f"{len(df):,}")
    df.to_parquet(RAW_URL_CACHE, index=False)
    log.info("Saved raw URLs to %s", RAW_URL_CACHE)
    return df


# ──────────────────────────── Scraping ─────────────────────────────────────
def scrape_article(url: str) -> str | None:
    """
    Download and parse a single article using newspaper3k.
    Returns the article text, or None if the text is too short / unreachable.
    """
    try:
        from newspaper import Article

        article = Article(url)
        article.download()
        article.parse()
        text = (article.text or "").strip()
        if len(text) < MIN_TEXT_LENGTH:
            return None
        return text
    except Exception:
        return None


def scrape_all(urls: pd.Series) -> list[str | None]:
    """Scrape a list of URLs in parallel and return texts in order."""
    total = len(urls)
    results: list[str | None] = [None] * total
    done = 0

    log.info("Starting parallel scrape  ➜  %s URLs with %s workers", f"{total:,}", MAX_WORKERS)
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(scrape_article, url): idx
            for idx, url in enumerate(urls)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            done += 1
            if done % LOG_EVERY == 0 or done == total:
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed > 0 else 0
                log.info(
                    "Scraped %s / %s  (%.1f art/s, elapsed %s)",
                    f"{done:,}",
                    f"{total:,}",
                    rate,
                    time.strftime("%H:%M:%S", time.gmtime(elapsed)),
                )

    elapsed = time.perf_counter() - t0
    success = sum(1 for t in results if t is not None)
    log.info(
        "Scraping complete  ➜  %s / %s succeeded in %s",
        f"{success:,}",
        f"{total:,}",
        time.strftime("%H:%M:%S", time.gmtime(elapsed)),
    )
    return results


# ──────────────────────────── Main ─────────────────────────────────────────
def main() -> None:
    log.info("=" * 60)
    log.info("GDELT Energy News Parallel Scraper")
    log.info("=" * 60)

    # 1. Fetch / load URLs
    df = fetch_urls(SERVICE_ACCOUNT_KEY)

    # 2. Scrape article texts
    df["text"] = scrape_all(df["url"])

    # 3. Save full output (including rows with None text for traceability)
    df.to_parquet(OUTPUT_FILE, index=False)
    log.info("Saved final dataset to %s  (%s rows)", OUTPUT_FILE, f"{len(df):,}")

    # 4. Quick summary
    has_text = df["text"].notna().sum()
    log.info(
        "Articles with text: %s / %s  (%.1f%%)",
        f"{has_text:,}",
        f"{len(df):,}",
        100 * has_text / len(df) if len(df) > 0 else 0,
    )
    log.info("Done ✓")


if __name__ == "__main__":
    main()
