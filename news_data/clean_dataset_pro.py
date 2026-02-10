"""
clean_dataset_pro.py
====================
Advanced cleaning pipeline for GDELT Energy News.
Implements "Hedge Fund Grade" filters:
1. Drop rows with no scraped text.
2. HTML artifact removal & whitespace normalization.
3. Paywall / junk marker filtering.
4. Sentence count & length quality gates.
5. Language detection (English only).
6. Near-duplicate removal (exact text + first-N-chars dedup for syndication).

Usage:
    python clean_dataset_pro.py                  # full run
    python clean_dataset_pro.py --sample 1000    # test on 1000 rows
"""

import argparse
import re
import sys
import logging
import time

import pandas as pd
import numpy as np
from langdetect import detect, LangDetectException

# ─── Configuration ──────────────────────────────────────────────────────────
INPUT_FILE  = "energy_news_signal_2015_2026.parquet"
OUTPUT_FILE = "energy_news_cleaned_pro.parquet"

MIN_CHAR_LENGTH    = 300   # after cleaning
MIN_SENTENCE_COUNT = 3
DEDUP_PREFIX_LEN   = 500   # first N chars used for syndication dedup

# Markers that indicate junk / paywall / error pages
# Checked against first 400 chars (lowercased)
JUNK_MARKERS_HEADER = [
    "javascript is disabled", "enable cookies", "403 forbidden", "404 not found",
    "access denied", "please subscribe", "robot check", "verify you are human",
    "subscribe to continue", "register to continue", "sign in to read",
    "you have reached your limit", "this content is for subscribers",
    "please enable javascript", "cookies must be enabled",
    "we noticed you're using an adblocker", "turn off your ad blocker",
    "page not found", "sorry, this page",
    "too many requests", "http error code", "rate limit",
    "your browser does not support", "let us read it for you",
    "accept cookies to continue", "we use cookies",
    "continue reading", "log in to read", "create a free account",
    "this article is only available", "you need to be a subscriber",
    "unauthorized", "error 5", "502 bad gateway", "503 service",
]

HTML_TAG_RE    = re.compile(r'<[^>]+>')
MULTI_SPACE_RE = re.compile(r'[ \t]+')
MULTI_NL_RE    = re.compile(r'\n{3,}')

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)

# ─── Helpers ────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Strip HTML tags, collapse whitespace, trim."""
    text = HTML_TAG_RE.sub(' ', text)
    text = MULTI_SPACE_RE.sub(' ', text)
    text = MULTI_NL_RE.sub('\n\n', text)
    return text.strip()


def is_junk(text: str) -> bool:
    """True if the text looks like a paywall, error page, or stub."""
    if len(text) < MIN_CHAR_LENGTH:
        return True
    # Count sentences (period followed by space/newline)
    sentences = text.count('. ') + text.count('.\n') + text.count('."')
    if sentences < MIN_SENTENCE_COUNT:
        return True
    # Check first 400 chars for junk markers
    header = text[:400].lower()
    for marker in JUNK_MARKERS_HEADER:
        if marker in header:
            return True
    return False


def is_english(text: str) -> bool:
    """Detect language on the first 500 chars (fast & reliable enough)."""
    try:
        return detect(text[:500]) == 'en'
    except LangDetectException:
        return False


# ─── Main pipeline ──────────────────────────────────────────────────────────

def clean_data(sample_n: int | None = None) -> None:
    t0 = time.perf_counter()

    # 1. Load
    log.info("Loading %s ...", INPUT_FILE)
    df = pd.read_parquet(INPUT_FILE)
    total = len(df)
    log.info("Loaded %s rows  (%s with text)", f"{total:,}", f"{df['text'].notna().sum():,}")

    if sample_n:
        df = df[df['text'].notna()].sample(n=min(sample_n, df['text'].notna().sum()), random_state=42)
        log.info("*** SAMPLE MODE: working on %s rows ***", f"{len(df):,}")

    # 2. Drop nulls / empty
    before = len(df)
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != ""]
    log.info("Step 1 — Drop nulls/empty:      %s → %s  (-%s)", f"{before:,}", f"{len(df):,}", f"{before - len(df):,}")

    # 3. Clean HTML & whitespace
    log.info("Step 2 — Cleaning HTML & whitespace ...")
    df['text'] = df['text'].apply(clean_text)

    # 4. Quality gates (junk / paywall / too short / too few sentences)
    before = len(df)
    junk_mask = df['text'].apply(is_junk)
    n_junk = junk_mask.sum()
    df = df[~junk_mask]
    log.info("Step 3 — Quality gates:          %s → %s  (-%s junk/stub/paywall)", f"{before:,}", f"{len(df):,}", f"{n_junk:,}")

    # 5. Language filter
    before = len(df)
    log.info("Step 4 — Language detection (this is the slow step) ...")
    en_mask = df['text'].apply(is_english)
    n_non_en = (~en_mask).sum()
    df = df[en_mask]
    log.info("Step 4 — Language filter:         %s → %s  (-%s non-English)", f"{before:,}", f"{len(df):,}", f"{n_non_en:,}")

    # 6. Deduplication
    before = len(df)
    # A. Exact text duplicates
    df = df.drop_duplicates(subset=['text'], keep='first')
    after_exact = len(df)
    # B. Syndication dedup: same first N chars = same wire story
    df['_prefix'] = df['text'].str[:DEDUP_PREFIX_LEN]
    df = df.drop_duplicates(subset=['_prefix'], keep='first')
    df = df.drop(columns=['_prefix'])
    log.info("Step 5 — Deduplication:          %s → %s  (-%s exact, -%s syndication)",
             f"{before:,}", f"{len(df):,}", f"{before - after_exact:,}", f"{after_exact - len(df):,}")

    # 7. Summary
    elapsed = time.perf_counter() - t0
    log.info("=" * 60)
    log.info("CLEANING COMPLETE")
    log.info("=" * 60)
    log.info("  Input:     %s rows (%s with text)", f"{total:,}", f"{'sample ' + str(sample_n) if sample_n else 'all'}")
    log.info("  Output:    %s rows", f"{len(df):,}")
    log.info("  Retention: %.1f%% of text rows", 100 * len(df) / max(1, df['text'].notna().sum() if not sample_n else (sample_n)))
    log.info("  Time:      %.1fs", elapsed)

    # Show some stats
    log.info("  Avg text length: %s chars", f"{df['text'].str.len().mean():,.0f}")
    log.info("  Median text len: %s chars", f"{df['text'].str.len().median():,.0f}")

    # 8. Save
    if sample_n:
        out = OUTPUT_FILE.replace('.parquet', '_sample.parquet')
        df.to_parquet(out, index=False)
        log.info("  Saved SAMPLE to %s", out)
    else:
        df.to_parquet(OUTPUT_FILE, index=False)
        log.info("  Saved to %s", OUTPUT_FILE)


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Test on N rows instead of full dataset")
    args = parser.parse_args()
    clean_data(sample_n=args.sample)
