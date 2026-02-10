"""
aggregate_signals_enhanced.py

Transforms per-article FinBERT sentiment scores into a daily time-series
matrix using Max Pooling + Dispersion aggregation logic.

Input:  energy_news_with_sentiment.parquet  (~148k rows)
Output: daily_energy_signals_enhanced.parquet (daily x 91 feature columns)
"""

import pathlib
import pandas as pd
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path(__file__).parent
INPUT_FILE = DATA_DIR / "energy_news_with_sentiment.parquet"
OUTPUT_FILE = DATA_DIR / "daily_energy_signals_enhanced.parquet"

BUCKETS = [
    "D1_DataCenters", "D2_Compute_Hw", "D3_Grid_Alerts", "D4_Weather_Extremes",
    "G1_Gas_Plant_Adds", "G2_Retirements",
    "S1_Pipeline_Cap", "S2_LNG_Infra", "S3_Upstream", "S4_Supply_Shock",
    "R1_Policy", "R2_Market_Rules",
]

SENT_COLS = ["finbert_positive", "finbert_negative", "finbert_neutral"]

METRICS = {
    "Vol":        ("finbert_positive", "count"),
    "Avg_Pos":    ("finbert_positive", "mean"),
    "Avg_Neg":    ("finbert_negative", "mean"),
    "Avg_Neu":    ("finbert_neutral",  "mean"),
    "Max_Pos":    ("finbert_positive", "max"),
    "Max_Neg":    ("finbert_negative", "max"),
    "Dispersion": ("finbert_neutral",  "std"),
}


def load_and_prepare(path: pathlib.Path) -> pd.DataFrame:
    """Load parquet, extract date, filter to rows matching at least one bucket."""
    df = pd.read_parquet(path, columns=["publish_time"] + BUCKETS + SENT_COLS)
    df["date"] = pd.to_datetime(df["publish_time"]).dt.date
    # Keep only rows that match at least one bucket
    mask = df[BUCKETS].any(axis=1)
    df = df.loc[mask].drop(columns=["publish_time"])
    print(f"Loaded {len(df):,} rows matching at least one bucket.")
    return df


def melt_to_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Melt bucket booleans so each (article, bucket) pair where True becomes a row."""
    melted = df.melt(
        id_vars=["date"] + SENT_COLS,
        value_vars=BUCKETS,
        var_name="bucket",
        value_name="active",
    )
    melted = melted.loc[melted["active"] == 1].drop(columns=["active"])
    print(f"Melted to {len(melted):,} article-bucket pairs.")
    return melted


def aggregate(melted: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Compute the 7 metrics grouped by the given columns."""
    agg_dict = {
        "finbert_positive": ["count", "mean", "max"],
        "finbert_negative": ["mean", "max"],
        "finbert_neutral":  ["mean", "std"],
    }
    agg = melted.groupby(group_cols).agg(agg_dict)

    # Flatten multi-index columns into metric suffixes
    records = {}
    for metric_name, (col, func) in METRICS.items():
        records[metric_name] = agg[(col, func)]

    result = pd.DataFrame(records, index=agg.index)
    # Single-article days produce NaN std → fill with 0
    result["Dispersion"] = result["Dispersion"].fillna(0.0)
    return result


def build_daily_matrix(df: pd.DataFrame, melted: pd.DataFrame) -> pd.DataFrame:
    """Build the full daily matrix with per-bucket + global columns."""
    # ── Per-bucket aggregation ──
    bucket_agg = aggregate(melted, ["date", "bucket"])
    # Pivot: rows=date, columns = (bucket, metric)
    pivoted = bucket_agg.unstack(level="bucket")
    # Flatten columns: metric x bucket → bucket_metric
    pivoted.columns = [f"{bucket}_{metric}" for metric, bucket in pivoted.columns]
    pivoted = pivoted.sort_index(axis=1)

    # ── Global aggregation ──
    global_agg = aggregate(melted, ["date"])
    global_agg.columns = [f"Global_{c}" for c in global_agg.columns]

    # ── Merge ──
    combined = pivoted.join(global_agg, how="outer")

    # ── Continuous daily index ──
    date_range = pd.date_range(
        start=min(df["date"]),
        end=max(df["date"]),
        freq="D",
    )
    combined.index = pd.to_datetime(combined.index)
    combined = combined.reindex(date_range).fillna(0.0)
    combined.index.name = "date"

    # Ensure Vol columns are int
    vol_cols = [c for c in combined.columns if c.endswith("_Vol")]
    combined[vol_cols] = combined[vol_cols].astype(int)

    return combined


def main():
    print("=" * 60)
    print("Daily Signal Aggregation (Max Pooling + Dispersion)")
    print("=" * 60)

    df = load_and_prepare(INPUT_FILE)
    melted = melt_to_pairs(df)
    daily = build_daily_matrix(df, melted)

    daily.to_parquet(OUTPUT_FILE)

    # ── Summary ──
    print(f"\nOutput: {OUTPUT_FILE.name}")
    print(f"Shape:  {daily.shape[0]} days x {daily.shape[1]} columns")
    print(f"Range:  {daily.index.min().date()} → {daily.index.max().date()}")
    print(f"NaN remaining: {daily.isna().sum().sum()}")
    print(f"\nColumn groups ({len(BUCKETS)} buckets + Global):")
    for bucket in BUCKETS + ["Global"]:
        cols = [c for c in daily.columns if c.startswith(bucket + "_")]
        print(f"  {bucket}: {len(cols)} metrics")
    print(f"\nTotal columns: {daily.shape[1]}")
    print("Done.")


if __name__ == "__main__":
    main()
