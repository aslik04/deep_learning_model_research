"""
apply_finbert_gpu.py
====================
Production FinBERT inference on GDELT Energy News.

Features:
- Automatic OOM fallback (batch 64 → 32 → 16 → 8).
- Incremental checkpointing (resumes if job killed).
- Head truncation (512 tokens).
- Softmax probability output (pos/neg/neu).
- Built-in test mode (--sample N).

Usage:
    python apply_finbert_gpu.py --sample 50    # test on 50 rows
    python apply_finbert_gpu.py                # full run
"""

import argparse
import hashlib
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ─── Configuration ──────────────────────────────────────────────────────────
INPUT_FILE       = "energy_news_cleaned_pro.parquet"
OUTPUT_FILE      = "energy_news_with_sentiment.parquet"
CHECKPOINT_FILE  = "sentiment_checkpoint.parquet"
MODEL_NAME       = "ProsusAI/finbert"
BATCH_SIZE       = 64
MAX_LEN          = 512
SAVE_EVERY       = 10_000   # checkpoint every N rows
NUM_WORKERS      = 2

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)

# ─── Dataset ────────────────────────────────────────────────────────────────

class NewsTextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_len: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
        }


# ─── Helpers ────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info("GPU: %s (%.1f GB VRAM)", name, vram)
        return dev
    log.warning("No GPU found — running on CPU (will be very slow).")
    return torch.device("cpu")


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def run_batch_with_oom_retry(model, batch, device, current_bs):
    """
    Run a single batch through the model.
    On OOM, retry with halved batch size down to 8.
    Returns (probs_numpy, effective_batch_size).
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy(), current_bs

    except RuntimeError as e:
        if "out of memory" not in str(e):
            raise
        torch.cuda.empty_cache()

        # Split the batch and retry with smaller sub-batches
        new_bs = max(current_bs // 2, 8)
        log.warning("OOM on batch_size=%d → retrying with sub-batches of %d", current_bs, new_bs)

        results = []
        n = input_ids.size(0)
        for start in range(0, n, new_bs):
            end = min(start + new_bs, n)
            sub_ids = input_ids[start:end]
            sub_mask = attention_mask[start:end]
            try:
                out = model(input_ids=sub_ids, attention_mask=sub_mask)
                probs = torch.nn.functional.softmax(out.logits, dim=1)
                results.append(probs.cpu().numpy())
            except RuntimeError:
                torch.cuda.empty_cache()
                # Last resort: process one at a time
                log.warning("OOM again at sub-batch %d → falling back to batch_size=1", new_bs)
                for j in range(start, end):
                    out = model(input_ids=sub_ids[j:j+1], attention_mask=sub_mask[j:j+1])
                    probs = torch.nn.functional.softmax(out.logits, dim=1)
                    results.append(probs.cpu().numpy())

        return np.concatenate(results, axis=0), new_bs


# ─── Main ───────────────────────────────────────────────────────────────────

def main(sample_n: int | None = None) -> None:
    device = get_device()
    t0 = time.perf_counter()

    # 1. Load data
    log.info("Loading %s ...", INPUT_FILE)
    df = pd.read_parquet(INPUT_FILE)
    log.info("Loaded %s rows.", f"{len(df):,}")

    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=42).reset_index(drop=True)
        log.info("*** SAMPLE MODE: %d rows ***", len(df))

    # 2. Check for checkpoint (resume support)
    start_idx = 0
    checkpoint_data = None
    if not sample_n and os.path.exists(CHECKPOINT_FILE):
        checkpoint_data = pd.read_parquet(CHECKPOINT_FILE)
        start_idx = len(checkpoint_data)
        log.info("Resuming from checkpoint at row %s", f"{start_idx:,}")
        if start_idx >= len(df):
            log.info("Already complete — renaming checkpoint to final output.")
            os.rename(CHECKPOINT_FILE, OUTPUT_FILE)
            return

    texts_remaining = df["text"].iloc[start_idx:].tolist()
    total = len(texts_remaining)
    log.info("Processing %s rows (starting from idx %s)...", f"{total:,}", f"{start_idx:,}")

    # 3. Load model + tokenizer
    log.info("Loading FinBERT model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    log.info("Model loaded and set to eval mode.")

    # 4. Create dataloader (deterministic order — no shuffle)
    dataset = NewsTextDataset(texts_remaining, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    # 5. Inference loop
    all_probs = []
    processed = 0
    current_bs = BATCH_SIZE

    log.info("Starting inference (batch_size=%d, max_len=%d)...", BATCH_SIZE, MAX_LEN)

    with torch.no_grad():
        for batch in tqdm(loader, desc="FinBERT", file=sys.stderr):
            probs, current_bs = run_batch_with_oom_retry(model, batch, device, current_bs)
            all_probs.append(probs)
            processed += probs.shape[0]

            # Periodic checkpoint
            if not sample_n and processed % SAVE_EVERY < BATCH_SIZE and processed > 0:
                elapsed = time.perf_counter() - t0
                rate = processed / elapsed
                eta = (total - processed) / rate if rate > 0 else 0
                log.info(
                    "Progress: %s / %s  (%.1f rows/s, ETA %s)",
                    f"{processed:,}", f"{total:,}", rate,
                    time.strftime("%H:%M:%S", time.gmtime(eta)),
                )
                # Save checkpoint
                _save_checkpoint(df, start_idx, all_probs, checkpoint_data)

    # 6. Assemble results
    all_probs_arr = np.concatenate(all_probs, axis=0)
    log.info("Inference complete. Assembling output...")

    result_df = df.iloc[start_idx:].copy()
    result_df["finbert_positive"] = all_probs_arr[:, 0]
    result_df["finbert_negative"] = all_probs_arr[:, 1]
    result_df["finbert_neutral"]  = all_probs_arr[:, 2]
    result_df["text_hash"] = result_df["text"].apply(text_hash)

    # Combine with checkpoint if resuming
    if checkpoint_data is not None:
        final_df = pd.concat([checkpoint_data, result_df], ignore_index=True)
    else:
        final_df = result_df

    # 7. Save
    out_file = OUTPUT_FILE if not sample_n else OUTPUT_FILE.replace(".parquet", "_sample.parquet")
    final_df.to_parquet(out_file, index=False)
    log.info("Saved to %s  (%s rows)", out_file, f"{len(final_df):,}")

    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log.info("Removed checkpoint file.")

    elapsed = time.perf_counter() - t0
    log.info("Total time: %s  (%.1f rows/s)", time.strftime("%H:%M:%S", time.gmtime(elapsed)), len(final_df) / elapsed)

    # Quick summary
    log.info("=" * 60)
    log.info("SENTIMENT SUMMARY")
    log.info("=" * 60)
    for col in ["finbert_positive", "finbert_negative", "finbert_neutral"]:
        log.info("  %s: mean=%.4f, std=%.4f", col, final_df[col].mean(), final_df[col].std())

    dominant = final_df[["finbert_positive", "finbert_negative", "finbert_neutral"]].idxmax(axis=1)
    for label in ["finbert_positive", "finbert_negative", "finbert_neutral"]:
        short = label.replace("finbert_", "")
        count = (dominant == label).sum()
        log.info("  Dominant %s: %s (%.1f%%)", short, f"{count:,}", 100 * count / len(final_df))

    log.info("Done ✓")


def _save_checkpoint(df, start_idx, all_probs, checkpoint_data):
    """Save incremental checkpoint to disk."""
    try:
        arr = np.concatenate(all_probs, axis=0)
        chunk = df.iloc[start_idx : start_idx + arr.shape[0]].copy()
        chunk["finbert_positive"] = arr[:, 0]
        chunk["finbert_negative"] = arr[:, 1]
        chunk["finbert_neutral"]  = arr[:, 2]
        chunk["text_hash"] = chunk["text"].apply(text_hash)

        if checkpoint_data is not None:
            combined = pd.concat([checkpoint_data, chunk], ignore_index=True)
        else:
            combined = chunk

        combined.to_parquet(CHECKPOINT_FILE, index=False)
        log.info("  Checkpoint saved (%s rows total).", f"{len(combined):,}")
    except Exception as e:
        log.warning("  Checkpoint save failed: %s (continuing...)", e)


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Test on N rows")
    args = parser.parse_args()
    main(sample_n=args.sample)
