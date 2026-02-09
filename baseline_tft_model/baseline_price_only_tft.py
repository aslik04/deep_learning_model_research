# %% [markdown]
# ## Price-Only Temporal Fusion Transformer — Baseline (Henry Hub)

# %% [markdown]
# ### Imports and Configuration
# 
# For the implemmentation of TFT, we are using pytorch. Tuning will be done using optuna.

# %%
import os, json, time, pickle, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import torch
import lightning.pytorch as pl
import optuna

from datetime import datetime, timezone

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.importance import get_param_importances
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 1337
pl.seed_everything(SEED, workers=True)
torch.set_float32_matmul_precision("high") 

# ------------------------------------------------------------------
# Environment overrides for quick tests / HPC control
# ------------------------------------------------------------------
ENV_MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "60"))
ENV_PATIENCE = int(os.environ.get("PATIENCE", "8"))
ENV_N_TRIALS = int(os.environ.get("N_TRIALS", "300"))
ENV_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
ENV_SKIP_BASELINE_FIT = os.environ.get("SKIP_BASELINE_FIT", "0") == "1"
ENV_BASELINE_MAX_EPOCHS = int(os.environ.get("BASELINE_MAX_EPOCHS", str(ENV_MAX_EPOCHS)))

TASK_ID = os.environ.get("SLURM_ARRAY_TASK_ID", "local")
EXPERIMENT = os.environ.get("EXP_NAME", "price_only")

# ------------------------------------------------------------------
# Run configuration (matching LSTM pattern)
# ------------------------------------------------------------------
EXPERIMENT_NAME = "Price Only"
MODEL_TAG = "tft"
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

TOP_K = 10
SAVED_RESULTS_DIR = "saved_results"

# run output dir: saved_results/<timestamp>_<experimentname>_tft/
RUN_NAME = f"{RUN_TS}_{EXPERIMENT_NAME.replace(' ', '')}_{MODEL_TAG}"
RUN_DIR = os.path.join(SAVED_RESULTS_DIR, RUN_NAME)
TOPMODELS_DIR = os.path.join(RUN_DIR, "top_models")
TRIAL_SUMMARY_CSV = os.path.join(RUN_DIR, "trial_summary.csv")

# DB inside the run folder (so each run is self-contained)
DB_PATH = os.path.join(RUN_DIR, f"optuna_{EXPERIMENT_NAME.replace(' ', '')}_{MODEL_TAG}.db")

# study name includes timestamp + experiment + model
STUDY_NAME = RUN_NAME

os.makedirs(TOPMODELS_DIR, exist_ok=True)

# %%
# ─────────────────────────────────────────────────────────────
# SLURM / HPC: DataLoader + GPU config
# ─────────────────────────────────────────────────────────────
SLURM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
NUM_WORKERS = max(1, min(4, SLURM_CPUS - 1))  # Reduced to avoid file descriptor exhaustion
PIN_MEMORY = torch.cuda.is_available()

print("=" * 60)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print(f"SLURM_CPUS = {SLURM_CPUS}")
print(f"NUM_WORKERS = {NUM_WORKERS}, PIN_MEMORY = {PIN_MEMORY}")
print(f"ENV_MAX_EPOCHS={ENV_MAX_EPOCHS}, ENV_PATIENCE={ENV_PATIENCE}, ENV_N_TRIALS={ENV_N_TRIALS}")
print(f"ENV_BATCH_SIZE={ENV_BATCH_SIZE}, ENV_SKIP_BASELINE_FIT={ENV_SKIP_BASELINE_FIT}")
print("=" * 60)

import subprocess
print("\n--- nvidia-smi ---")
try:
    print(subprocess.check_output(["nvidia-smi"]).decode("utf-8"))
except Exception as e:
    print("nvidia-smi not available:", e)

# %% [markdown]
# ### Step 1 - Load Master CSV and Transform Data Set
# 
# We load our master csv with all its columns, and transform the data set in memory to follow the format needed for PyTorch TFT.

# %%
CSV = "../numeric_data/henryhub_master.csv"
TARGET = "price"
GROUP_COL = "id"

# numeric features in our data set
NUM_COLS = ["storage_bcf", "production_bcf", "usd_index", "temp_c", "temp_max_c", "temp_min_c"]

def load_tft_ready_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values([GROUP_COL, "date"]).reset_index(drop=True)

    # group id as categorical
    df[GROUP_COL] = df[GROUP_COL].astype("category")

    # target numeric
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype("float32")

    # covariates numeric
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # TFT-required time index (0..N-1 per id)
    df["time_idx"] = df.groupby(GROUP_COL).cumcount().astype(np.int64)

    # known future calendar covariates
    df["dow"] = df["date"].dt.dayofweek.astype("category")
    df["month"] = df["date"].dt.month.astype("category")

    # sanity checks
    assert df[[GROUP_COL, "time_idx"]].duplicated().sum() == 0
    assert df[TARGET].isna().sum() == 0

    return df

data = load_tft_ready_df(CSV)
data.head()

# %% [markdown]
# ### Step 2 — Chronological Train/Validation/Test Split
# 
# We split the dataset into train, validation, and test sets, to avoid leaking future information into training:
# - We first sort by date to guarantee chronological ordering.
# - We use a (70/15/15) split
# - We also compute:
#     - train_cutoff = last time_idx in the train block
#     - val_cutoff   = last time_idx in the validation block
#     
# These cutoffs are useful later when creating TFT datasets that require lookback context (e.g., 60 past days) at the start of validation/test windows. <br>
# Finally, we print the date ranges and time_idx cutoffs for sanity-checking that the split boundaries look correct.

# %%
TEST_FRAC = 0.15
VAL_FRAC  = 0.15
LOOKBACK  = 60
HORIZON   = 1

# sort once
data = data.sort_values("date").reset_index(drop=True).copy()
n = len(data)

# sizes
n_test     = int(n * TEST_FRAC)
n_trainval = n - n_test
n_val      = int(n_trainval * VAL_FRAC)
n_train    = n_trainval - n_val

# split
train_data = data.iloc[:n_train].copy()
val_data   = data.iloc[n_train:n_train + n_val].copy()
test_data  = data.iloc[n_train + n_val:].copy()

# cutoffs (last time_idx in each block)
train_cutoff = int(train_data["time_idx"].iloc[-1])
val_cutoff   = int(val_data["time_idx"].iloc[-1])

print(f"train {len(train_data)}: {train_data.date.min().date()} → {train_data.date.max().date()} | "
      f"val {len(val_data)}: {val_data.date.min().date()} → {val_data.date.max().date()} | "
      f"test {len(test_data)}: {test_data.date.min().date()} → {test_data.date.max().date()}")
print(f"Cutoffs (time_idx): train_cutoff={train_cutoff}, val_cutoff={val_cutoff}")

# %% [markdown]
# ### Step 3 - Build TFT DataSets
# 
# We convert id, dow, month into string-based pandas categoricals so pytorch-forecasting won’t complain about numeric categories.<br>
# common_args tells TimeSeriesDataSet what the time index is (time_idx), what the target is (price), what identifies the series (id), and what features are known (calendar) vs unknown (price history).<br>
# Uses GroupNormalizer so the target is normalized per series (no manual scaling step needed).<br>
# min_encoder_length = max_encoder_length = lookback forces every sample to use exactly LOOKBACK past days (consistent input size).<br>
# Builds train/val/test without leakage but with lookback context<br>
# Train uses data up to train_cutoff, and predictions start only once enough history exists (min_prediction_idx=lookback).<br>
# Validation includes earlier rows (so it has the last LOOKBACK train days available), but predictions start at train_cutoff + 1.<br>
# Test includes the full history for context, but predictions start at val_cutoff + 1.

# %%
def prepare_tft_categoricals(df: pd.DataFrame,
                            cat_cols=("id", "dow", "month")) -> pd.DataFrame:
    """
    pytorch-forecasting (some versions) rejects categoricals whose categories are numeric.
    Force numeric categoricals to string categories, then cast to 'category'.
    """
    df = df.copy()
    for c in cat_cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(int).astype(str)
        else:
            df[c] = df[c].astype(str)
        df[c] = df[c].astype("category")
    return df

def build_tft_datasets(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    train_cutoff: int,
    val_cutoff: int,
):
    """
    Version-compatible TFT dataset builder (no max_prediction_idx).
    Enforces split boundaries by slicing df to <= cutoff and using min_prediction_idx.
    Strict fixed lookback: min_encoder_length = max_encoder_length = lookback.
    """
    df = prepare_tft_categoricals(df)

    common_args = dict(
        time_idx="time_idx",
        target="price",
        group_ids=["id"],

        max_encoder_length=lookback,
        min_encoder_length=lookback,          # strict fixed lookback
        max_prediction_length=horizon,

        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=list(("dow", "month")),
        time_varying_unknown_reals=["price"],

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,

        target_normalizer=GroupNormalizer(groups=["id"]),
    )

    train_ds = TimeSeriesDataSet(
        df[df["time_idx"] <= train_cutoff].copy(),
        **common_args,
        min_prediction_idx=lookback
    )

    val_ds = TimeSeriesDataSet(
        df[df["time_idx"] <= val_cutoff].copy(),
        **common_args,
        min_prediction_idx=train_cutoff + 1
    )

    test_ds = TimeSeriesDataSet(
        df.copy(),
        **common_args,
        min_prediction_idx=val_cutoff + 1
    )

    return train_ds, val_ds, test_ds


# ---- usage ----
train_ds, val_ds, test_ds = build_tft_datasets(
    df=data,
    lookback=LOOKBACK,
    horizon=HORIZON,
    train_cutoff=train_cutoff,
    val_cutoff=val_cutoff,
)

print("Samples | train:", len(train_ds), "| val:", len(val_ds), "| test:", len(test_ds))


# %% [markdown]
# ### Step 4 - Dataloading
# 
# Creates PyTorch DataLoaders for TFT training/evaluation <br>
# Converts train_ds, val_ds, and test_ds (TimeSeriesDataSets) into iterable batches that the Lightning trainer can consume.<br>
# Uses BATCH_SIZE = 64<br>
# Controls how many time-series samples are processed per gradient step (train) or per forward pass (val/test).

# %%
BATCH_SIZE = ENV_BATCH_SIZE

train_loader = train_ds.to_dataloader(
    train=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=False,  # Disabled to avoid file descriptor exhaustion
    pin_memory=PIN_MEMORY,
)

val_loader = val_ds.to_dataloader(
    train=False,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=False,  # Disabled to avoid file descriptor exhaustion
    pin_memory=PIN_MEMORY,
)

test_loader = test_ds.to_dataloader(
    train=False,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=False,  # Disabled to avoid file descriptor exhaustion
    pin_memory=PIN_MEMORY,
)

# %% [markdown]
# ### Step 5 - One Run of TFT model for Verfication
# 
# Runs one end-to-end TFT baseline training to confirm the whole pipeline works (datasets → dataloaders → model → training loop).<br>
# Builds a Temporal Fusion Transformer from train_ds using a small, fixed configuration (learning rate, hidden sizes, dropout) and QuantileLoss (outputs multiple quantile forecasts).<br>
# Trains with EarlyStopping on val_loss (stop if validation loss stops improving) for up to 60 epochs on the MPS GPU.<br>
# The output logs/model summary are just sanity checks that the model instantiated correctly and training is progressing, giving you a reference baseline before running hyperparameter experiments.

# %%
if not ENV_SKIP_BASELINE_FIT:
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)

    tft_model = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=1e-3,
        hidden_size=32,
        attention_head_size=4,
        hidden_continuous_size=16,
        dropout=0.1,
        loss=QuantileLoss(list(quantiles)),
    )

    # --- fit ---
    callbacks = [EarlyStopping(monitor="val_loss", patience=min(ENV_PATIENCE, 8), mode="min")]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if torch.cuda.is_available() else None

    trainer = pl.Trainer(
        max_epochs=ENV_BASELINE_MAX_EPOCHS,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(tft_model, train_loader, val_loader)

# %% [markdown]
# ### Step 6 -  Evaluate TFT on the Test Set (Quantiles → Point Forecast Metrics)
# 
# Generate quantile forecasts from the TFT model<br>
# Convert quantiles into a single point forecast for comparability<br>
# evaluate_tft_baseline uses the P50 (median) quantile as the point prediction y_pred, enabling direct comparison to point-forecast models like the LSTM.
# Calculates MAE, RMSE, MAPE, and Directional Accuracy (whether the model predicts up/down movements correctly).<br>
# Plots Actual vs Predicted (P50), and (if available) shades the P05–P95 interval to visualise prediction uncertainty.

# %%
def predict_tft_quantiles(model, loader, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """
    Returns:
      preds_q: np.ndarray [N, H, Q]  (H=1 in your setup)
      y_true:  np.ndarray [N]
    """
    model.eval()

    preds_q = model.predict(loader, mode="quantiles")
    # robust to torch/numpy return types
    if hasattr(preds_q, "detach"):
        preds_q = preds_q.detach().cpu().numpy()
    else:
        preds_q = np.asarray(preds_q)

    actuals = []
    for x, y in loader:
        yy = y[0] if isinstance(y, (tuple, list)) else y
        actuals.append(yy.detach().cpu().numpy())
    y_true = np.concatenate(actuals, axis=0).reshape(-1)

    return preds_q, y_true


def evaluate_tft_baseline(
    model,
    test_loader,
    quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
    plot=True,
    save_path=None,          # <-- NEW: e.g. "runs/.../pred_vs_actual.png"
    dpi=150,
    title="TFT Next-Day Forecast — Test Set",
):
    """
    Computes MAE/RMSE/MAPE/Directional Accuracy using P50 as point forecast.
    If plot=True:
      - shows the plot (if save_path is None)
      - or saves to save_path (and closes) if save_path is provided
    Returns metrics dict + y_true/y_pred for saving elsewhere if desired.
    """
    preds_q, y_true = predict_tft_quantiles(model, test_loader, quantiles=quantiles)

    q_list = list(quantiles)
    if 0.5 not in q_list:
        raise ValueError("Quantiles must include 0.5 for P50 point forecast.")
    q50_idx = q_list.index(0.5)

    # H=1 => [N, Q]
    y_pred_q = preds_q[:, 0, :]
    y_pred = y_pred_q[:, q50_idx]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)

    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    da = float((true_dir == pred_dir).mean() * 100)

    if plot:
        x = np.arange(len(y_pred))
        # optional band if P05/P95 exist
        if 0.05 in q_list and 0.95 in q_list:
            qlo = y_pred_q[:, q_list.index(0.05)]
            qhi = y_pred_q[:, q_list.index(0.95)]
        else:
            qlo = qhi = None

        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="Actual")
        plt.plot(y_pred, label="Pred (P50)")
        if qlo is not None:
            plt.fill_between(x, qlo, qhi, alpha=0.2, label="P05–P95")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=dpi)
            plt.close()
        else:
            plt.show()

    print(f"MAE              : {mae:.4f}")
    print(f"RMSE             : {rmse:.4f}")
    print(f"MAPE             : {mape:.2f}%")
    print(f"Directional Acc. : {da:.2f}%")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Directional_Accuracy": da,
        "y_true": y_true,
        "y_pred_p50": y_pred,
    }

if not ENV_SKIP_BASELINE_FIT:
    tft_test_metrics = evaluate_tft_baseline(
        model=tft_model,
        test_loader=test_loader,
        quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        plot=True
    )
    print(tft_test_metrics)

# %% [markdown]
# ### Step 7 - HyperParameter Tuning
# 
# 

# %%
# ---- locked-in spec ----
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
MAX_EPOCHS = ENV_MAX_EPOCHS
PATIENCE = ENV_PATIENCE

# ---- search space ----
ENCODER_CHOICES = [20, 30, 45, 60, 90, 120, 180]
BATCH_CHOICES   = [32, 64, 128, 256]
CLIP_CHOICES    = [0.1, 0.25, 0.5, 1.0, 2.0]
HIDDEN_CHOICES  = [16, 24, 32, 48, 64, 96, 128, 192]
HEAD_CHOICES    = [1, 2, 4, 8]
HCONT_CHOICES   = [8, 16, 24, 32, 48, 64]
LSTM_LAYER_CHOICES = [1, 2, 3]

# cache datasets by lookback to avoid rebuilding for repeated encoder lengths
_DATASET_CACHE = {}

class OptunaPruningCallback(pl.Callback):
    def __init__(self, trial, monitor="val_loss"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.best = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        val = trainer.callback_metrics.get(self.monitor)
        if val is None:
            return
        score = float(val.detach().cpu().item()) if hasattr(val, "detach") else float(val)

        self.best = min(self.best, score)           # <-- keep best
        self.trial.report(score, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def _valid_heads(hidden_size: int):
    hs = [h for h in HEAD_CHOICES if (hidden_size % h == 0)]
    return hs if hs else [1]

def _valid_hidden_cont(hidden_size: int):
    hc = [c for c in HCONT_CHOICES if c <= hidden_size]
    return hc if hc else [min(HCONT_CHOICES)]

def _clamp_hidden_cont(hidden_cont: int, hidden_size: int) -> int:
    """Clamp hidden_continuous_size to be at most hidden_size."""
    return min(hidden_cont, hidden_size)


def _get_datasets_for_lookback(lookback: int):
    """
    Uses your existing build_tft_datasets(...) and global:
      - data, HORIZON, train_cutoff, val_cutoff
    Strict lookback is enforced inside build_tft_datasets (min=max=lookback).
    """
    if lookback in _DATASET_CACHE:
        return _DATASET_CACHE[lookback]

    train_ds, val_ds, test_ds = build_tft_datasets(
        df=data,
        lookback=lookback,
        horizon=HORIZON,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
    )
    _DATASET_CACHE[lookback] = (train_ds, val_ds, test_ds)
    return train_ds, val_ds, test_ds


def _make_loaders(train_ds, val_ds, test_ds, batch_size: int, num_workers: int = NUM_WORKERS):
    train_loader = train_ds.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=False,  # Disabled to avoid file descriptor exhaustion
        pin_memory=PIN_MEMORY,
    )
    val_loader = val_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=False,  # Disabled to avoid file descriptor exhaustion
        pin_memory=PIN_MEMORY,
    )
    test_loader = test_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=False,  # Disabled to avoid file descriptor exhaustion
        pin_memory=PIN_MEMORY,
    )
    return train_loader, val_loader, test_loader


def build_tft_model_from_trial(trial: optuna.Trial, train_ds, lookback: int):
    """
    Build TFT model from trial. `lookback` is passed in (already sampled in objective)
    to avoid double-sampling max_encoder_length.
    """
    # Store lookback as a user attribute (not re-sampled) so it shows in trial results
    trial.set_user_attr("max_encoder_length", lookback)

    # batch/optim
    batch_size = trial.suggest_categorical("batch_size", BATCH_CHOICES)
    lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    clip = trial.suggest_categorical("gradient_clip_val", CLIP_CHOICES)

    use_wd = trial.suggest_categorical("use_weight_decay", [False, True])
    wd = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True) if use_wd else 0.0

    # capacity
    hidden_size = trial.suggest_categorical("hidden_size", HIDDEN_CHOICES)
    
    # For attention heads: sample from fixed choices, then find a valid divisor
    head_size_raw = trial.suggest_categorical("attention_head_size", HEAD_CHOICES)
    valid_heads = _valid_heads(hidden_size)
    head_size = max(h for h in valid_heads if h <= head_size_raw) if any(h <= head_size_raw for h in valid_heads) else valid_heads[0]
    
    # For hidden_continuous_size: sample from fixed choices, then clamp to hidden_size
    hidden_cont_raw = trial.suggest_categorical("hidden_continuous_size", HCONT_CHOICES)
    hidden_cont = _clamp_hidden_cont(hidden_cont_raw, hidden_size)
    
    dropout     = trial.suggest_float("dropout", 0.0, 0.4)
    lstm_layers = trial.suggest_categorical("lstm_layers", LSTM_LAYER_CHOICES)

    loss = QuantileLoss(list(QUANTILES))

    # build model (be tolerant to version differences)
    # NOTE: pytorch_forecasting TFT has its own weight_decay param, don't use optimizer_params
    model_kwargs = dict(
        learning_rate=lr,
        hidden_size=hidden_size,
        attention_head_size=head_size,
        hidden_continuous_size=hidden_cont,
        dropout=dropout,
        loss=loss,
        weight_decay=wd,  # pass weight_decay directly to TFT
    )

    # these exist in most pytorch-forecasting versions; keep robust
    extra_kwargs = dict(
        lstm_layers=lstm_layers,
    )

    try:
        model = TemporalFusionTransformer.from_dataset(train_ds, **model_kwargs, **extra_kwargs)
    except TypeError:
        # fallback if lstm_layers not supported in your version
        model = TemporalFusionTransformer.from_dataset(train_ds, **model_kwargs)

    hparams = dict(
        max_encoder_length=lookback,
        batch_size=batch_size,
        learning_rate=lr,
        gradient_clip_val=clip,
        weight_decay=wd,
        hidden_size=hidden_size,
        attention_head_size=head_size,
        hidden_continuous_size=hidden_cont,
        dropout=dropout,
        lstm_layers=lstm_layers,
        quantiles=list(QUANTILES),
    )
    return model, hparams


def objective(trial: optuna.Trial):
    start_time = time.time()
    pl.seed_everything(SEED, workers=True)

    # sample lookback ONCE here
    lookback = trial.suggest_categorical("max_encoder_length", ENCODER_CHOICES)
    train_ds, val_ds, test_ds = _get_datasets_for_lookback(lookback)

    # build model (lookback passed in to avoid double-sampling)
    model, hparams = build_tft_model_from_trial(trial, train_ds=train_ds, lookback=lookback)

    batch_size = hparams["batch_size"]
    train_loader, val_loader, test_loader = _make_loaders(train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=NUM_WORKERS)

    # callbacks: early stopping + pruning (track best val_loss)
    prune_cb = OptunaPruningCallback(trial, monitor="val_loss")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"),
        prune_cb,
    ]

    clip_val = hparams.get("gradient_clip_val", 0.5)
    
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if torch.cuda.is_available() else None
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=clip_val,
        enable_checkpointing=False,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(model, train_loader, val_loader)

    best_val = prune_cb.best
    epochs_ran = trainer.current_epoch + 1
    duration_sec = float(time.time() - start_time)

    # ---- Store training metrics in Optuna DB (user_attrs) ----
    trial.set_user_attr("duration_sec", duration_sec)
    trial.set_user_attr("epochs_ran", epochs_ran)
    trial.set_user_attr("val_loss_best", float(best_val))

    # ---- Evaluate on test set (like LSTM) ----
    try:
        test_metrics = evaluate_tft_baseline(
            model=model,
            test_loader=test_loader,
            quantiles=QUANTILES,
            plot=False,
        )
        # Store test metrics in Optuna DB (matching LSTM column names)
        trial.set_user_attr("test_mae", float(test_metrics["MAE"]))
        trial.set_user_attr("test_rmse", float(test_metrics["RMSE"]))
        trial.set_user_attr("test_mape", float(test_metrics["MAPE"]))
        trial.set_user_attr("test_directional_accuracy", float(test_metrics["Directional_Accuracy"]))
    except Exception as e:
        print(f"[trial {trial.number}] Test eval failed: {e}")
        trial.set_user_attr("test_mae", None)
        trial.set_user_attr("test_rmse", None)
        trial.set_user_attr("test_mape", None)
        trial.set_user_attr("test_directional_accuracy", None)

    # Store hyperparameters as user attrs for easy CSV export
    trial.set_user_attr("lookback", int(lookback))
    trial.set_user_attr("batch_size", int(batch_size))
    trial.set_user_attr("hidden_size", int(hparams["hidden_size"]))
    trial.set_user_attr("attention_head_size", int(hparams["attention_head_size"]))
    trial.set_user_attr("hidden_continuous_size", int(hparams["hidden_continuous_size"]))
    trial.set_user_attr("dropout", float(hparams["dropout"]))
    trial.set_user_attr("lstm_layers", int(hparams["lstm_layers"]))
    trial.set_user_attr("learning_rate", float(hparams["learning_rate"]))
    trial.set_user_attr("gradient_clip_val", float(hparams["gradient_clip_val"]))
    trial.set_user_attr("weight_decay", float(hparams["weight_decay"]))

    print(f"[trial {trial.number:03d}] ✅ COMPLETED in {duration_sec:.1f}s | best val_loss={best_val:.4f}")

    # cleanup
    del trainer, model
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return float(best_val)


# ---- Study runner with SQLite storage for resumable runs ----
# Use the RUN_DIR for self-contained experiment  
STORAGE_URL = f"sqlite:///{DB_PATH}"
# STUDY_NAME already defined above with timestamp for unique identification

study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE_URL,
    load_if_exists=True,          # <-- resume from previous runs if DB exists
    direction="minimize",
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_warmup_steps=5),
)

optuna.logging.set_verbosity(optuna.logging.INFO)

# set as high as you want (you said thousands)
N_TRIALS = ENV_N_TRIALS
study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

print("Best value (val_loss):", study.best_value)
print("Best params:", study.best_params)

# %% [markdown]
# ### Step 8 - Export Trial Summary CSV (Matching LSTM Format)
# 
# Creates a unique output directory for this run using the current UTC timestamp, e.g. saved_tft_models/20260111-154233_price_only_tft/, so results don’t overwrite previous runs.
# Exports the full Optuna trial log to optuna_trials.csv (trial id, state, objective value, sampled hyperparameters).
# Saves the best trial summary to best_params.json (best value + best hyperparameters).
# Generates and saves Optuna plots:
# - optuna_history.png (objective over trials)
# - optuna_param_importances.png + optuna_param_importances.json (hyperparameter importance), or an error file if importances can’t be computed.

# %%
def export_trial_summary(study: optuna.Study, out_csv: str):
    """
    Export a flat CSV with key metrics as columns (RMSE/MAE/MAPE/DA etc),
    plus params/user_attrs blobs for debugging. Matches LSTM format.
    """
    wanted = [
        # runtime / training
        "duration_sec", "epochs_ran",
        "val_loss_best",
        # test metrics (real units)
        "test_mae", "test_rmse", "test_mape", "test_directional_accuracy",
        # TFT-specific hyperparameters
        "lookback", "batch_size", "hidden_size", "attention_head_size",
        "hidden_continuous_size", "dropout", "lstm_layers", "learning_rate",
        "gradient_clip_val", "weight_decay",
        # prune diagnostics (if available)
        "pruned_epoch", "last_val_loss",
        # optional export info (if you set it later)
        "exported_rank", "export_dir",
    ]

    rows = []
    for t in study.trials:
        ua = t.user_attrs or {}

        row = {
            "trial_number": t.number,
            "state": t.state.name,
            "value": t.value,
            "datetime_start": str(t.datetime_start) if t.datetime_start else None,
            "datetime_complete": str(t.datetime_complete) if t.datetime_complete else None,
            "params": json.dumps(t.params, default=str),
        }

        # Flatten selected user attrs into dedicated columns
        for k in wanted:
            row[k] = ua.get(k, None)

        # Keep the full blob too
        row["user_attrs"] = json.dumps(ua, default=str)

        rows.append(row)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote trial summary: {out_csv}")


def save_optuna_study_artifacts(study: optuna.Study, out_dir: str):
    """Save best params + basic Optuna plots (PNG)."""
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)

    # optimization history plot
    fig1 = plot_optimization_history(study)
    fig1.figure.savefig(os.path.join(out_dir, "optuna_history.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1.figure)

    # param importances plot (can fail if too few completed trials)
    try:
        fig2 = plot_param_importances(study)
        fig2.figure.savefig(os.path.join(out_dir, "optuna_param_importances.png"), dpi=150, bbox_inches="tight")
        plt.close(fig2.figure)

        imp = get_param_importances(study)
        with open(os.path.join(out_dir, "optuna_param_importances.json"), "w") as f:
            json.dump(imp, f, indent=2)
    except Exception as e:
        with open(os.path.join(out_dir, "optuna_param_importances_error.txt"), "w") as f:
            f.write(str(e))


# Export trial summary CSV (like LSTM)
export_trial_summary(study, TRIAL_SUMMARY_CSV)

# Save Optuna artifacts to RUN_DIR
save_optuna_study_artifacts(study, RUN_DIR)

# %% [markdown]
# ### Step 9 - Save Best TFT Run Artifacts (Checkpoint + Metrics + Plots)
# 
# Trains the final TFT model using the best Optuna hyperparameters, then saves all outputs to the timestamped out_dir run folder:
# - Best model checkpoint (lowest val_loss): out_dir/checkpoints/best-*.ckpt
# - Training logs (epoch metrics): out_dir/logs/metrics.csv
# - Test evaluation plot: out_dir/pred_vs_actual.png
# - Final test metrics + checkpoint path: out_dir/final_test_metrics.json
# - Validation-loss training curve: out_dir/training_curve_val_loss.png

# %%
def build_tft_from_best_params(train_ds, best_params: dict, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Build TFT robustly across pytorch-forecasting versions."""
    loss = QuantileLoss(list(quantiles))

    # required / core knobs
    model_kwargs = dict(
        learning_rate=float(best_params["learning_rate"]),
        hidden_size=int(best_params["hidden_size"]),
        attention_head_size=int(best_params["attention_head_size"]),
        hidden_continuous_size=int(best_params["hidden_continuous_size"]),
        dropout=float(best_params["dropout"]),
        loss=loss,
    )

    # optional knobs (might not exist in every version)
    extra_kwargs = {}
    if "lstm_layers" in best_params:
        extra_kwargs["lstm_layers"] = int(best_params["lstm_layers"])

    # weight decay (only if you used it)
    wd = float(best_params.get("weight_decay", 0.0))
    if wd > 0:
        extra_kwargs["optimizer_params"] = {"weight_decay": wd}

    # try with extras first; fallback if your version doesn't support them
    try:
        model = TemporalFusionTransformer.from_dataset(train_ds, **model_kwargs, **extra_kwargs)
    except TypeError:
        model = TemporalFusionTransformer.from_dataset(train_ds, **model_kwargs)

    return model

def fit_best_tft(
    model,
    train_loader,
    val_loader,
    out_dir: str,
    max_epochs=60,
    patience=8,
    gradient_clip_val=0.5,
):
    """Train best TFT once, save best checkpoint, and log epoch metrics to CSV."""
    logger = CSVLogger(save_dir=out_dir, name="logs")

    ckpt = ModelCheckpoint(
        dirpath=os.path.join(out_dir, "checkpoints"),
        filename="best-{epoch:02d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if torch.cuda.is_available() else None

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=float(gradient_clip_val),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
            ckpt,
        ],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(model, train_loader, val_loader)

    best_path = ckpt.best_model_path
    best_score = ckpt.best_model_score
    best_score = float(best_score.detach().cpu().item()) if best_score is not None else None
    return trainer, best_path, best_score, logger.log_dir

def save_training_curves_from_csv(log_dir: str, out_dir: str):
    """Plot train/val curves from Lightning CSVLogger metrics.csv."""
    metrics_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        return

    m = pd.read_csv(metrics_path)
    # Lightning stores metrics across steps; keep one point per epoch where val_loss exists
    if "epoch" not in m.columns:
        return

    # plot val_loss vs epoch
    if "val_loss" in m.columns:
        vv = m.dropna(subset=["val_loss"]).groupby("epoch", as_index=False)["val_loss"].last()
        plt.figure(figsize=(8, 4))
        plt.plot(vv["epoch"], vv["val_loss"], label="val_loss")
        plt.title("Training Curve — val_loss")
        plt.xlabel("epoch"); plt.ylabel("val_loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_curve_val_loss.png"), dpi=150)
        plt.close()

# 2) rebuild datasets/loaders using best lookback + best batch_size
best = study.best_params
best_lookback = int(best["max_encoder_length"])
best_batch    = int(best["batch_size"])

train_ds_best, val_ds_best, test_ds_best = build_tft_datasets(
    df=data,
    lookback=best_lookback,
    horizon=HORIZON,
    train_cutoff=train_cutoff,
    val_cutoff=val_cutoff,
)

train_loader_best = train_ds_best.to_dataloader(
    train=True,
    batch_size=best_batch,
    num_workers=NUM_WORKERS,
    persistent_workers=False,  # Disabled to avoid file descriptor exhaustion
    pin_memory=PIN_MEMORY,
)
val_loader_best = val_ds_best.to_dataloader(
    train=False,
    batch_size=best_batch,
    num_workers=NUM_WORKERS,
    persistent_workers=False,  # Disabled to avoid file descriptor exhaustion
    pin_memory=PIN_MEMORY,
)
test_loader_best = test_ds_best.to_dataloader(
    train=False,
    batch_size=best_batch,
    num_workers=NUM_WORKERS,
    persistent_workers=False,  # Disabled to avoid file descriptor exhaustion
    pin_memory=PIN_MEMORY,
)

# 3) build + fit best model once (early stop on val_loss, save checkpoint)
best_model = build_tft_from_best_params(train_ds_best, best, quantiles=QUANTILES)

trainer, best_ckpt_path, best_val, log_dir = fit_best_tft(
    model=best_model,
    train_loader=train_loader_best,
    val_loader=val_loader_best,
    out_dir=RUN_DIR,
    max_epochs=ENV_MAX_EPOCHS,
    patience=ENV_PATIENCE,
    gradient_clip_val=float(best.get("gradient_clip_val", 0.5)),
)

print("Best checkpoint:", best_ckpt_path)
print("Best val_loss:", best_val)

# 4) evaluate on test and save plot/metrics
# PyTorch 2.6+ changed weights_only default to True, but Lightning checkpoints
# contain custom objects (pandas DataFrames, etc.) that require weights_only=False
# We monkey-patch torch.load to force weights_only=False for checkpoint loading
import torch
_original_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False  # Force weights_only=False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_load

best_model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt_path)

# Restore original torch.load
torch.load = _original_torch_load

metrics = evaluate_tft_baseline(
    model=best_model,
    test_loader=test_loader_best,
    quantiles=QUANTILES,
    plot=True,
    save_path=os.path.join(RUN_DIR, "pred_vs_actual.png"),
)

with open(os.path.join(RUN_DIR, "final_test_metrics.json"), "w") as f:
    json.dump(
        {"best_val_loss": best_val, "best_checkpoint": best_ckpt_path, "test_metrics": metrics},
        f,
        indent=2,
        default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
    )

# 5) save training curves
save_training_curves_from_csv(log_dir, RUN_DIR)

print("Test metrics:", metrics)
print("Saved best-run artifacts to:", RUN_DIR)

# cleanup GPU memory
try:
    torch.cuda.empty_cache()
except Exception:
    pass


