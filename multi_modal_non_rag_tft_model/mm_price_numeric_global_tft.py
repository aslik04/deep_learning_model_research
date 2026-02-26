# %% [markdown]
# ## Price + Numeric + Global News — Multimodal Non-RAG TFT (Henry Hub)
#
# Feature-engineered multimodal Temporal Fusion Transformer.
# Combines 6 numeric fundamentals with 7 Global FinBERT sentiment features.

# %% [markdown]
# ### Imports and Configuration

# %%
import os, json, time, pickle, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import torch
import pytorch_lightning as pl
import optuna

from contextlib import contextmanager
from datetime import datetime, timezone

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.importance import get_param_importances
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

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
ENV_SKIP_BASELINE_FIT = os.environ.get("SKIP_BASELINE_FIT", "1") == "1"
ENV_BASELINE_MAX_EPOCHS = int(os.environ.get("BASELINE_MAX_EPOCHS", str(ENV_MAX_EPOCHS)))

TASK_ID = os.environ.get("SLURM_ARRAY_TASK_ID", "local")
EXPERIMENT = os.environ.get("EXP_NAME", "price_numeric_global_news")

# ------------------------------------------------------------------
# Run configuration
# ------------------------------------------------------------------
EXPERIMENT_NAME = "Price+Numeric+GlobalNews"
MODEL_TAG = "tft"
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

TOP_K = 10
SAVED_RESULTS_DIR = "saved_results"

RUN_NAME = f"{RUN_TS}_{EXPERIMENT_NAME.replace(' ', '')}_{MODEL_TAG}"
RUN_DIR = os.path.join(SAVED_RESULTS_DIR, RUN_NAME)
TRIAL_SUMMARY_CSV = os.path.join(RUN_DIR, "trial_summary.csv")

DB_PATH = os.path.join(RUN_DIR, f"optuna_{EXPERIMENT_NAME.replace(' ', '')}_{MODEL_TAG}.db")
STUDY_NAME = RUN_NAME

os.makedirs(RUN_DIR, exist_ok=True)

# %%
# ─────────────────────────────────────────────────────────────
# SLURM / HPC: DataLoader + GPU config
# ─────────────────────────────────────────────────────────────
SLURM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
NUM_WORKERS = max(1, min(4, SLURM_CPUS - 1))
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

# ------------------------------------------------------------------
# Utility: atomic JSON writes & safe torch.load
# ------------------------------------------------------------------
def safe_json_dump(obj, path, **kwargs):
    """Write JSON atomically via temp file to prevent corruption on crash."""
    kwargs.setdefault("indent", 2)
    kwargs.setdefault("default", lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kwargs)
    os.replace(tmp, path)


@contextmanager
def patch_torch_load():
    """Temporarily force weights_only=False for Lightning checkpoint loading."""
    original = torch.load
    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)
    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


# %% [markdown]
# ### Step 1 — Load Master CSV + News Features and Transform Dataset
#
# ── Feature Roles ──
# News features are modeled as time_varying_unknown_reals:
#   "unknown at prediction time" — the model cannot see future news.
#   Zero-filled pre-2015 = explicit absence of signal (not missing data).
# Calendar features (dow, month) are time_varying_known_categoricals:
#   known in advance for any future date.
# price is both target and time_varying_unknown_real (autoregressive).

# %%
CSV = "../numeric_data/henryhub_master.csv"
NEWS_PARQUET = "../news_data/daily_energy_signals_enhanced.parquet"
TARGET = "price"
GROUP_COL = "id"

# ── Per-experiment configuration (ONLY these lines differ between scripts) ──
NUM_COLS = ["storage_bcf", "production_bcf", "usd_index", "temp_c", "temp_max_c", "temp_min_c"]

NEWS_COLS = [
    "Global_Vol", "Global_Avg_Pos", "Global_Avg_Neg", "Global_Avg_Neu",
    "Global_Max_Pos", "Global_Max_Neg", "Global_Dispersion",
]

# All covariates fed to TFT as time_varying_unknown_reals
COVARIATE_COLS = NUM_COLS + NEWS_COLS


def load_tft_ready_df(csv_path: str, news_path: str, news_cols: list) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values([GROUP_COL, "date"]).reset_index(drop=True)

    # ── Merge news features ──
    news = pd.read_parquet(news_path).reset_index()
    news["date"] = pd.to_datetime(news["date"])
    news = news[["date"] + news_cols]
    df = df.merge(news, on="date", how="left")

    # Fill NaN with 0.0 (pre-2015 dates have no news = no signal)
    for c in news_cols:
        df[c] = df[c].fillna(0.0).astype("float32")

    # group id as categorical
    df[GROUP_COL] = df[GROUP_COL].astype("category")

    # target numeric
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype("float32")

    # numeric covariates
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

data = load_tft_ready_df(CSV, NEWS_PARQUET, NEWS_COLS)
print(f"Loaded {len(data)} rows with {len(COVARIATE_COLS)} covariates + price")
data.head()

# ── Save data config for reproducibility ──
safe_json_dump({
    "csv_path": CSV,
    "news_parquet_path": NEWS_PARQUET,
    "experiment_name": EXPERIMENT_NAME,
    "news_cols": NEWS_COLS,
    "num_cols": NUM_COLS,
    "n_rows": len(data),
    "date_range": [str(data.date.min().date()), str(data.date.max().date())],
    "n_news_features": len(NEWS_COLS),
    "n_numeric_features": len(NUM_COLS),
    "news_fill_strategy": "zero",
}, os.path.join(RUN_DIR, "data_config.json"))


# %% [markdown]
# ### Step 2 — Chronological Train/Validation/Test Split (70/15/15)

# %%
TEST_FRAC = 0.15
VAL_FRAC  = 0.15
LOOKBACK  = 60
HORIZON   = 1

data = data.sort_values("date").reset_index(drop=True).copy()
n = len(data)

n_test     = int(n * TEST_FRAC)
n_trainval = n - n_test
n_val      = int(n_trainval * VAL_FRAC)
n_train    = n_trainval - n_val

train_data = data.iloc[:n_train].copy()
val_data   = data.iloc[n_train:n_train + n_val].copy()
test_data  = data.iloc[n_train + n_val:].copy()

train_cutoff = int(train_data["time_idx"].iloc[-1])
val_cutoff   = int(val_data["time_idx"].iloc[-1])

print(f"train {len(train_data)}: {train_data.date.min().date()} → {train_data.date.max().date()} | "
      f"val {len(val_data)}: {val_data.date.min().date()} → {val_data.date.max().date()} | "
      f"test {len(test_data)}: {test_data.date.min().date()} → {test_data.date.max().date()}")
print(f"Cutoffs (time_idx): train_cutoff={train_cutoff}, val_cutoff={val_cutoff}")


# %% [markdown]
# ### Step 3 — Build TFT DataSets

# %%
def prepare_tft_categoricals(df: pd.DataFrame,
                            cat_cols=("id", "dow", "month")) -> pd.DataFrame:
    """Force numeric categoricals to string categories for pytorch-forecasting."""
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
    Version-compatible TFT dataset builder.
    Strict fixed lookback: min_encoder_length = max_encoder_length = lookback.
    """
    df = prepare_tft_categoricals(df)

    common_args = dict(
        time_idx="time_idx",
        target="price",
        group_ids=["id"],

        max_encoder_length=lookback,
        min_encoder_length=lookback,
        max_prediction_length=horizon,

        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=list(("dow", "month")),
        time_varying_unknown_reals=["price"] + COVARIATE_COLS,

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


train_ds, val_ds, test_ds = build_tft_datasets(
    df=data,
    lookback=LOOKBACK,
    horizon=HORIZON,
    train_cutoff=train_cutoff,
    val_cutoff=val_cutoff,
)

print("Samples | train:", len(train_ds), "| val:", len(val_ds), "| test:", len(test_ds))


# %% [markdown]
# ### Step 4 — DataLoaders

# %%
BATCH_SIZE = ENV_BATCH_SIZE

train_loader = train_ds.to_dataloader(
    train=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=False,
    pin_memory=PIN_MEMORY,
)

val_loader = val_ds.to_dataloader(
    train=False,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=False,
    pin_memory=PIN_MEMORY,
)

test_loader = test_ds.to_dataloader(
    train=False,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=False,
    pin_memory=PIN_MEMORY,
)

# %% [markdown]
# ### Step 5 — Baseline Verification Run (skippable)

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
# ### Step 6 — Evaluate TFT on Test Set (Quantiles → Point Forecast Metrics)

# %%
def predict_tft_quantiles(model, loader, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Returns preds_q [N, H, Q] and y_true [N]."""
    model.eval()

    preds_q = model.predict(loader, mode="quantiles")
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
    save_path=None,
    dpi=150,
    title="TFT Next-Day Forecast — Test Set",
):
    """Computes MAE/RMSE/MAPE/Directional Accuracy using P50 as point forecast."""
    preds_q, y_true = predict_tft_quantiles(model, test_loader, quantiles=quantiles)

    q_list = list(quantiles)
    if 0.5 not in q_list:
        raise ValueError("Quantiles must include 0.5 for P50 point forecast.")
    q50_idx = q_list.index(0.5)

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
        if 0.05 in q_list and 0.95 in q_list:
            qlo = y_pred_q[:, q_list.index(0.05)]
            qhi = y_pred_q[:, q_list.index(0.95)]
        else:
            qlo = qhi = None

        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="Actual")
        plt.plot(y_pred, label="Pred (P50)")
        if qlo is not None:
            plt.fill_between(x, qlo, qhi, alpha=0.2, label="P05-P95")
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
# ### Step 7 — Hyperparameter Tuning (Optuna)

# %%
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

        self.best = min(self.best, score)
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
    return min(hidden_cont, hidden_size)


def _get_datasets_for_lookback(lookback: int):
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
        persistent_workers=False,
        pin_memory=PIN_MEMORY,
    )
    val_loader = val_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=PIN_MEMORY,
    )
    test_loader = test_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=PIN_MEMORY,
    )
    return train_loader, val_loader, test_loader


def build_tft_model_from_trial(trial: optuna.Trial, train_ds, lookback: int):
    """Build TFT model from Optuna trial with granular version fallbacks."""
    trial.set_user_attr("max_encoder_length", lookback)

    batch_size = trial.suggest_categorical("batch_size", BATCH_CHOICES)
    lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    clip = trial.suggest_categorical("gradient_clip_val", CLIP_CHOICES)

    use_wd = trial.suggest_categorical("use_weight_decay", [False, True])
    wd = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True) if use_wd else 0.0

    hidden_size = trial.suggest_categorical("hidden_size", HIDDEN_CHOICES)

    head_size_raw = trial.suggest_categorical("attention_head_size", HEAD_CHOICES)
    valid_heads = _valid_heads(hidden_size)
    head_size = max(h for h in valid_heads if h <= head_size_raw) if any(h <= head_size_raw for h in valid_heads) else valid_heads[0]

    hidden_cont_raw = trial.suggest_categorical("hidden_continuous_size", HCONT_CHOICES)
    hidden_cont = _clamp_hidden_cont(hidden_cont_raw, hidden_size)

    dropout     = trial.suggest_float("dropout", 0.0, 0.4)
    lstm_layers = trial.suggest_categorical("lstm_layers", LSTM_LAYER_CHOICES)

    loss = QuantileLoss(list(QUANTILES))

    model_kwargs = dict(
        learning_rate=lr,
        hidden_size=hidden_size,
        attention_head_size=head_size,
        hidden_continuous_size=hidden_cont,
        dropout=dropout,
        loss=loss,
        weight_decay=wd,
    )

    # Granular fallback: try all extras → without lstm_layers → bare minimum
    try:
        model = TemporalFusionTransformer.from_dataset(train_ds, **model_kwargs, lstm_layers=lstm_layers)
    except TypeError:
        print(f"  [trial {trial.number}] Warning: lstm_layers not supported, falling back")
        try:
            model = TemporalFusionTransformer.from_dataset(train_ds, **model_kwargs)
        except TypeError:
            model_kwargs.pop("weight_decay", None)
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

    lookback = trial.suggest_categorical("max_encoder_length", ENCODER_CHOICES)
    train_ds, val_ds, test_ds = _get_datasets_for_lookback(lookback)

    model, hparams = build_tft_model_from_trial(trial, train_ds=train_ds, lookback=lookback)

    batch_size = hparams["batch_size"]
    train_loader, val_loader, test_loader = _make_loaders(train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=NUM_WORKERS)

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

    trial.set_user_attr("duration_sec", duration_sec)
    trial.set_user_attr("epochs_ran", epochs_ran)
    trial.set_user_attr("val_loss_best", float(best_val))

    try:
        test_metrics = evaluate_tft_baseline(
            model=model,
            test_loader=test_loader,
            quantiles=QUANTILES,
            plot=False,
        )
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

    print(f"[trial {trial.number:03d}] COMPLETED in {duration_sec:.1f}s | best val_loss={best_val:.4f}")

    del trainer, model
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return float(best_val)


# ---- Study runner with SQLite storage for resumable runs ----
STORAGE_URL = f"sqlite:///{DB_PATH}"

study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE_URL,
    load_if_exists=True,
    direction="minimize",
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_warmup_steps=5),
)

optuna.logging.set_verbosity(optuna.logging.INFO)

N_TRIALS = ENV_N_TRIALS
study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

print("Best value (val_loss):", study.best_value)
print("Best params:", study.best_params)

# %% [markdown]
# ### Step 8 — Export Trial Summary CSV

# %%
def export_trial_summary(study: optuna.Study, out_csv: str):
    """Export flat CSV with key metrics. Also exports completed-only version."""
    wanted = [
        "duration_sec", "epochs_ran", "val_loss_best",
        "test_mae", "test_rmse", "test_mape", "test_directional_accuracy",
        "lookback", "batch_size", "hidden_size", "attention_head_size",
        "hidden_continuous_size", "dropout", "lstm_layers", "learning_rate",
        "gradient_clip_val", "weight_decay",
        "pruned_epoch", "last_val_loss",
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

        for k in wanted:
            row[k] = ua.get(k, None)

        row["user_attrs"] = json.dumps(ua, default=str)
        rows.append(row)

    df_all = pd.DataFrame(rows)
    df_all.to_csv(out_csv, index=False)
    print(f"Wrote trial summary: {out_csv}")

    # Also export completed-only for cleaner analysis
    completed_csv = out_csv.replace(".csv", "_completed.csv")
    df_all[df_all["state"] == "COMPLETE"].to_csv(completed_csv, index=False)
    print(f"Wrote completed trials: {completed_csv}")


def save_optuna_study_artifacts(study: optuna.Study, out_dir: str):
    """Save best params + Optuna plots."""
    safe_json_dump({"best_value": study.best_value, "best_params": study.best_params},
                   os.path.join(out_dir, "best_params.json"))

    fig1 = plot_optimization_history(study)
    fig1.figure.savefig(os.path.join(out_dir, "optuna_history.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1.figure)

    try:
        fig2 = plot_param_importances(study)
        fig2.figure.savefig(os.path.join(out_dir, "optuna_param_importances.png"), dpi=150, bbox_inches="tight")
        plt.close(fig2.figure)

        imp = get_param_importances(study)
        safe_json_dump(imp, os.path.join(out_dir, "optuna_param_importances.json"))
    except Exception as e:
        with open(os.path.join(out_dir, "optuna_param_importances_error.txt"), "w") as f:
            f.write(str(e))


export_trial_summary(study, TRIAL_SUMMARY_CSV)
save_optuna_study_artifacts(study, RUN_DIR)

# %% [markdown]
# ### Step 9 — Retrain Best Model, Save Checkpoint + Metrics + VSN

# %%
def build_tft_from_best_params(train_ds, best_params: dict, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Build TFT robustly with granular version fallbacks."""
    loss = QuantileLoss(list(quantiles))

    model_kwargs = dict(
        learning_rate=float(best_params["learning_rate"]),
        hidden_size=int(best_params["hidden_size"]),
        attention_head_size=int(best_params["attention_head_size"]),
        hidden_continuous_size=int(best_params["hidden_continuous_size"]),
        dropout=float(best_params["dropout"]),
        loss=loss,
    )

    wd = float(best_params.get("weight_decay", 0.0))
    lstm_layers = best_params.get("lstm_layers", None)

    # Granular fallback: try all → without lstm_layers → bare minimum
    try:
        extra = {}
        if lstm_layers is not None:
            extra["lstm_layers"] = int(lstm_layers)
        if wd > 0:
            extra["weight_decay"] = wd
        model = TemporalFusionTransformer.from_dataset(train_ds, **model_kwargs, **extra)
    except TypeError:
        print("  Warning: fallback without lstm_layers")
        try:
            if wd > 0:
                model = TemporalFusionTransformer.from_dataset(train_ds, **model_kwargs, weight_decay=wd)
            else:
                model = TemporalFusionTransformer.from_dataset(train_ds, **model_kwargs)
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
    """Train best TFT, save checkpoint, log epoch metrics."""
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
    """Plot train/val curves from Lightning CSVLogger."""
    metrics_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        return

    m = pd.read_csv(metrics_path)
    if "epoch" not in m.columns:
        return

    if "val_loss" in m.columns:
        vv = m.dropna(subset=["val_loss"]).groupby("epoch", as_index=False)["val_loss"].last()
        plt.figure(figsize=(8, 4))
        plt.plot(vv["epoch"], vv["val_loss"], label="val_loss")
        plt.title("Training Curve — val_loss")
        plt.xlabel("epoch"); plt.ylabel("val_loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_curve_val_loss.png"), dpi=150)
        plt.close()


def save_vsn_importances(model, val_loader, out_dir):
    """Extract and save Variable Selection Network importances."""
    try:
        raw_preds = model.predict(val_loader, return_x=True, mode="raw")
        interpretation = model.interpret_output(raw_preds, reduction="mean")

        vsn_data = {}
        for key in ["encoder_variables", "decoder_variables", "static_variables"]:
            if key not in interpretation:
                continue
            values = interpretation[key]
            if isinstance(values, dict):
                # pf 0.10 with reduction="mean" returns {name: tensor_scalar}
                vsn_data[key] = {str(k): float(v) for k, v in values.items()}
            else:
                if hasattr(values, "detach"):
                    values = values.detach().cpu().numpy()
                names = getattr(model, key, [str(i) for i in range(len(values))])
                vsn_data[key] = {str(n): float(v) for n, v in zip(names, values)}

        safe_json_dump(vsn_data, os.path.join(out_dir, "vsn_importances.json"))

        # Plot top-20 encoder variable importances
        if "encoder_variables" in interpretation:
            enc_raw = interpretation["encoder_variables"]
            if isinstance(enc_raw, dict):
                sorted_items = sorted(enc_raw.items(),
                                      key=lambda x: float(x[1]), reverse=True)
                n_show = min(20, len(sorted_items))
                top_names = [it[0] for it in sorted_items[:n_show]]
                top_vals  = [float(it[1]) for it in sorted_items[:n_show]]
            else:
                if hasattr(enc_raw, "detach"):
                    enc_raw = enc_raw.detach().cpu().numpy()
                var_names = getattr(model, "encoder_variables",
                                    [str(i) for i in range(len(enc_raw))])
                n_show = min(20, len(var_names))
                sorted_idx = np.argsort(enc_raw)[::-1][:n_show]
                top_names = [var_names[i] for i in sorted_idx]
                top_vals  = enc_raw[sorted_idx].tolist()

            plt.figure(figsize=(10, 6))
            plt.barh(range(n_show), top_vals[::-1])
            plt.yticks(range(n_show), top_names[::-1])
            plt.xlabel("Variable Importance")
            plt.title(f"TFT Encoder VSN — Top {n_show} Variables ({EXPERIMENT_NAME})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "vsn_importances.png"), dpi=150)
            plt.close()

        print("VSN importances saved.")
    except Exception as e:
        import traceback
        print(f"Warning: VSN extraction failed: {e}")
        with open(os.path.join(out_dir, "vsn_importances_error.txt"), "w") as f:
            traceback.print_exc(file=f)


# ── Retrain best model ──
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
    persistent_workers=False,
    pin_memory=PIN_MEMORY,
)
val_loader_best = val_ds_best.to_dataloader(
    train=False,
    batch_size=best_batch,
    num_workers=NUM_WORKERS,
    persistent_workers=False,
    pin_memory=PIN_MEMORY,
)
test_loader_best = test_ds_best.to_dataloader(
    train=False,
    batch_size=best_batch,
    num_workers=NUM_WORKERS,
    persistent_workers=False,
    pin_memory=PIN_MEMORY,
)

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

# Load best checkpoint and evaluate
with patch_torch_load():
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt_path)

metrics = evaluate_tft_baseline(
    model=best_model,
    test_loader=test_loader_best,
    quantiles=QUANTILES,
    plot=True,
    save_path=os.path.join(RUN_DIR, "pred_vs_actual.png"),
)

safe_json_dump(
    {"best_val_loss": best_val, "best_checkpoint": best_ckpt_path, "test_metrics": metrics},
    os.path.join(RUN_DIR, "final_test_metrics.json"),
)

# Save training curves
save_training_curves_from_csv(log_dir, RUN_DIR)

# Save VSN importances
save_vsn_importances(best_model, val_loader_best, RUN_DIR)

print("Test metrics:", metrics)
print("Saved best-run artifacts to:", RUN_DIR)

# cleanup GPU memory
try:
    torch.cuda.empty_cache()
except Exception:
    pass
