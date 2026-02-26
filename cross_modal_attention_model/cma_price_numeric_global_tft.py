# %% [markdown]
# ## Price + Numeric + Global News — Cross-Modal Attention Model (Henry Hub)
#
# Dual-encoder cross-modal attention with modality gating.
# Numeric stream (price + 6 covariates) queries news stream (7 global FinBERT features).
# Ablation toggles: use_cross_attention, use_gate.

# %% [markdown]
# ### Imports and Configuration

# %%
import os, json, time, platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from sklearn.metrics import mean_absolute_error, mean_squared_error

from cross_modal_modules import CrossModalAttentionForecaster

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

# Ablation toggles (default: full model)
ENV_USE_CROSS_ATTENTION = os.environ.get("USE_CROSS_ATTENTION", "1") == "1"
ENV_USE_GATE = os.environ.get("USE_GATE", "1") == "1"

TASK_ID = os.environ.get("SLURM_ARRAY_TASK_ID", "local")

# ------------------------------------------------------------------
# Run configuration
# ------------------------------------------------------------------
_attn_tag = "attn" if ENV_USE_CROSS_ATTENTION else "noattn"
_gate_tag = "+gate" if (ENV_USE_CROSS_ATTENTION and ENV_USE_GATE) else ""
EXPERIMENT_NAME = f"Price+Numeric+GlobalNews_CMA_{_attn_tag}{_gate_tag}"
MODEL_TAG = "cma"
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

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
NUM_WORKERS = max(0, min(4, SLURM_CPUS - 1))
PIN_MEMORY = torch.cuda.is_available()

print("=" * 60)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print(f"SLURM_CPUS = {SLURM_CPUS}")
print(f"NUM_WORKERS = {NUM_WORKERS}, PIN_MEMORY = {PIN_MEMORY}")
print(f"ENV_MAX_EPOCHS={ENV_MAX_EPOCHS}, ENV_PATIENCE={ENV_PATIENCE}, ENV_N_TRIALS={ENV_N_TRIALS}")
print(f"USE_CROSS_ATTENTION={ENV_USE_CROSS_ATTENTION}, USE_GATE={ENV_USE_GATE}")
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

# %%
CSV = "../numeric_data/henryhub_master.csv"
NEWS_PARQUET = "../news_data/daily_energy_signals_enhanced.parquet"
TARGET = "price"
GROUP_COL = "id"

NUM_COLS = ["storage_bcf", "production_bcf", "usd_index", "temp_c", "temp_max_c", "temp_min_c"]

NEWS_COLS = [
    "Global_Vol", "Global_Avg_Pos", "Global_Avg_Neg", "Global_Avg_Neu",
    "Global_Max_Pos", "Global_Max_Neg", "Global_Dispersion",
]

COVARIATE_COLS = NUM_COLS + NEWS_COLS

# Number of features per modality (used by the model to split encoder_cont)
N_NUMERIC_FEATURES = 1 + len(NUM_COLS)   # price + 6 = 7
N_NEWS_FEATURES = len(NEWS_COLS)          # 7


def load_tft_ready_df(csv_path: str, news_path: str, news_cols: list) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values([GROUP_COL, "date"]).reset_index(drop=True)

    news = pd.read_parquet(news_path).reset_index()
    news["date"] = pd.to_datetime(news["date"])
    news = news[["date"] + news_cols]
    df = df.merge(news, on="date", how="left")

    for c in news_cols:
        df[c] = df[c].fillna(0.0).astype("float32")

    df[GROUP_COL] = df[GROUP_COL].astype("category")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype("float32")

    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    df["time_idx"] = df.groupby(GROUP_COL).cumcount().astype(np.int64)
    df["dow"] = df["date"].dt.dayofweek.astype("category")
    df["month"] = df["date"].dt.month.astype("category")

    assert df[[GROUP_COL, "time_idx"]].duplicated().sum() == 0
    assert df[TARGET].isna().sum() == 0

    return df


data = load_tft_ready_df(CSV, NEWS_PARQUET, NEWS_COLS)
print(f"Loaded {len(data)} rows with {len(COVARIATE_COLS)} covariates + price")

safe_json_dump({
    "csv_path": CSV,
    "news_parquet_path": NEWS_PARQUET,
    "experiment_name": EXPERIMENT_NAME,
    "news_cols": NEWS_COLS,
    "num_cols": NUM_COLS,
    "n_rows": len(data),
    "date_range": [str(data.date.min().date()), str(data.date.max().date())],
    "n_news_features": N_NEWS_FEATURES,
    "n_numeric_features": N_NUMERIC_FEATURES,
    "news_fill_strategy": "zero",
    "use_cross_attention": ENV_USE_CROSS_ATTENTION,
    "use_gate": ENV_USE_GATE,
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


def build_tft_datasets(df, lookback, horizon, train_cutoff, val_cutoff):
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
        min_prediction_idx=lookback,
    )
    val_ds = TimeSeriesDataSet(
        df[df["time_idx"] <= val_cutoff].copy(),
        **common_args,
        min_prediction_idx=train_cutoff + 1,
    )
    test_ds = TimeSeriesDataSet(
        df.copy(),
        **common_args,
        min_prediction_idx=val_cutoff + 1,
    )
    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = build_tft_datasets(
    df=data, lookback=LOOKBACK, horizon=HORIZON,
    train_cutoff=train_cutoff, val_cutoff=val_cutoff,
)
print("Samples | train:", len(train_ds), "| val:", len(val_ds), "| test:", len(test_ds))

# ── Column ordering assertion ──
expected_unknowns = ["price"] + NUM_COLS + NEWS_COLS
# reals includes internal TFT columns before user features; find the offset
print(f"Dataset reals: {train_ds.reals}")
NUMERIC_OFFSET = train_ds.reals.index("price")  # first user feature
print(f"Numeric features start at reals index {NUMERIC_OFFSET}")
for i, name in enumerate(expected_unknowns):
    actual = train_ds.reals[NUMERIC_OFFSET + i]
    assert actual == name, \
        f"Column order mismatch at offset {i}: expected '{name}', got '{actual}'"
print("Column ordering verified ✓")


# %% [markdown]
# ### Step 4 — DataLoaders

# %%
BATCH_SIZE = ENV_BATCH_SIZE

train_loader = train_ds.to_dataloader(
    train=True, batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS, persistent_workers=False, pin_memory=PIN_MEMORY,
)
val_loader = val_ds.to_dataloader(
    train=False, batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS, persistent_workers=False, pin_memory=PIN_MEMORY,
)
test_loader = test_ds.to_dataloader(
    train=False, batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS, persistent_workers=False, pin_memory=PIN_MEMORY,
)


# %% [markdown]
# ### Step 5 — Baseline Verification Run (skippable)

# %%
if not ENV_SKIP_BASELINE_FIT:
    print("Running baseline verification with CrossModalAttentionForecaster...")
    baseline_model = CrossModalAttentionForecaster(
        n_numeric_features=N_NUMERIC_FEATURES,
        n_news_features=N_NEWS_FEATURES,
        numeric_offset=NUMERIC_OFFSET,
        hidden_dim=32,
        n_heads=4,
        n_lstm_layers_numeric=1,
        n_lstm_layers_news=1,
        dropout=0.1,
        learning_rate=1e-3,
        quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        use_cross_attention=ENV_USE_CROSS_ATTENTION,
        use_gate=ENV_USE_GATE,
    )

    callbacks = [EarlyStopping(monitor="val_loss", patience=min(ENV_PATIENCE, 8), mode="min")]
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    trainer = pl.Trainer(
        max_epochs=min(ENV_MAX_EPOCHS, 10),
        accelerator=accelerator, devices=devices,
        callbacks=callbacks,
        enable_checkpointing=False, logger=False,
    )
    trainer.fit(baseline_model, train_loader, val_loader)
    print("Baseline verification complete.")
    del trainer, baseline_model
    gc.collect()


# %% [markdown]
# ### Step 6 — Evaluate CMA Model (Quantiles → Point Forecast Metrics)

# %%
def predict_and_denormalize(model, loader):
    """Run inference and de-normalize quantile predictions."""
    model.eval()
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for x, y in loader:
            # Move to model device
            device = next(model.parameters()).device
            x_numeric, x_news = model._split_features(x)
            x_numeric = x_numeric.to(device)
            x_news = x_news.to(device)

            preds_norm = model(x_numeric, x_news)  # [B, n_q] in normalised space

            # De-normalise using target_scale from the dataset
            center = x["target_scale"][:, 0:1].to(device)  # [B, 1]
            scale = x["target_scale"][:, 1:2].to(device)   # [B, 1]
            preds_real = preds_norm * scale + center        # [B, n_q]

            all_preds.append(preds_real.cpu().numpy())

            y_true = y[0] if isinstance(y, (tuple, list)) else y
            all_actuals.append(y_true.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)       # [N, n_q]
    actuals = np.concatenate(all_actuals, axis=0).reshape(-1)  # [N]
    return preds, actuals


def evaluate_cma_model(
    model, test_loader,
    quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
    plot=True, save_path=None, dpi=150,
    title="CMA Next-Day Forecast — Test Set",
):
    """Compute MAE/RMSE/MAPE/DA using P50 as point forecast."""
    preds_q, y_true = predict_and_denormalize(model, test_loader)

    q_list = list(quantiles)
    q50_idx = q_list.index(0.5)
    y_pred = preds_q[:, q50_idx]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)

    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    da = float((true_dir == pred_dir).mean() * 100)

    if plot:
        x_axis = np.arange(len(y_pred))
        if 0.05 in q_list and 0.95 in q_list:
            qlo = preds_q[:, q_list.index(0.05)]
            qhi = preds_q[:, q_list.index(0.95)]
        else:
            qlo = qhi = None

        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="Actual")
        plt.plot(y_pred, label="Pred (P50)")
        if qlo is not None:
            plt.fill_between(x_axis, qlo, qhi, alpha=0.2, label="P05-P95")
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
        "MAE": mae, "RMSE": rmse, "MAPE": mape,
        "Directional_Accuracy": da,
        "y_true": y_true, "y_pred_p50": y_pred,
    }


# %% [markdown]
# ### Step 7 — Hyperparameter Tuning (Optuna)

# %%
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
MAX_EPOCHS = ENV_MAX_EPOCHS
PATIENCE = ENV_PATIENCE

# ---- search space ----
ENCODER_CHOICES    = [20, 30, 45, 60, 90]
BATCH_CHOICES      = [32, 64, 128, 256]
CLIP_CHOICES       = [0.1, 0.25, 0.5, 1.0, 2.0]
HIDDEN_DIM_CHOICES = [32, 48, 64, 96, 128]
HEAD_CHOICES       = [1, 2, 4, 8]
NUM_LSTM_CHOICES   = [1, 2, 3]
# Frozen for first run:
NEWS_LSTM_LAYERS   = 1
GATE_TYPE          = "sigmoid"

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


def _valid_heads(hidden_dim: int):
    hs = [h for h in HEAD_CHOICES if hidden_dim % h == 0]
    return hs if hs else [1]


def _get_datasets_for_lookback(lookback: int):
    if lookback in _DATASET_CACHE:
        return _DATASET_CACHE[lookback]
    train_ds, val_ds, test_ds = build_tft_datasets(
        df=data, lookback=lookback, horizon=HORIZON,
        train_cutoff=train_cutoff, val_cutoff=val_cutoff,
    )
    _DATASET_CACHE[lookback] = (train_ds, val_ds, test_ds)
    return train_ds, val_ds, test_ds


def _make_loaders(train_ds, val_ds, test_ds, batch_size):
    tl = train_ds.to_dataloader(train=True, batch_size=batch_size,
                                num_workers=NUM_WORKERS, persistent_workers=False, pin_memory=PIN_MEMORY)
    vl = val_ds.to_dataloader(train=False, batch_size=batch_size,
                              num_workers=NUM_WORKERS, persistent_workers=False, pin_memory=PIN_MEMORY)
    sl = test_ds.to_dataloader(train=False, batch_size=batch_size,
                               num_workers=NUM_WORKERS, persistent_workers=False, pin_memory=PIN_MEMORY)
    return tl, vl, sl


def objective(trial: optuna.Trial):
    start_time = time.time()
    pl.seed_everything(SEED, workers=True)

    # ── Sample hyperparameters ──
    lookback   = trial.suggest_categorical("max_encoder_length", ENCODER_CHOICES)
    batch_size = trial.suggest_categorical("batch_size", BATCH_CHOICES)
    clip       = trial.suggest_categorical("gradient_clip_val", CLIP_CHOICES)
    hidden_dim = trial.suggest_categorical("hidden_dim", HIDDEN_DIM_CHOICES)

    n_heads_raw = trial.suggest_categorical("n_heads", HEAD_CHOICES)
    valid_heads = _valid_heads(hidden_dim)
    n_heads = max(h for h in valid_heads if h <= n_heads_raw) if any(
        h <= n_heads_raw for h in valid_heads) else valid_heads[0]

    n_lstm_numeric = trial.suggest_categorical("n_lstm_layers_numeric", NUM_LSTM_CHOICES)

    dropout     = trial.suggest_float("dropout", 0.0, 0.4)
    lr          = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    gate_lambda = trial.suggest_float("gate_lambda", 0.0, 0.01)

    wd_choice = trial.suggest_categorical("weight_decay", [0, 1e-6, 1e-5, 1e-4])
    wd = float(wd_choice)

    trial.set_user_attr("max_encoder_length", lookback)

    # ── Build datasets and loaders ──
    train_ds_t, val_ds_t, test_ds_t = _get_datasets_for_lookback(lookback)
    train_loader_t, val_loader_t, test_loader_t = _make_loaders(
        train_ds_t, val_ds_t, test_ds_t, batch_size)

    # ── Build model ──
    model = CrossModalAttentionForecaster(
        n_numeric_features=N_NUMERIC_FEATURES,
        n_news_features=N_NEWS_FEATURES,
        numeric_offset=NUMERIC_OFFSET,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_lstm_layers_numeric=n_lstm_numeric,
        n_lstm_layers_news=NEWS_LSTM_LAYERS,
        dropout=dropout,
        learning_rate=lr,
        weight_decay=wd,
        quantiles=QUANTILES,
        gradient_clip_val=clip,
        gate_lambda=gate_lambda,
        use_cross_attention=ENV_USE_CROSS_ATTENTION,
        use_gate=ENV_USE_GATE,
    )

    # ── Trainer ──
    prune_cb = OptunaPruningCallback(trial, monitor="val_loss")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"),
        prune_cb,
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accelerator, devices=devices,
        gradient_clip_val=float(clip),
        enable_checkpointing=False,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_loader_t, val_loader_t)

    best_val = prune_cb.best
    epochs_ran = trainer.current_epoch + 1
    duration_sec = float(time.time() - start_time)

    trial.set_user_attr("duration_sec", duration_sec)
    trial.set_user_attr("epochs_ran", epochs_ran)
    trial.set_user_attr("val_loss_best", float(best_val))

    # ── Test evaluation ──
    try:
        test_metrics = evaluate_cma_model(
            model=model, test_loader=test_loader_t,
            quantiles=QUANTILES, plot=False,
        )
        trial.set_user_attr("test_mae", float(test_metrics["MAE"]))
        trial.set_user_attr("test_rmse", float(test_metrics["RMSE"]))
        trial.set_user_attr("test_mape", float(test_metrics["MAPE"]))
        trial.set_user_attr("test_directional_accuracy", float(test_metrics["Directional_Accuracy"]))
    except Exception as e:
        print(f"[trial {trial.number}] Test eval failed: {e}")
        for k in ("test_mae", "test_rmse", "test_mape", "test_directional_accuracy"):
            trial.set_user_attr(k, None)

    # ── Store all params as user attrs ──
    for k, v in dict(
        lookback=lookback, batch_size=batch_size, hidden_dim=hidden_dim,
        n_heads=n_heads, n_lstm_layers_numeric=n_lstm_numeric,
        n_lstm_layers_news=NEWS_LSTM_LAYERS,
        dropout=dropout, learning_rate=lr, gradient_clip_val=clip,
        weight_decay=wd, gate_lambda=gate_lambda,
    ).items():
        trial.set_user_attr(k, float(v) if isinstance(v, float) else int(v))

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
def export_trial_summary(study, out_csv):
    wanted = [
        "duration_sec", "epochs_ran", "val_loss_best",
        "test_mae", "test_rmse", "test_mape", "test_directional_accuracy",
        "lookback", "batch_size", "hidden_dim", "n_heads",
        "n_lstm_layers_numeric", "n_lstm_layers_news",
        "dropout", "learning_rate", "gradient_clip_val",
        "weight_decay", "gate_lambda",
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

    completed_csv = out_csv.replace(".csv", "_completed.csv")
    df_all[df_all["state"] == "COMPLETE"].to_csv(completed_csv, index=False)
    print(f"Wrote completed trials: {completed_csv}")


def save_optuna_study_artifacts(study, out_dir):
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
# ### Step 9 — Retrain Best Model, Save Checkpoint + Interpretability Artifacts

# %%
def build_best_model(best_params: dict):
    """Build CMA model from best Optuna params."""
    hidden_dim = int(best_params["hidden_dim"])
    n_heads_raw = int(best_params["n_heads"])
    valid_heads = _valid_heads(hidden_dim)
    n_heads = max(h for h in valid_heads if h <= n_heads_raw) if any(
        h <= n_heads_raw for h in valid_heads) else valid_heads[0]

    return CrossModalAttentionForecaster(
        n_numeric_features=N_NUMERIC_FEATURES,
        n_news_features=N_NEWS_FEATURES,
        numeric_offset=NUMERIC_OFFSET,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_lstm_layers_numeric=int(best_params["n_lstm_layers_numeric"]),
        n_lstm_layers_news=NEWS_LSTM_LAYERS,
        dropout=float(best_params["dropout"]),
        learning_rate=float(best_params["learning_rate"]),
        weight_decay=float(best_params.get("weight_decay", 0)),
        quantiles=QUANTILES,
        gradient_clip_val=float(best_params.get("gradient_clip_val", 0.5)),
        gate_lambda=float(best_params.get("gate_lambda", 0.0)),
        use_cross_attention=ENV_USE_CROSS_ATTENTION,
        use_gate=ENV_USE_GATE,
    )


def fit_best_model(model, train_loader, val_loader, out_dir,
                   max_epochs=60, patience=8, gradient_clip_val=0.5):
    logger = CSVLogger(save_dir=out_dir, name="logs")
    ckpt = ModelCheckpoint(
        dirpath=os.path.join(out_dir, "checkpoints"),
        filename="best-{epoch:02d}-{val_loss:.6f}",
        monitor="val_loss", mode="min", save_top_k=1,
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator, devices=devices,
        gradient_clip_val=float(gradient_clip_val),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
            ckpt,
        ],
        logger=logger,
        enable_progress_bar=True, enable_model_summary=True,
    )
    trainer.fit(model, train_loader, val_loader)

    best_path = ckpt.best_model_path
    best_score = ckpt.best_model_score
    best_score = float(best_score.detach().cpu().item()) if best_score is not None else None
    return trainer, best_path, best_score, logger.log_dir


def save_training_curves_from_csv(log_dir, out_dir):
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

    # Gate statistics over training
    if "gate_mean" in m.columns:
        gg = m.dropna(subset=["gate_mean"]).groupby("epoch", as_index=False)["gate_mean"].last()
        plt.figure(figsize=(8, 4))
        plt.plot(gg["epoch"], gg["gate_mean"], label="gate_mean", color="orange")
        plt.title("Gate Mean Value Over Training")
        plt.xlabel("epoch"); plt.ylabel("mean gate value")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "gate_mean_over_training.png"), dpi=150)
        plt.close()


def extract_interpretability_artifacts(model, test_loader, out_dir, test_dates=None):
    """Extract attention heatmaps and gate values for all test samples."""
    model.eval()
    all_attn_weights = []
    all_gate_values = []
    all_preds = []
    all_actuals = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for x, y in test_loader:
            x_numeric, x_news = model._split_features(x)
            x_numeric = x_numeric.to(device)
            x_news = x_news.to(device)

            preds = model(x_numeric, x_news)
            all_preds.append(preds.cpu())

            if model._last_attn_weights is not None:
                all_attn_weights.append(model._last_attn_weights.cpu())
            if model._last_gate_values is not None:
                all_gate_values.append(model._last_gate_values.cpu())

            y_true = y[0] if isinstance(y, (tuple, list)) else y
            all_actuals.append(y_true.cpu())

    # Save raw artifacts
    artifacts = {}
    if all_attn_weights:
        attn_raw = torch.cat(all_attn_weights, dim=0)    # [N, n_heads, T, T]
        attn_avg = attn_raw.mean(dim=1)                   # [N, T, T]
        artifacts["attn_weights_per_head"] = attn_raw
        artifacts["attn_weights_avg"] = attn_avg
    if all_gate_values:
        gate_vals = torch.cat(all_gate_values, dim=0)     # [N, T, 1]
        artifacts["gate_values"] = gate_vals

    if artifacts:
        torch.save(artifacts, os.path.join(out_dir, "interpretability_artifacts.pt"))
        print(f"Saved interpretability artifacts: {list(artifacts.keys())}")

    # ── Gate analysis plots ──
    if all_gate_values:
        gate_vals = artifacts["gate_values"]
        avg_gate_per_sample = gate_vals.squeeze(-1).mean(dim=1).numpy()  # [N]

        # Plot 1: Gate value over test samples
        plt.figure(figsize=(12, 4))
        plt.plot(avg_gate_per_sample, alpha=0.7, linewidth=0.8)
        plt.title("Modality Gate Value (mean per sample) — Test Set")
        plt.xlabel("Test Sample Index")
        plt.ylabel("Gate Value (0=closed, 1=open)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "gate_values_over_time.png"), dpi=150)
        plt.close()

        # Summary stats
        stats = {
            "mean_gate_value": float(gate_vals.mean()),
            "std_gate_value": float(gate_vals.std()),
            "min_gate_value": float(gate_vals.min()),
            "max_gate_value": float(gate_vals.max()),
            "median_gate_value": float(gate_vals.median()),
            "sparsity_fraction_lt_0.1": float((gate_vals < 0.1).float().mean()),
        }
        safe_json_dump(stats, os.path.join(out_dir, "interpretability_stats.json"))
        print(f"Gate stats: mean={stats['mean_gate_value']:.4f}, "
              f"sparsity={stats['sparsity_fraction_lt_0.1']:.4f}")

    # ── Attention heatmaps for representative samples ──
    if all_attn_weights:
        attn_avg = artifacts["attn_weights_avg"]
        n_samples = len(attn_avg)
        sample_indices = [0, n_samples // 2, n_samples - 1]
        for idx in sample_indices:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(attn_avg[idx].numpy(), aspect="auto", cmap="viridis")
            ax.set_xlabel("News Timestep (Key)")
            ax.set_ylabel("Numeric Timestep (Query)")
            ax.set_title(f"Cross-Modal Attention (head-avg) — Sample {idx}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"attn_heatmap_sample_{idx}.png"), dpi=150)
            plt.close()


# ── Retrain best model ──
best = study.best_params
best_lookback = int(best["max_encoder_length"])
best_batch = int(best["batch_size"])

train_ds_best, val_ds_best, test_ds_best = build_tft_datasets(
    df=data, lookback=best_lookback, horizon=HORIZON,
    train_cutoff=train_cutoff, val_cutoff=val_cutoff,
)

train_loader_best = train_ds_best.to_dataloader(
    train=True, batch_size=best_batch,
    num_workers=NUM_WORKERS, persistent_workers=False, pin_memory=PIN_MEMORY,
)
val_loader_best = val_ds_best.to_dataloader(
    train=False, batch_size=best_batch,
    num_workers=NUM_WORKERS, persistent_workers=False, pin_memory=PIN_MEMORY,
)
test_loader_best = test_ds_best.to_dataloader(
    train=False, batch_size=best_batch,
    num_workers=NUM_WORKERS, persistent_workers=False, pin_memory=PIN_MEMORY,
)

best_model = build_best_model(best)

trainer, best_ckpt_path, best_val, log_dir = fit_best_model(
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
    best_model = CrossModalAttentionForecaster.load_from_checkpoint(best_ckpt_path)

metrics = evaluate_cma_model(
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

# Extract interpretability artifacts
extract_interpretability_artifacts(best_model, test_loader_best, RUN_DIR)

print("Test metrics:", metrics)
print("Saved best-run artifacts to:", RUN_DIR)

try:
    torch.cuda.empty_cache()
except Exception:
    pass
