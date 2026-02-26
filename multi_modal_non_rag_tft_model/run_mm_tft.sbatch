#!/bin/bash
#SBATCH --job-name=mm_tft
#SBATCH --partition=gecko
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --array=0-6%2
#SBATCH --output=slurm_logs/mm_tft_%A_%a.out
#SBATCH --error=slurm_logs/mm_tft_%A_%a.err

# ──────────────────────────────────────────────────────────────
# Multimodal Non-RAG TFT — FinBERT News-Enhanced Ablation Suite
# Array index → script mapping (7 experiments, max 2 concurrent)
# ──────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p slurm_logs

SCRIPTS=(
    mm_price_global_tft.py          # 0: Price + Global (7 news cols)
    mm_price_demand_tft.py          # 1: Price + Demand (28 news cols)
    mm_price_supply_tft.py          # 2: Price + Supply (28 news cols)
    mm_price_allnews_tft.py         # 3: Price + AllNews (91 news cols)
    mm_price_numeric_global_tft.py  # 4: Price + Numeric + Global (6 num + 7 news)
    mm_price_numeric_allnews_tft.py # 5: Price + Numeric + AllNews (6 num + 91 news)
    mm_price_shuffled_news_tft.py   # 6: Price + Shuffled News (permutation control)
)

EXP_NAMES=(
    price_global_news
    price_demand_news
    price_supply_news
    price_allnews
    price_numeric_global_news
    price_numeric_allnews
    price_shuffled_news
)

IDX=${SLURM_ARRAY_TASK_ID}
SCRIPT=${SCRIPTS[$IDX]}
EXP=${EXP_NAMES[$IDX]}

echo "============================================================"
echo "SLURM Job: ${SLURM_JOB_ID}, Array Task: ${IDX}"
echo "Script:    ${SCRIPT}"
echo "Experiment: ${EXP}"
echo "Node:      $(hostname)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-none}"
echo "Time:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# Activate environment
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
module load python-ml/22-12-21-python3.9
# Note: python-ml auto-loads CUDA/11.6 (matches torch 1.13.1+cu117)
source /dcs/23/u5561033/deep_learning_news_data/.venv/bin/activate

# Ensure venv packages override module packages (numpy compat fix)
VENV_SP="/dcs/23/u5561033/deep_learning_news_data/.venv/lib64/python3.9/site-packages"
VENV_SP2="/dcs/23/u5561033/deep_learning_news_data/.venv/lib/python3.9/site-packages"
MODULE_SP="/local/java/python-ml/22-12-21-python3.9/lib/python3.9/site-packages"
MODULE_SP2="/local/java/python-ml/22-12-21-python3.9/lib64/python3.9/site-packages"
export PYTHONPATH="${VENV_SP}:${VENV_SP2}:${MODULE_SP}:${MODULE_SP2}"

# Environment overrides for HPC
export EXP_NAME="${EXP}"
export MAX_EPOCHS="${MAX_EPOCHS:-60}"
export PATIENCE="${PATIENCE:-8}"
export N_TRIALS="${N_TRIALS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export SKIP_BASELINE_FIT="1"

cd "${SLURM_SUBMIT_DIR}"

echo "Running: python ${SCRIPT}"
echo "N_TRIALS=${N_TRIALS}, MAX_EPOCHS=${MAX_EPOCHS}, PATIENCE=${PATIENCE}"
echo "============================================================"

python "${SCRIPT}"

echo "============================================================"
echo "Finished: ${SCRIPT} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
