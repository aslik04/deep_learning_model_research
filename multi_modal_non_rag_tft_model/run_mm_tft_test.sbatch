#!/bin/bash
#SBATCH --job-name=mm_tft_test
#SBATCH --partition=gecko
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=3,5
#SBATCH --output=slurm_logs/mm_tft_test_%A_%a.out
#SBATCH --error=slurm_logs/mm_tft_test_%A_%a.err

# ──────────────────────────────────────────────────────────────
# TEST RUN: Tasks 3 & 5 with FP16 - verify no OOM
# Quick test with 3 trials, 5 epochs
# ──────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p slurm_logs

SCRIPTS=(
    ""                               # 0
    ""                               # 1
    ""                               # 2
    mm_price_allnews_tft.py          # 3: Price + AllNews (91 news cols) - FP16
    ""                               # 4
    mm_price_numeric_allnews_tft.py  # 5: Price + Numeric + AllNews (6 num + 91 news) - FP16
    ""                               # 6
)

IDX=${SLURM_ARRAY_TASK_ID}
SCRIPT=${SCRIPTS[$IDX]}

echo "============================================================"
echo "TEST RUN: SLURM Job ${SLURM_JOB_ID}, Task ${IDX}"
echo "Script:    ${SCRIPT}"
echo "Node:      $(hostname)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-none}"
echo "Time:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# Activate environment
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
module load python-ml/22-12-21-python3.9

source /dcs/23/u5561033/deep_learning_news_data/.venv/bin/activate

# Ensure venv packages override module packages
VENV_SP="/dcs/23/u5561033/deep_learning_news_data/.venv/lib64/python3.9/site-packages"
VENV_SP2="/dcs/23/u5561033/deep_learning_news_data/.venv/lib/python3.9/site-packages"
MODULE_SP="/local/java/python-ml/22-12-21-python3.9/lib/python3.9/site-packages"
MODULE_SP2="/local/java/python-ml/22-12-21-python3.9/lib64/python3.9/site-packages"
export PYTHONPATH="${VENV_SP}:${VENV_SP2}:${MODULE_SP}:${MODULE_SP2}"

# SHORT TEST CONFIG
export N_TRIALS="3"
export MAX_EPOCHS="5"
export PATIENCE="3"
export BATCH_SIZE="32"
export SKIP_BASELINE_FIT="1"

cd "${SLURM_SUBMIT_DIR}"

echo "Running TEST: python ${SCRIPT}"
echo "N_TRIALS=${N_TRIALS}, MAX_EPOCHS=${MAX_EPOCHS}, PATIENCE=${PATIENCE}"
echo "============================================================"

nvidia-smi

python "${SCRIPT}"

echo "============================================================"
echo "TEST PASSED: ${SCRIPT} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
