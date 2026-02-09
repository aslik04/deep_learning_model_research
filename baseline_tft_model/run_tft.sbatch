#!/bin/bash
#SBATCH --job-name=tft-models
#SBATCH --partition=gecko
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0,1%3
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=joboutput_%A_%a.out
#SBATCH --error=joboutput_%A_%a.err

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Activate virtual environment ##
cd "/dcs/23/u5561033/CS310 - 3rd Year Project/Baseline TFT Model"
source ".venv/bin/activate"

## Load CUDA ##
module load CUDA/12.2

## Define scripts + experiment names (used to make unique Optuna DBs / study names) ##
SCRIPTS=(
  # "baseline_price_only_tft.py"
  # "baseline_price_prod_tft.py"
  "baseline_price_storage_tft.py"
  "baseline_price_usd_tft.py"
  # "baseline_price_weather.py"
  # "baseline_price_prod_stor_usd_weather_tft.py"
)

EXPS=(
  # "price_only"
  # "price_prod"
  "price_storage"
  "price_usd"
  # "price_weather"
  # "price_prod_stor_usd_weather"
)

SCRIPT="${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"
EXP_NAME="${EXPS[$SLURM_ARRAY_TASK_ID]}"

export EXP_NAME
export SLURM_ARRAY_TASK_ID

echo "=============================================="
echo "Job Array Task: $SLURM_ARRAY_TASK_ID"
echo "Experiment:     $EXP_NAME"
echo "Running:        $SCRIPT"
echo "Host:           $(hostname)"
echo "Started at:     $(date)"
echo "=============================================="

python "$SCRIPT"

echo "=============================================="
echo "Completed: $SCRIPT at $(date)"
echo "=============================================="
