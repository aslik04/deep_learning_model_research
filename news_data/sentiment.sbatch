#!/bin/bash
#
#SBATCH --job-name=finbert-gpu
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=finbert_%j.out
#SBATCH --error=finbert_%j.err

cd /dcs/23/u5561033/deep_learning_news_data/news_data

source .venv/bin/activate

# Ensure deps are present (quick no-op if already installed by test job)
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --quiet transformers pandas pyarrow tqdm

echo "=== Starting Full FinBERT Inference ==="
python3 apply_finbert_gpu.py

echo "=== Job Complete ==="
