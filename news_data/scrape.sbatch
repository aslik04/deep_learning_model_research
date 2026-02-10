#!/bin/bash
#
#SBATCH --job-name=gdelt-scrape
#SBATCH --partition=tiger      # Use the high-core AMD EPYC nodes
#SBATCH --cpus-per-task=64     # Request 64 cores (plenty for 100 threads)
#SBATCH --mem=32G              # 32GB RAM is safe for text scraping
#SBATCH --time=24:00:00        # 24 hour limit (safe for 700k rows)
#SBATCH --output=scrape_%j.out # Standard output log
#SBATCH --error=scrape_%j.err  # Error log

# 1. Setup Virtual Environment (Created once, reused on future runs)
#    python3.12 is available system-wide; no module load needed.
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv .venv
fi

source .venv/bin/activate

# 2. Install / update dependencies (runs every time to catch any missing pkgs)
pip install --upgrade pip
pip install google-cloud-bigquery db-dtypes google-cloud-bigquery-storage \
            pandas newspaper3k pyarrow lxml_html_clean

# 3. Run the Scraper
echo "Starting Scraper on Tiger Node..."
python fetch_gdelt_energy_news_parallel.py
echo "Job Complete."
