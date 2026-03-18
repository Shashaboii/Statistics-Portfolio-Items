# Portfolio Items 6–9 Reproducibility Repository

This repository contains plain Python scripts that reproduce the analyses for portfolio Items 6, 7, 8, and 9 from the **Students Social Media Addiction** dataset.

## Repository structure

- `scripts/data_utils.py` — data loading and preprocessing
- `scripts/item6_analysis.py` — logistic regression with interaction term
- `scripts/item7_comparison.py` — OLS vs logistic regression comparison
- `scripts/item8_model_selection.py` — likelihood-based model comparison
- `scripts/item9_causal_adjustment.py` — crude vs adjusted causal-style analysis
- `scripts/run_all.py` — runs all analyses and saves outputs
- `requirements.txt` — Python dependencies

## Expected dataset

Place the dataset CSV file here:

```text
data/Students Social Media Addiction.csv
```

The scripts assume the dataset contains at least these columns:

- `Age`
- `Gender`
- `Academic_Level`
- `Avg_Daily_Usage_Hours`
- `Sleep_Hours_Per_Night`
- `Affects_Academic_Performance`

## How to run

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Run the full pipeline:

```bash
python scripts/run_all.py --data "data/Students Social Media Addiction.csv" --output output
```

## Outputs

The pipeline writes plots and summary tables to the `output/` directory, including:

- `item6_results.csv`
- `item6_plot.png`
- `item7_comparison.csv`
- `item7_plot.png`
- `item8_model_comparison.csv`
- `item8_lr_tests.txt`
- `item8_plot.png`
- `item9_crude_summary.txt`
- `item9_adjusted_summary.txt`
- `item9_effect_estimates.txt`
- `item9_plot.png`

## Notes

- These scripts are a direct script-based rewrite of the original notebook workflow.
- The notebook itself is intentionally **not required** for reproduction.
- For submission, replace the placeholder repository link in the report with your public GitHub or GitLab URL if you publish this repository online.
