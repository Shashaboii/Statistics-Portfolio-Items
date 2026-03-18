from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from data_utils import load_data, prepare_main_dataframe


def run_item7(csv_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    df_main = prepare_main_dataframe(df)

    ols_model = smf.ols(
        "academic_affected ~ usage_c + sleep_c + usage_sleep_c + Age",
        data=df_main,
    ).fit()

    logit_model = smf.logit(
        "academic_affected ~ usage_c + sleep_c + usage_sleep_c + Age",
        data=df_main,
    ).fit(disp=False)

    comparison = pd.DataFrame(
        {
            "OLS_coef": ols_model.params,
            "OLS_pvalue": ols_model.pvalues,
            "Logit_coef": logit_model.params,
            "Logit_pvalue": logit_model.pvalues,
            "Odds_ratio": np.exp(logit_model.params),
            "OLS_direction": np.where(ols_model.params > 0, "Positive", "Negative"),
            "Logit_direction": np.where(logit_model.params > 0, "Positive", "Negative"),
            "OLS_significant_5pct": ols_model.pvalues < 0.05,
            "Logit_significant_5pct": logit_model.pvalues < 0.05,
        }
    )
    comparison.to_csv(output_dir / "item7_comparison.csv", index=True)

    with open(output_dir / "item7_ols_summary.txt", "w", encoding="utf-8") as f:
        f.write(ols_model.summary().as_text())

    with open(output_dir / "item7_logit_summary.txt", "w", encoding="utf-8") as f:
        f.write(logit_model.summary().as_text())

    df_main["pred_ols"] = ols_model.predict(df_main)
    df_main["pred_logit"] = logit_model.predict(df_main)

    plt.style.use("ggplot")
    plt.figure(figsize=(6, 5))
    plt.scatter(df_main["pred_ols"], df_main["pred_logit"], alpha=0.5)
    plt.xlabel("OLS Predicted Value")
    plt.ylabel("Logistic Predicted Probability")
    plt.title("Item 7: OLS vs Logistic Predictions")
    plt.tight_layout()
    plt.savefig(output_dir / "item7_plot.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to input CSV dataset")
    parser.add_argument("--output", default="output", help="Directory to store outputs")
    args = parser.parse_args()

    run_item7(args.data, args.output)
