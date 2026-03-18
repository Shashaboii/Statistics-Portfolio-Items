from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import chi2

from data_utils import load_data, prepare_main_dataframe


def run_item8(csv_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    df_main = prepare_main_dataframe(df)

    model1 = smf.logit(
        "academic_affected ~ usage_c + sleep_c",
        data=df_main,
    ).fit(disp=False)

    model2 = smf.logit(
        "academic_affected ~ usage_c + sleep_c + usage_sleep_c",
        data=df_main,
    ).fit(disp=False)

    model3 = smf.logit(
        "academic_affected ~ usage_c + sleep_c + usage_sleep_c + Age",
        data=df_main,
    ).fit(disp=False)

    comparison = pd.DataFrame(
        {
            "Model": ["Baseline", "Interaction", "Extended"],
            "Parameters": [
                int(model1.df_model + 1),
                int(model2.df_model + 1),
                int(model3.df_model + 1),
            ],
            "LogLikelihood": [model1.llf, model2.llf, model3.llf],
            "AIC": [model1.aic, model2.aic, model3.aic],
            "BIC": [model1.bic, model2.bic, model3.bic],
        }
    )
    comparison.to_csv(output_dir / "item8_model_comparison.csv", index=False)

    lr_stat_12 = 2 * (model2.llf - model1.llf)
    df_12 = int(model2.df_model - model1.df_model)
    p_12 = chi2.sf(lr_stat_12, df_12)

    lr_stat_23 = 2 * (model3.llf - model2.llf)
    df_23 = int(model3.df_model - model2.df_model)
    p_23 = chi2.sf(lr_stat_23, df_23)

    with open(output_dir / "item8_lr_tests.txt", "w", encoding="utf-8") as f:
        f.write(f"Model 1 vs Model 2 LR stat: {lr_stat_12:.6f}\n")
        f.write(f"Degrees of freedom: {df_12}\n")
        f.write(f"p-value: {p_12:.6g}\n\n")
        f.write(f"Model 2 vs Model 3 LR stat: {lr_stat_23:.6f}\n")
        f.write(f"Degrees of freedom: {df_23}\n")
        f.write(f"p-value: {p_23:.6g}\n")

    usage_range = np.linspace(df_main["usage_c"].min(), df_main["usage_c"].max(), 100)
    sleep_low = df_main["sleep_c"].quantile(0.25)
    sleep_high = df_main["sleep_c"].quantile(0.75)
    age_mean = df_main["Age"].mean()

    predictions = []
    for sleep_level, label in [(sleep_low, "Low Sleep"), (sleep_high, "High Sleep")]:
        temp = pd.DataFrame(
            {
                "usage_c": usage_range,
                "sleep_c": sleep_level,
                "usage_sleep_c": usage_range * sleep_level,
                "Age": age_mean,
            }
        )
        temp["baseline"] = model1.predict(temp)
        temp["interaction"] = model2.predict(temp)
        temp["extended"] = model3.predict(temp)
        temp["Sleep"] = label
        predictions.append(temp)

    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df.to_csv(output_dir / "item8_predictions.csv", index=False)

    plt.style.use("ggplot")
    plt.figure(figsize=(9, 6))
    for model_name in ["baseline", "interaction", "extended"]:
        for sleep_label, line_style in [("Low Sleep", "--"), ("High Sleep", "-")]:
            subset = pred_df[pred_df["Sleep"] == sleep_label]
            plt.plot(
                subset["usage_c"],
                subset[model_name],
                linestyle=line_style,
                linewidth=2,
                label=f"{model_name.capitalize()} ({sleep_label})",
            )

    plt.xlabel("Centered Daily Social Media Usage")
    plt.ylabel("Predicted Probability Academic Performance Affected")
    plt.title("Item 8: Predicted Probability Comparison Across Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "item8_plot.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to input CSV dataset")
    parser.add_argument("--output", default="output", help="Directory to store outputs")
    args = parser.parse_args()

    run_item8(args.data, args.output)
