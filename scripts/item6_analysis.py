from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from data_utils import load_data, prepare_main_dataframe


def run_item6(csv_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    df_main = prepare_main_dataframe(df)

    model = smf.logit(
        "academic_affected ~ usage_c + sleep_c + usage_sleep_c + Age",
        data=df_main,
    ).fit(disp=False)

    results = pd.DataFrame(
        {
            "Coefficient": model.params,
            "p_value": model.pvalues,
            "Odds_Ratio": np.exp(model.params),
            "Direction": np.where(model.params > 0, "Positive", "Negative"),
            "Significant_5pct": model.pvalues < 0.05,
        }
    )
    results.to_csv(output_dir / "item6_results.csv", index=True)

    with open(output_dir / "item6_summary.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    usage_range = np.linspace(df_main["usage_c"].min(), df_main["usage_c"].max(), 100)
    sleep_levels = [
        df_main["sleep_c"].quantile(0.25),
        df_main["sleep_c"].quantile(0.50),
        df_main["sleep_c"].quantile(0.75),
    ]
    sleep_labels = ["Low Sleep", "Average Sleep", "High Sleep"]

    pred_list = []
    for sleep_val, label in zip(sleep_levels, sleep_labels):
        temp = pd.DataFrame(
            {
                "usage_c": usage_range,
                "sleep_c": sleep_val,
                "usage_sleep_c": usage_range * sleep_val,
                "Age": df_main["Age"].mean(),
            }
        )
        temp["pred_prob"] = model.predict(temp)
        temp["Sleep_Level"] = label
        pred_list.append(temp)

    pred_df = pd.concat(pred_list, ignore_index=True)
    pred_df.to_csv(output_dir / "item6_predictions.csv", index=False)

    plt.style.use("ggplot")
    plt.figure(figsize=(8, 5))
    for label in sleep_labels:
        subset = pred_df[pred_df["Sleep_Level"] == label]
        plt.plot(subset["usage_c"], subset["pred_prob"], linewidth=2, label=label)

    plt.xlabel("Centered Daily Social Media Usage")
    plt.ylabel("Predicted Probability Academic Performance Affected")
    plt.title("Item 6: Predicted Academic Impact by Usage and Sleep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "item6_plot.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to input CSV dataset")
    parser.add_argument("--output", default="output", help="Directory to store outputs")
    args = parser.parse_args()

    run_item6(args.data, args.output)
