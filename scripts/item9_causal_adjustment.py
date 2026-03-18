from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

from data_utils import load_data, prepare_main_dataframe, prepare_item9_dataframe


def run_item9(csv_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    df_item9, median_usage = prepare_item9_dataframe(df)
    df_main = prepare_main_dataframe(df)

    raw_means = df_item9.groupby("high_usage")["academic_affected"].mean()

    crude_model = smf.ols("academic_affected ~ high_usage", data=df_item9).fit()
    adjusted_model = smf.ols(
        "academic_affected ~ high_usage + Age + C(Gender) + C(Academic_Level)",
        data=df_item9,
    ).fit(cov_type="HC3")

    with open(output_dir / "item9_crude_summary.txt", "w", encoding="utf-8") as f:
        f.write(crude_model.summary().as_text())

    with open(output_dir / "item9_adjusted_summary.txt", "w", encoding="utf-8") as f:
        f.write(adjusted_model.summary().as_text())

    with open(output_dir / "item9_effect_estimates.txt", "w", encoding="utf-8") as f:
        f.write(f"Median usage cut-point: {median_usage:.6f}\n")
        f.write("Raw mean outcome by treatment group:\n")
        f.write(raw_means.to_string())
        f.write("\n\nCrude effect estimate:\n")
        f.write(str(crude_model.params["high_usage"]))
        f.write("\n\nAdjusted effect estimate:\n")
        f.write(str(adjusted_model.params["high_usage"]))
        f.write("\n")

    plt.style.use("ggplot")
    sns.lmplot(
        data=df_main,
        x="Avg_Daily_Usage_Hours",
        y="academic_affected",
        hue="Sleep_Group",
        height=5,
        aspect=1.5,
        scatter_kws={"alpha": 0.45},
    )
    plt.title("Academic Impact vs Social Media Usage by Sleep Group")
    plt.xlabel("Average Daily Usage Hours")
    plt.ylabel("Academic Performance Affected (0/1)")
    plt.tight_layout()
    plt.savefig(output_dir / "item9_plot.png", dpi=200)
    plt.close("all")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to input CSV dataset")
    parser.add_argument("--output", default="output", help="Directory to store outputs")
    args = parser.parse_args()

    run_item9(args.data, args.output)
