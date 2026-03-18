from __future__ import annotations

from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "Age",
    "Gender",
    "Academic_Level",
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Affects_Academic_Performance",
]


def load_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Put the CSV in the data/ folder or pass --data."
        )

    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def prepare_main_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_main = df[
        [
            "Age",
            "Gender",
            "Academic_Level",
            "Avg_Daily_Usage_Hours",
            "Sleep_Hours_Per_Night",
            "Affects_Academic_Performance",
        ]
    ].copy()

    df_main = df_main.dropna()

    df_main["academic_affected"] = (
        df_main["Affects_Academic_Performance"].astype(str).str.strip().str.lower() == "yes"
    ).astype(int)

    df_main["usage_c"] = (
        df_main["Avg_Daily_Usage_Hours"] - df_main["Avg_Daily_Usage_Hours"].mean()
    )
    df_main["sleep_c"] = (
        df_main["Sleep_Hours_Per_Night"] - df_main["Sleep_Hours_Per_Night"].mean()
    )
    df_main["usage_sleep_c"] = df_main["usage_c"] * df_main["sleep_c"]

    df_main["Sleep_Group"] = pd.cut(
        df_main["Sleep_Hours_Per_Night"],
        bins=[0, 5, 7, 10],
        labels=["Low Sleep", "Medium Sleep", "High Sleep"],
        include_lowest=True,
    )

    return df_main


def prepare_item9_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    df_item9 = df[
        [
            "Age",
            "Gender",
            "Academic_Level",
            "Avg_Daily_Usage_Hours",
            "Affects_Academic_Performance",
        ]
    ].copy().dropna()

    df_item9["academic_affected"] = (
        df_item9["Affects_Academic_Performance"].astype(str).str.strip().str.lower() == "yes"
    ).astype(int)

    median_usage = df_item9["Avg_Daily_Usage_Hours"].median()
    df_item9["high_usage"] = (
        df_item9["Avg_Daily_Usage_Hours"] >= median_usage
    ).astype(int)

    return df_item9, float(median_usage)
