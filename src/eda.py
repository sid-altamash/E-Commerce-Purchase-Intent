"""Reusable exploratory summaries and charts for the supplied sessions."""

from __future__ import annotations

import pandas as pd

from src.config import ProjectConfig
from src.data import load_data


def conversion_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("Converted", observed=True)
        .agg(
            sessions=("Converted", "size"),
            page_value=("PageValues", "median"),
            product_pages=("ProductRelated", "median"),
            exit_rate=("ExitRates", "median"),
            product_time=("ProductRelated_Duration", "median"),
        )
        .rename(index={0: "No purchase", 1: "Purchase"})
    )


def monthly_conversion(frame: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    month_order = config.data["months"]
    result = frame.groupby("Month", observed=True)["Converted"].agg(
        sessions="size", conversions="sum", conversion_rate="mean"
    )
    result["month_order"] = result.index.map(
        {month: position for position, month in enumerate(month_order)}
    )
    return result.sort_values("month_order").drop(columns="month_order").reset_index()


if __name__ == "__main__":
    data = load_data()
    print("Conversion rate:", f"{data['Converted'].mean():.1%}")
    print(conversion_summary(data))
