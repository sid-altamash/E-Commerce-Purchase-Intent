"""Validated data ingestion and behavioral feature engineering."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FeatureSpec, ProjectConfig

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the session dataset violates the configured schema."""


def load_data(
    path: Path | None = None,
    config: ProjectConfig | None = None,
) -> pd.DataFrame:
    """Read the source CSV and fail explicitly when its schema is invalid."""
    settings = config or ProjectConfig.load()
    data_path = path or settings.resolve(settings.data["path"])
    if not data_path.is_file():
        raise FileNotFoundError(f"Session dataset not found: {data_path}")

    try:
        frame = pd.read_csv(data_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        logger.exception("Unable to read session dataset at %s", data_path)
        raise DataValidationError(f"Could not parse session dataset: {exc}") from exc

    spec = FeatureSpec.from_config(settings)
    required = set(spec.input_features) | {spec.target}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"Dataset is missing required columns: {missing}")
    if frame.empty:
        raise DataValidationError("Dataset contains no session rows.")

    for column in spec.numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[spec.target] = pd.to_numeric(frame[spec.target], errors="coerce")
    invalid_target = frame[spec.target].isna() | ~frame[spec.target].isin([0, 1])
    if invalid_target.any():
        raise DataValidationError(
            f"Target '{spec.target}' must contain only binary values 0 and 1."
        )
    frame[spec.target] = frame[spec.target].astype("int8")

    for column in spec.categorical:
        frame[column] = frame[column].astype("string").fillna("Unknown")

    if frame.loc[:, spec.numeric].isna().all(axis=None):
        raise DataValidationError("All configured numeric inputs are missing.")

    logger.info(
        "Loaded %s rows and %s columns from %s",
        len(frame),
        len(frame.columns),
        data_path,
    )
    return frame


def engineer_features(
    frame: pd.DataFrame,
    spec: FeatureSpec | None = None,
) -> pd.DataFrame:
    """Create stable, interpretable engagement features without using the target."""
    feature_spec = spec or FeatureSpec.from_config(ProjectConfig.load())
    missing = set(feature_spec.input_features).difference(frame.columns)
    if missing:
        raise DataValidationError(
            f"Cannot engineer features; inputs are missing: {sorted(missing)}"
        )

    result = frame.copy()
    total_pages = (
        result["Administrative"] + result["Informational"] + result["ProductRelated"]
    )
    total_duration = (
        result["Administrative_Duration"]
        + result["Informational_Duration"]
        + result["ProductRelated_Duration"]
    )
    result["TotalPages"] = total_pages
    result["TotalDuration"] = total_duration
    result["ProductPageShare"] = np.divide(
        result["ProductRelated"],
        total_pages,
        out=np.zeros(len(result), dtype=float),
        where=total_pages.to_numpy() > 0,
    )
    result["ProductSecondsPerPage"] = np.divide(
        result["ProductRelated_Duration"],
        result["ProductRelated"],
        out=np.zeros(len(result), dtype=float),
        where=result["ProductRelated"].to_numpy() > 0,
    )
    result["ExitBounceGap"] = result["ExitRates"] - result["BounceRates"]

    for column in feature_spec.categorical:
        result[column] = result[column].astype("string").fillna("Unknown")
    return result
