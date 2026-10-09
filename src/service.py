"""Shared artifact loading and prediction logic used by the API and dashboard."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import FeatureSpec, ProjectConfig
from src.data import engineer_features

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_bundle() -> tuple[Any, dict[str, Any], ProjectConfig, FeatureSpec]:
    config = ProjectConfig.load()
    model_path = config.resolve(config.paths["artifact"])
    metadata_path = config.resolve(config.paths["metadata"])
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Trained model artifacts are missing. Run "
            "'python -m src.train' before requesting predictions."
        )
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata, config, FeatureSpec.from_config(config)


def predict_session(
    session: dict[str, Any],
    *,
    include_explanations: bool = True,
) -> dict[str, Any]:
    model, metadata, _, spec = load_bundle()
    raw = pd.DataFrame([{name: session[name] for name in spec.input_features}])
    for column in spec.categorical:
        raw[column] = raw[column].astype("string")
    model_input = engineer_features(raw, spec).loc[:, spec.model_features]
    probability = float(model.predict_proba(model_input)[0, 1])
    threshold = float(metadata["threshold"])
    result: dict[str, Any] = {
        "prediction": int(probability >= threshold),
        "label": "Likely to purchase" if probability >= threshold else "Unlikely to purchase",
        "conversion_probability": probability,
        "decision_threshold": threshold,
        "confidence": max(probability, 1 - probability),
        "model": metadata["best_model"],
        "contributors": [],
    }
    if include_explanations:
        from src.explain import local_shap_values

        try:
            contributions = local_shap_values(model, model_input).iloc[0]
            result["contributors"] = [
                {"feature": str(name), "impact": float(value)}
                for name, value in contributions.abs().sort_values(ascending=False)
                .head(5)
                .items()
            ]
            result["explanation_method"] = "SHAP"
        except (ImportError, ValueError, TypeError, AttributeError) as exc:
            logger.warning("SHAP explanation unavailable for this prediction: %s", exc)
            result["explanation_method"] = "Unavailable"
    return result


def load_feature_importance() -> pd.DataFrame:
    config = ProjectConfig.load()
    path: Path = config.resolve(config.paths["feature_importance"])
    if not path.is_file():
        return pd.DataFrame(columns=["feature", "permutation_importance"])
    return pd.read_csv(path)
