"""Global and local explanations for the saved prediction pipeline."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _positive_class_values(values: Any) -> np.ndarray:
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[-1] >= 2:
        array = array[:, :, 1]
    return array


def local_shap_values(
    pipeline: Pipeline,
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    """Explain transformed features with SHAP and aggregate one-hot levels."""
    import shap

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(sessions)
    if hasattr(classifier, "feature_importances_"):
        explainer = shap.TreeExplainer(classifier)
    else:
        explainer = shap.LinearExplainer(classifier, transformed)
    values = _positive_class_values(explainer.shap_values(transformed))
    transformed_names = preprocessor.get_feature_names_out()
    names = list(sessions.columns)
    aggregated = np.zeros((len(sessions), len(names)), dtype=float)
    for position, transformed_name in enumerate(transformed_names):
        short_name = str(transformed_name).split("__", maxsplit=1)[-1]
        original = next(
            (
                name
                for name in names
                if short_name == name or short_name.startswith(f"{name}_")
            ),
            short_name,
        )
        if original in names:
            aggregated[:, names.index(original)] += values[:, position]
    return pd.DataFrame(aggregated, columns=names, index=sessions.index)


def local_lime_values(
    pipeline: Pipeline,
    training_sample: pd.DataFrame,
    session: pd.DataFrame,
    seed: int = 42,
) -> pd.Series:
    """Create an independent LIME explanation in the transformed feature space."""
    from lime.lime_tabular import LimeTabularExplainer

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed_training = preprocessor.transform(training_sample)
    transformed_session = preprocessor.transform(session)
    feature_names = list(preprocessor.get_feature_names_out())
    explainer = LimeTabularExplainer(
        training_data=np.asarray(transformed_training.toarray()
                                 if hasattr(transformed_training, "toarray")
                                 else transformed_training),
        feature_names=feature_names,
        class_names=["No purchase", "Purchase"],
        mode="classification",
        discretize_continuous=True,
        random_state=seed,
    )
    explanation = explainer.explain_instance(
        np.asarray(
            transformed_session.toarray()[0]
            if hasattr(transformed_session, "toarray")
            else transformed_session[0]
        ),
        classifier.predict_proba,
        num_features=min(12, len(feature_names)),
        labels=(1,),
    )
    return pd.Series(dict(explanation.as_list(label=1)), dtype=float)
