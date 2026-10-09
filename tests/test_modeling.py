import numpy as np
from sklearn.compose import ColumnTransformer

from src.config import FeatureSpec, ProjectConfig
from src.data import engineer_features, load_data
from src.models import build_preprocessor, choose_value_threshold


def test_preprocessing_keeps_encoded_ids_categorical_and_handles_unknowns() -> None:
    config = ProjectConfig.load()
    spec = FeatureSpec.from_config(config)
    frame = engineer_features(load_data().head(40), spec)
    features = frame.loc[:, spec.model_features]
    preprocessor = build_preprocessor(spec)
    transformed = preprocessor.fit_transform(features.iloc[:30])
    transformed_unknown = preprocessor.transform(
        features.iloc[[30]].assign(Region="not-seen-before")
    )
    names = preprocessor.get_feature_names_out()

    assert transformed.shape[1] == len(names)
    assert transformed_unknown.shape == (1, transformed.shape[1])
    assert any(name.startswith("categorical__Region_") for name in names)
    assert isinstance(preprocessor, ColumnTransformer)


def test_threshold_maximizes_configured_net_value() -> None:
    config = ProjectConfig.load()
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.35, 0.65, 0.95])

    threshold, value = choose_value_threshold(y_true, probabilities, config)

    assert 0 < threshold < 1
    assert value >= 0
