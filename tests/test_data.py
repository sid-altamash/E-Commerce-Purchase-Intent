from pathlib import Path

import pandas as pd
import pytest

from src.config import FeatureSpec, ProjectConfig
from src.data import DataValidationError, engineer_features, load_data


def test_source_data_loads_with_binary_target() -> None:
    frame = load_data()

    assert len(frame) == 12_000
    assert set(frame["Converted"].unique()) == {0, 1}
    assert frame.isna().sum().sum() == 0
    assert frame["OperatingSystems"].dtype.name == "string"


def test_missing_dataset_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Session dataset not found"):
        load_data(path=tmp_path / "missing.csv")


def test_schema_validation_lists_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({"Converted": [0]}).to_csv(path, index=False)

    with pytest.raises(DataValidationError, match="missing required columns"):
        load_data(path=path)


def test_engineered_features_are_safe_for_empty_sessions() -> None:
    config = ProjectConfig.load()
    spec = FeatureSpec.from_config(config)
    frame = load_data().loc[[0]].copy()
    frame.loc[:, ["Administrative", "Informational", "ProductRelated"]] = 0

    result = engineer_features(frame, spec)

    assert result["TotalPages"].iloc[0] == 0
    assert result["ProductPageShare"].iloc[0] == 0
    assert result["ProductSecondsPerPage"].iloc[0] == 0
