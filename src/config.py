"""Centralized project configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path | None = None) -> "ProjectConfig":
        config_path = path or ROOT_DIR / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with config_path.open(encoding="utf-8") as file:
            values = yaml.safe_load(file)
        if not isinstance(values, dict):
            raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")
        logger.info("Loaded project configuration from %s", config_path)
        return cls(raw=values)

    @property
    def project(self) -> dict[str, Any]:
        return self.raw["project"]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw["paths"]

    def resolve(self, value: str) -> Path:
        return (ROOT_DIR / value).resolve()


@dataclass(frozen=True)
class FeatureSpec:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    derived_numeric: tuple[str, ...]
    target: str

    @classmethod
    def from_config(cls, config: ProjectConfig) -> "FeatureSpec":
        data = config.data
        return cls(
            numeric=tuple(data["numerical"]),
            categorical=tuple(data["categorical"]),
            derived_numeric=tuple(data["derived_numerical"]),
            target=data["target"],
        )

    @property
    def input_features(self) -> tuple[str, ...]:
        return self.numeric + self.categorical

    @property
    def model_features(self) -> tuple[str, ...]:
        return self.numeric + self.derived_numeric + self.categorical


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
