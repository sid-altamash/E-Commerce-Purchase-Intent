"""Command-line entry point for training and evaluation."""

from __future__ import annotations

import logging

from src.config import ProjectConfig, configure_logging
from src.models import ModelTrainer


def main() -> None:
    configure_logging()
    config = ProjectConfig.load()
    metadata = ModelTrainer(config).train()
    logging.getLogger(__name__).info(
        "Training complete: %s | PR-AUC %.3f | threshold %.3f",
        metadata["best_model"],
        metadata["test_metrics"]["pr_auc"],
        metadata["threshold"],
    )


if __name__ == "__main__":
    main()
