"""Leak-safe training, model comparison, and threshold selection."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import FeatureSpec, ProjectConfig
from src.data import engineer_features, load_data

logger = logging.getLogger(__name__)
SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "f1": "f1",
    "precision": "precision",
    "accuracy": "accuracy",
}
TUNING_GRIDS: dict[str, dict[str, list[Any]]] = {
    "Logistic regression": {"classifier__C": [0.5, 1.0, 2.0]},
    "Decision tree": {
        "classifier__max_depth": [6, None],
        "classifier__min_samples_leaf": [4, 8],
    },
    "Random forest": {
        "classifier__max_depth": [12, None],
        "classifier__min_samples_leaf": [3, 8],
    },
    "XGBoost": {
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.04, 0.08],
    },
    "LightGBM": {
        "classifier__num_leaves": [15, 31],
        "classifier__max_depth": [-1, 8],
    },
    "CatBoost": {
        "classifier__depth": [5, 7],
        "classifier__learning_rate": [0.04, 0.08],
    },
}


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """Build all transformations inside the model pipeline to prevent leakage."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(spec.numeric + spec.derived_numeric)),
            ("categorical", categorical_pipeline, list(spec.categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _build_estimators(seed: int, positive_weight: float) -> dict[str, Any]:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "Logistic regression": LogisticRegression(
            class_weight="balanced", max_iter=1200, random_state=seed
        ),
        "Decision tree": DecisionTreeClassifier(
            class_weight="balanced", min_samples_leaf=8, random_state=seed
        ),
        "Random forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            min_samples_leaf=3,
            n_estimators=300,
            n_jobs=1,
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            learning_rate=0.06,
            max_depth=4,
            n_estimators=250,
            n_jobs=1,
            random_state=seed,
            scale_pos_weight=positive_weight,
            subsample=0.85,
            colsample_bytree=0.85,
        ),
        "LightGBM": LGBMClassifier(
            class_weight="balanced",
            learning_rate=0.06,
            max_depth=-1,
            n_estimators=250,
            n_jobs=1,
            random_state=seed,
            verbosity=-1,
        ),
        "CatBoost": CatBoostClassifier(
            auto_class_weights="Balanced",
            depth=6,
            iterations=300,
            learning_rate=0.06,
            random_seed=seed,
            verbose=False,
            thread_count=1,
        ),
    }


def _expected_value(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    conversion_value: float,
    intervention_cost: float,
) -> float:
    predicted = probabilities >= threshold
    true_positives = np.sum(predicted & (y_true == 1))
    intervention_count = np.sum(predicted)
    return float(true_positives * conversion_value - intervention_count * intervention_cost)


def choose_value_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    config: ProjectConfig,
) -> tuple[float, float]:
    training = config.training
    thresholds = np.linspace(
        training["threshold_min"],
        training["threshold_max"],
        training["threshold_steps"],
    )
    values = [
        _expected_value(
            y_true,
            probabilities,
            threshold,
            training["conversion_value"],
            training["intervention_cost"],
        )
        for threshold in thresholds
    ]
    best_index = int(np.argmax(values))
    return float(thresholds[best_index]), float(values[best_index])


def _evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


@dataclass
class ModelTrainer:
    """Train and compare six classifiers using stratified cross-validation."""

    config: ProjectConfig

    def train(self) -> dict[str, Any]:
        spec = FeatureSpec.from_config(self.config)
        frame = load_data(config=self.config)
        features = engineer_features(frame, spec)
        x = features.loc[:, spec.model_features]
        y = frame[spec.target].to_numpy()
        seed = int(self.config.project["random_seed"])

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=float(self.config.data["test_size"]),
            random_state=seed,
            stratify=y,
        )
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train,
            y_train,
            test_size=float(self.config.data["validation_size"]),
            random_state=seed,
            stratify=y_train,
        )

        positive_weight = float(np.sum(y_train == 0) / np.sum(y_train == 1))
        estimators = _build_estimators(seed, positive_weight)
        folds = StratifiedKFold(
            n_splits=int(self.config.training["cv_folds"]),
            shuffle=True,
            random_state=seed,
        )
        cv_rows: list[dict[str, Any]] = []
        fitted: dict[str, Pipeline] = {}
        estimators_by_name = estimators
        started = time.perf_counter()

        for model_name, estimator in estimators.items():
            logger.info("Cross-validating %s", model_name)
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(spec)),
                    ("classifier", estimator),
                ]
            )
            cv_result = cross_validate(
                pipeline,
                x_train,
                y_train,
                cv=folds,
                scoring=SCORING,
                n_jobs=1,
                return_train_score=False,
                error_score="raise",
            )
            pipeline.fit(x_train, y_train)
            fitted[model_name] = pipeline
            cv_rows.append(
                {
                    "model": model_name,
                    "cv_pr_auc": float(np.mean(cv_result["test_average_precision"])),
                    "cv_pr_auc_std": float(np.std(cv_result["test_average_precision"])),
                    "cv_roc_auc": float(np.mean(cv_result["test_roc_auc"])),
                    "cv_recall": float(np.mean(cv_result["test_recall"])),
                    "cv_f1": float(np.mean(cv_result["test_f1"])),
                    "cv_precision": float(np.mean(cv_result["test_precision"])),
                    "cv_accuracy": float(np.mean(cv_result["test_accuracy"])),
                    "tuned": False,
                }
            )

        cv_table = pd.DataFrame(cv_rows).sort_values(
            ["cv_pr_auc", "cv_recall"], ascending=False
        )
        tuning_folds = folds
        tuned_parameters: dict[str, dict[str, Any]] = {}
        for model_name in cv_table.head(2)["model"].tolist():
            logger.info("Tuning top candidate %s", model_name)
            candidate = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(spec)),
                    ("classifier", estimators_by_name[model_name]),
                ]
            )
            search = GridSearchCV(
                candidate,
                param_grid=TUNING_GRIDS[model_name],
                cv=tuning_folds,
                scoring="average_precision",
                n_jobs=1,
                refit=True,
                return_train_score=False,
                error_score="raise",
            )
            search.fit(x_train, y_train)
            fitted[model_name] = search.best_estimator_
            tuned_parameters[model_name] = search.best_params_
            result_index = cv_table.index[cv_table["model"] == model_name][0]
            cv_table.loc[result_index, "cv_pr_auc"] = float(search.best_score_)
            cv_table.loc[result_index, "cv_pr_auc_std"] = float(
                search.cv_results_["std_test_score"][search.best_index_]
            )
            cv_table.loc[result_index, "tuned"] = True

        cv_table = cv_table.sort_values(
            ["cv_pr_auc", "cv_recall"], ascending=False
        )
        best_name = str(cv_table.iloc[0]["model"])
        best_pipeline = fitted[best_name]
        validation_probabilities = best_pipeline.predict_proba(x_valid)[:, 1]
        threshold, validation_value = choose_value_threshold(
            y_valid, validation_probabilities, self.config
        )
        test_probabilities = best_pipeline.predict_proba(x_test)[:, 1]
        test_metrics = _evaluate(y_test, test_probabilities, threshold)
        test_value = _expected_value(
            y_test,
            test_probabilities,
            threshold,
            float(self.config.training["conversion_value"]),
            float(self.config.training["intervention_cost"]),
        )

        model_dir = self.config.resolve(self.config.paths["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.config.resolve(self.config.paths["artifact"])
        joblib.dump(best_pipeline, artifact_path)
        cv_table.to_csv(self.config.resolve(self.config.paths["metrics"]), index=False)

        importance = self._feature_importance(
            best_pipeline, x_valid, y_valid, spec, seed
        )
        importance_path = self.config.resolve(
            self.config.paths["feature_importance"]
        )
        importance.to_csv(importance_path, index=False)
        from src.explain import local_shap_values

        shap_sample = x_valid.sample(
            n=min(int(self.config.training["shap_explanation_rows"]), len(x_valid)),
            random_state=seed,
        )
        shap_values = local_shap_values(best_pipeline, shap_sample)
        shap_importance = (
            shap_values.abs()
            .mean()
            .rename("mean_absolute_shap")
            .rename_axis("feature")
            .reset_index()
            .sort_values("mean_absolute_shap", ascending=False)
        )
        shap_importance_path = self.config.resolve(
            self.config.paths["shap_importance"]
        )
        shap_importance.to_csv(shap_importance_path, index=False)

        metadata: dict[str, Any] = {
            "project": self.config.project["name"],
            "version": self.config.project["version"],
            "best_model": best_name,
            "tuned_parameters": tuned_parameters.get(best_name, {}),
            "threshold": threshold,
            "threshold_selection": "Maximum validation expected value",
            "conversion_value": float(self.config.training["conversion_value"]),
            "intervention_cost": float(self.config.training["intervention_cost"]),
            "validation_expected_value": validation_value,
            "test_expected_value": test_value,
            "test_metrics": test_metrics,
            "class_balance": {
                "negative": int(np.sum(y == 0)),
                "positive": int(np.sum(y == 1)),
                "conversion_rate": float(np.mean(y)),
            },
            "page_values_policy": (
                "Included; score is intended for late-session use after PageValues "
                "is available."
            ),
            "training_rows": len(x_train),
            "validation_rows": len(x_valid),
            "test_rows": len(x_test),
            "duration_seconds": round(time.perf_counter() - started, 2),
            "features": list(spec.model_features),
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        metadata_path = self.config.resolve(self.config.paths["metadata"])
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._log_mlflow(metadata, cv_table, artifact_path, model_dir)

        logger.info("Best model: %s | validation threshold: %.3f", best_name, threshold)
        return metadata

    def _feature_importance(
        self,
        pipeline: Pipeline,
        x_valid: pd.DataFrame,
        y_valid: np.ndarray,
        spec: FeatureSpec,
        seed: int,
    ) -> pd.DataFrame:
        from sklearn.inspection import permutation_importance

        sample_size = min(len(x_valid), 1200)
        sample = x_valid.sample(n=sample_size, random_state=seed)
        y_sample = y_valid[x_valid.index.get_indexer(sample.index)]
        result = permutation_importance(
            pipeline,
            sample,
            y_sample,
            n_repeats=3,
            random_state=seed,
            scoring="average_precision",
            n_jobs=1,
        )
        importance = pd.DataFrame(
            {
                "feature": list(spec.model_features),
                "permutation_importance": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        return importance.sort_values("permutation_importance", ascending=False)

    def _log_mlflow(
        self,
        metadata: dict[str, Any],
        cv_table: pd.DataFrame,
        artifact_path: Path,
        model_dir: Path,
    ) -> None:
        try:
            import mlflow
        except ImportError:
            logger.warning("MLflow is unavailable; skipping experiment logging.")
            return

        tracking_dir = self.config.resolve(self.config.paths["mlflow"])
        tracking_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_dir.as_uri())
        mlflow.set_experiment("ecommerce-purchase-intent")
        with mlflow.start_run(run_name=str(metadata["best_model"])):
            mlflow.log_params(
                {
                    "best_model": metadata["best_model"],
                    "threshold": metadata["threshold"],
                    "page_values": "included",
                }
            )
            mlflow.log_metrics(
                {
                    f"test_{name}": float(value)
                    for name, value in metadata["test_metrics"].items()
                    if isinstance(value, (int, float))
                }
            )
            mlflow.log_metrics(
                {
                    f"cv_{row.model}_pr_auc": float(row.cv_pr_auc)
                    for row in cv_table.itertuples()
                }
            )
            mlflow.log_artifact(str(artifact_path), artifact_path="model")
            mlflow.log_artifact(str(model_dir / "metadata.json"), artifact_path="model")
            mlflow.log_artifact(
                str(model_dir / "feature_importance.csv"), artifact_path="analysis"
            )
            mlflow.log_artifact(
                str(model_dir / "shap_importance.csv"), artifact_path="analysis"
            )
