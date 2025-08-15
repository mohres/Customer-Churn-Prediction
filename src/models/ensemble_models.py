"""
Advanced Ensemble Models for Churn Prediction

This module implements advanced machine learning models including XGBoost and LightGBM
for high-performance churn prediction. It includes:
- XGBoost classifier with optimized parameters
- LightGBM classifier with feature importance analysis
- Hyperparameter optimization using Optuna
- Model evaluation and comparison utilities
- SHAP analysis for model interpretability

Key classes:
- XGBoostChurnModel: XGBoost implementation with optimization
- LightGBMChurnModel: LightGBM implementation with feature analysis
- EnsembleModelOptimizer: Hyperparameter optimization framework
- ModelComparison: Compare multiple models side by side

Usage:
    from models.ensemble_models import XGBoostChurnModel, LightGBMChurnModel

    xgb_model = XGBoostChurnModel()
    lgb_model = LightGBMChurnModel()

    xgb_results = xgb_model.train_and_evaluate(X_train, y_train, X_test, y_test)
    lgb_results = lgb_model.train_and_evaluate(X_train, y_train, X_test, y_test)
"""

import logging
import warnings
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseEnsembleModel:
    """Base class for ensemble models with common functionality."""

    def __init__(self, model_name: str = "BaseModel"):
        """
        Initialize base ensemble model.

        Args:
            model_name: Name of the model for logging and tracking
        """
        self.model_name = model_name
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.training_history = {}

    def _validate_inputs(self, x: pd.DataFrame, y: pd.Series) -> None:
        """Validate input data."""
        if x is None or y is None:
            raise ValueError("Features (x) and target (y) cannot be None")

        if len(x) != len(y):
            raise ValueError("Features and target must have same length")

        if x.isnull().any().any():
            logger.warning("Features contain missing values - consider preprocessing")

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Calculate standard classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
        }

        if y_pred_proba is not None:
            metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)

        return metrics

    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "model_name": self.model_name,
            "training_history": self.training_history,
            "timestamp": datetime.now().isoformat(),
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        self.model_name = model_data.get("model_name", "LoadedModel")
        self.training_history = model_data.get("training_history", {})
        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")


class XGBoostChurnModel(BaseEnsembleModel):
    """XGBoost implementation for churn prediction."""

    def __init__(self, **xgb_params):
        """
        Initialize XGBoost model.

        Args:
            **xgb_params: XGBoost parameters to override defaults
        """
        super().__init__("XGBoost")

        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost not available. Install with: pip install xgboost"
            )

        # Default parameters optimized for churn prediction
        self.default_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": 10,
            "scale_pos_weight": 1,  # Will be adjusted based on class imbalance
        }

        # Update with provided parameters
        self.params = {**self.default_params, **xgb_params}
        logger.info(f"Initialized {self.model_name} with parameters: {self.params}")

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Train XGBoost model.

        Args:
            x_train: Training features
            y_train: Training target
            x_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Dictionary with training results
        """
        self._validate_inputs(x_train, y_train)

        # Calculate scale_pos_weight for class imbalance
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

        self.params["scale_pos_weight"] = scale_pos_weight
        logger.info(f"Adjusted scale_pos_weight to {scale_pos_weight:.3f}")

        # Store feature names
        self.feature_names = list(x_train.columns)

        # Create evaluation set if validation data provided
        eval_set = []
        if x_val is not None and y_val is not None:
            eval_set = [(x_train, y_train), (x_val, y_val)]
        else:
            eval_set = [(x_train, y_train)]

        # Train model
        logger.info(f"Training {self.model_name} on {len(x_train)} samples...")
        start_time = datetime.now()

        self.model = xgb.XGBClassifier(**self.params)

        if eval_set:
            self.model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(x_train, y_train)

        training_time = (datetime.now() - start_time).total_seconds()

        # Store training history
        self.training_history = {
            "training_time_seconds": training_time,
            "n_features": len(self.feature_names),
            "n_samples": len(x_train),
            "class_distribution": dict(y_train.value_counts()),
            "parameters": self.params,
        }

        self.is_trained = True
        logger.info(f"Training completed in {training_time:.2f} seconds")

        return self.training_history

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(x)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict_proba(x)[:, 1]

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')

        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        if importance_type == "gain":
            importance = self.model.feature_importances_
        else:
            importance = self.model.get_booster().get_score(
                importance_type=importance_type
            )
            # Convert to array format
            importance = [
                importance.get(f"f{i}", 0) for i in range(len(self.feature_names))
            ]

        feature_importance = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        return feature_importance

    def train_and_evaluate(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Train model and evaluate performance.

        Args:
            x_train: Training features
            y_train: Training target
            x_test: Test features
            y_test: Test target
            x_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Dictionary with training and evaluation results
        """
        # Train the model
        training_results = self.train(x_train, y_train, x_val, y_val)

        # Evaluate on test set
        y_pred = self.predict(x_test)
        y_pred_proba = self.predict_proba(x_test)

        test_metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        # Evaluate on training set for comparison
        y_train_pred = self.predict(x_train)
        y_train_pred_proba = self.predict_proba(x_train)
        train_metrics = self._calculate_metrics(
            y_train, y_train_pred, y_train_pred_proba
        )

        results = {
            "model_name": self.model_name,
            "training_history": training_results,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": self.get_feature_importance(),
        }

        if x_val is not None and y_val is not None:
            y_val_pred = self.predict(x_val)
            y_val_pred_proba = self.predict_proba(x_val)
            val_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_pred_proba)
            results["val_metrics"] = val_metrics

        logger.info(f"{self.model_name} evaluation completed:")
        logger.info(f"  Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
        logger.info(f"  Test F1: {test_metrics['f1_score']:.4f}")

        return results


class LightGBMChurnModel(BaseEnsembleModel):
    """LightGBM implementation for churn prediction."""

    def __init__(self, **lgb_params):
        """
        Initialize LightGBM model.

        Args:
            **lgb_params: LightGBM parameters to override defaults
        """
        super().__init__("LightGBM")

        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM not available. Install with: pip install lightgbm"
            )

        # Default parameters optimized for churn prediction
        self.default_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": 10,
            "verbose": -1,
        }

        # Update with provided parameters
        self.params = {**self.default_params, **lgb_params}
        logger.info(f"Initialized {self.model_name} with parameters: {self.params}")

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Train LightGBM model.

        Args:
            x_train: Training features
            y_train: Training target
            x_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Dictionary with training results
        """
        self._validate_inputs(x_train, y_train)

        # Store feature names
        self.feature_names = list(x_train.columns)

        # Train model
        logger.info(f"Training {self.model_name} on {len(x_train)} samples...")
        start_time = datetime.now()

        self.model = lgb.LGBMClassifier(**self.params)

        if x_val is not None and y_val is not None:
            self.model.fit(
                x_train,
                y_train,
                eval_set=[(x_train, y_train), (x_val, y_val)],
                eval_names=["train", "val"],
            )
        else:
            self.model.fit(x_train, y_train)

        training_time = (datetime.now() - start_time).total_seconds()

        # Store training history
        self.training_history = {
            "training_time_seconds": training_time,
            "n_features": len(self.feature_names),
            "n_samples": len(x_train),
            "class_distribution": dict(y_train.value_counts()),
            "parameters": self.params,
        }

        self.is_trained = True
        logger.info(f"Training completed in {training_time:.2f} seconds")

        return self.training_history

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(x)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict_proba(x)[:, 1]

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            importance_type: Type of importance ('split', 'gain')

        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        if importance_type == "gain":
            importance = self.model.feature_importances_
        else:
            importance = self.model.booster_.feature_importance(
                importance_type=importance_type
            )

        feature_importance = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        return feature_importance

    def train_and_evaluate(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Train model and evaluate performance.

        Args:
            x_train: Training features
            y_train: Training target
            x_test: Test features
            y_test: Test target
            x_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Dictionary with training and evaluation results
        """
        # Train the model
        training_results = self.train(x_train, y_train, x_val, y_val)

        # Evaluate on test set
        y_pred = self.predict(x_test)
        y_pred_proba = self.predict_proba(x_test)

        test_metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        # Evaluate on training set for comparison
        y_train_pred = self.predict(x_train)
        y_train_pred_proba = self.predict_proba(x_train)
        train_metrics = self._calculate_metrics(
            y_train, y_train_pred, y_train_pred_proba
        )

        results = {
            "model_name": self.model_name,
            "training_history": training_results,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": self.get_feature_importance(),
        }

        if x_val is not None and y_val is not None:
            y_val_pred = self.predict(x_val)
            y_val_pred_proba = self.predict_proba(x_val)
            val_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_pred_proba)
            results["val_metrics"] = val_metrics

        logger.info(f"{self.model_name} evaluation completed:")
        logger.info(f"  Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
        logger.info(f"  Test F1: {test_metrics['f1_score']:.4f}")

        return results


class EnsembleModelOptimizer:
    """Hyperparameter optimization for ensemble models using Optuna."""

    def __init__(self, model_type: str = "xgboost", n_trials: int = 100):
        """
        Initialize optimizer.

        Args:
            model_type: Type of model to optimize ('xgboost' or 'lightgbm')
            n_trials: Number of optimization trials
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        self.model_type = model_type.lower()
        self.n_trials = n_trials
        self.study = None
        self.best_params = None

        if self.model_type not in ["xgboost", "lightgbm"]:
            raise ValueError("model_type must be 'xgboost' or 'lightgbm'")

    def _objective_xgboost(self, trial, x_train, y_train, x_val, y_val):
        """Objective function for XGBoost optimization."""
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        }

        model = XGBoostChurnModel(**params)
        model.train(x_train, y_train, x_val, y_val)

        y_val_pred_proba = model.predict_proba(x_val)
        auc_score = roc_auc_score(y_val, y_val_pred_proba)

        return auc_score

    def _objective_lightgbm(self, trial, x_train, y_train, x_val, y_val):
        """Objective function for LightGBM optimization."""
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 10, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }

        model = LightGBMChurnModel(**params)
        model.train(x_train, y_train, x_val, y_val)

        y_val_pred_proba = model.predict_proba(x_val)
        auc_score = roc_auc_score(y_val, y_val_pred_proba)

        return auc_score

    def optimize(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            x_train: Training features
            y_train: Training target
            x_val: Validation features
            y_val: Validation target

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting hyperparameter optimization for {self.model_type}...")
        logger.info(f"Running {self.n_trials} trials")

        # Create study
        self.study = optuna.create_study(direction="maximize")

        # Define objective function
        if self.model_type == "xgboost":

            def objective(trial):
                return self._objective_xgboost(trial, x_train, y_train, x_val, y_val)

        else:

            def objective(trial):
                return self._objective_lightgbm(trial, x_train, y_train, x_val, y_val)

        # Run optimization
        start_time = datetime.now()
        self.study.optimize(objective, n_trials=self.n_trials)
        optimization_time = (datetime.now() - start_time).total_seconds()

        self.best_params = self.study.best_params

        results = {
            "best_params": self.best_params,
            "best_score": self.study.best_value,
            "n_trials": len(self.study.trials),
            "optimization_time_seconds": optimization_time,
            "model_type": self.model_type,
        }

        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        logger.info(f"Best AUC-ROC: {self.study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return results

    def get_optimized_model(self):
        """Get model with optimized parameters."""
        if self.best_params is None:
            raise ValueError("Must run optimization before getting optimized model")

        if self.model_type == "xgboost":
            return XGBoostChurnModel(**self.best_params)
        else:
            return LightGBMChurnModel(**self.best_params)


class ModelComparison:
    """Compare multiple models side by side."""

    def __init__(self):
        """Initialize model comparison."""
        self.results = {}

    def add_model_results(self, model_name: str, results: dict[str, Any]) -> None:
        """Add model results to comparison."""
        self.results[model_name] = results

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Get comparison results as DataFrame."""
        comparison_data = []

        for model_name, results in self.results.items():
            row = {"model": model_name}

            # Add test metrics
            if "test_metrics" in results:
                for metric, value in results["test_metrics"].items():
                    row[f"test_{metric}"] = value

            # Add training metrics
            if "train_metrics" in results:
                for metric, value in results["train_metrics"].items():
                    row[f"train_{metric}"] = value

            # Add training info
            if "training_history" in results:
                row["training_time"] = results["training_history"].get(
                    "training_time_seconds", 0
                )
                row["n_features"] = results["training_history"].get("n_features", 0)

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def get_best_model(self, metric: str = "test_auc_roc") -> str:
        """Get name of best performing model."""
        comparison_df = self.get_comparison_dataframe()

        if metric not in comparison_df.columns:
            raise ValueError(f"Metric {metric} not found in results")

        best_model = comparison_df.loc[comparison_df[metric].idxmax(), "model"]
        return best_model

    def print_comparison(self) -> None:
        """Print formatted comparison results."""
        comparison_df = self.get_comparison_dataframe()

        print("MODEL COMPARISON RESULTS")
        print("=" * 50)

        # Key metrics to display
        key_metrics = [
            "test_auc_roc",
            "test_f1_score",
            "test_precision",
            "test_recall",
            "training_time",
        ]
        display_cols = ["model"] + [
            col for col in key_metrics if col in comparison_df.columns
        ]

        display_df = comparison_df[display_cols].round(4)
        print(display_df.to_string(index=False))

        # Highlight best performer
        if "test_auc_roc" in comparison_df.columns:
            best_model = self.get_best_model("test_auc_roc")
            best_score = comparison_df[comparison_df["model"] == best_model][
                "test_auc_roc"
            ].iloc[0]
            print(f"\nBest performing model: {best_model} (AUC-ROC: {best_score:.4f})")


# SHAP Analysis Integration
def analyze_model_with_shap(
    model, x_sample: pd.DataFrame, feature_names: list[str]
) -> dict[str, Any] | None:
    """
    Analyze model with SHAP values.

    Args:
        model: Trained model (XGBoost or LightGBM)
        x_sample: Sample of features for SHAP analysis
        feature_names: List of feature names

    Returns:
        Dictionary with SHAP analysis results or None if SHAP not available
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Install with: pip install shap")
        return None

    try:
        # Create SHAP explainer
        if hasattr(model, "model") and hasattr(model.model, "get_booster"):
            # XGBoost model
            explainer = shap.TreeExplainer(model.model)
        elif hasattr(model, "model") and hasattr(model.model, "booster_"):
            # LightGBM model
            explainer = shap.TreeExplainer(model.model)
        else:
            # Fallback to general explainer
            explainer = shap.Explainer(model.predict, x_sample)

        # Calculate SHAP values
        shap_values = explainer.shap_values(x_sample)

        # Get feature importance from SHAP
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification

        feature_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_importance": np.abs(shap_values).mean(axis=0),
            }
        ).sort_values("shap_importance", ascending=False)

        results = {
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "explainer": explainer,
        }

        logger.info("SHAP analysis completed successfully")
        return results

    except Exception as e:
        logger.error(f"SHAP analysis failed: {e!s}")
        return None
