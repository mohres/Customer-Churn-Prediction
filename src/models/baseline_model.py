"""
Baseline Churn Prediction Model

This module implements a logistic regression baseline model for customer churn prediction.
It includes proper temporal train/test splitting, comprehensive evaluation metrics, and
model interpretability features.

Key features:
- Temporal train/validation/test splits to prevent data leakage
- Multiple evaluation metrics (AUC-ROC, Precision@K, F1, etc.)
- Feature importance analysis
- Model calibration assessment
- Cross-validation with temporal ordering

Usage:
    from models.baseline_model import ChurnBaselineModel

    model = ChurnBaselineModel()
    results = model.train_and_evaluate(features_df, target_df)
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.calibration import calibration_curve
except ImportError:
    # Fallback for older sklearn versions
    calibration_curve = None
import warnings

from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnBaselineModel:
    """
    Baseline logistic regression model for churn prediction with temporal validation.

    This class provides a complete baseline implementation including:
    - Temporal data splitting to prevent leakage
    - Comprehensive evaluation metrics
    - Feature importance analysis
    - Model calibration assessment
    """

    def __init__(
        self,
        test_split_days: int = 14,
        val_split_days: int = 14,
        random_state: int = 42,
    ):
        """
        Initialize the baseline model.

        Args:
            test_split_days: Days to reserve for test set (most recent data)
            val_split_days: Days to reserve for validation set (before test)
            random_state: Random seed for reproducibility
        """
        self.test_split_days = test_split_days
        self.val_split_days = val_split_days
        self.random_state = random_state

        # Model components
        self.model = None
        self.scaler = None

        # Training metadata
        self.feature_names = None
        self.training_metadata = {}

        # Evaluation results
        self.results = {}

    def create_temporal_splits(
        self, features_df: pd.DataFrame, reference_date_col: str = "reference_date"
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/validation/test splits to prevent data leakage.

        Args:
            features_df: DataFrame with features and reference_date column
            reference_date_col: Name of the reference date column

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Creating temporal train/validation/test splits")

        # Ensure reference date is datetime
        if features_df[reference_date_col].dtype == "object":
            features_df[reference_date_col] = pd.to_datetime(
                features_df[reference_date_col]
            )

        # Calculate split dates
        max_date = features_df[reference_date_col].max()
        test_start = max_date - timedelta(days=self.test_split_days)
        val_start = test_start - timedelta(days=self.val_split_days)

        # Create splits
        train_df = features_df[features_df[reference_date_col] < val_start].copy()
        val_df = features_df[
            (features_df[reference_date_col] >= val_start)
            & (features_df[reference_date_col] < test_start)
        ].copy()
        test_df = features_df[features_df[reference_date_col] >= test_start].copy()

        logger.info("Data splits created:")
        logger.info(f"  Train: {len(train_df)} samples (until {val_start.date()})")
        logger.info(
            f"  Validation: {len(val_df)} samples ({val_start.date()} to {test_start.date()})"
        )
        logger.info(f"  Test: {len(test_df)} samples (from {test_start.date()})")

        self.training_metadata["split_dates"] = {
            "val_start": val_start,
            "test_start": test_start,
            "max_date": max_date,
        }

        return train_df, val_df, test_df

    def prepare_features(
        self,
        features_df: pd.DataFrame,
        target_col: str = "is_churned",
        exclude_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training by selecting numeric columns and handling missing values.

        Args:
            features_df: DataFrame with features and target
            target_col: Name of the target column
            exclude_cols: Additional columns to exclude from features

        Returns:
            Tuple of (x_features, y_target)
        """
        if exclude_cols is None:
            exclude_cols = [
                "userId",
                "reference_date",
                "computation_date",
                "feature_version",
            ]

        # Exclude target and metadata columns
        all_exclude = [*exclude_cols, target_col]

        feature_cols = [col for col in features_df.columns if col not in all_exclude]

        # Select only numeric features
        x = features_df[feature_cols].select_dtypes(include=[np.number]).copy()
        y = features_df[target_col].copy()

        # Handle missing values (fill with median)
        missing_counts = x.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(
                f"Found missing values in {(missing_counts > 0).sum()} features"
            )
            x = x.fillna(x.median())

        # Handle infinite values
        inf_counts = np.isinf(x).sum()
        if inf_counts.sum() > 0:
            logger.warning(
                f"Found infinite values in {(inf_counts > 0).sum()} features"
            )
            x = x.replace([np.inf, -np.inf], np.nan).fillna(x.median())

        self.feature_names = x.columns.tolist()

        logger.info(f"Prepared {len(self.feature_names)} features for training")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        return x, y

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict:
        """
        Train the logistic regression model with feature scaling.

        Args:
            x_train: Training features
            y_train: Training targets
            x_val: Validation features (optional, for monitoring)
            y_val: Validation targets (optional, for monitoring)

        Returns:
            Dictionary with training results
        """
        logger.info("Training baseline logistic regression model")

        # Initialize and fit scaler
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)

        # Train logistic regression with balanced class weights
        self.model = LogisticRegression(
            random_state=self.random_state, class_weight="balanced", max_iter=1000
        )

        self.model.fit(x_train_scaled, y_train)

        # Calculate training metrics
        y_train_pred_proba = self.model.predict_proba(x_train_scaled)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_pred_proba)

        results = {
            "train_auc": train_auc,
            "n_features": len(self.feature_names),
            "train_samples": len(x_train),
        }

        # Calculate validation metrics if provided
        if x_val is not None and y_val is not None:
            x_val_scaled = self.scaler.transform(x_val)
            y_val_pred_proba = self.model.predict_proba(x_val_scaled)[:, 1]
            val_auc = roc_auc_score(y_val, y_val_pred_proba)
            results["val_auc"] = val_auc
            results["val_samples"] = len(x_val)

            logger.info(f"Validation AUC: {val_auc:.4f}")

        logger.info(f"Training completed. Train AUC: {train_auc:.4f}")

        return results

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Comprehensive evaluation of the trained model.

        Args:
            x_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained before evaluation")

        logger.info("Evaluating model performance")

        # Scale test features
        x_test_scaled = self.scaler.transform(x_test)

        # Generate predictions
        y_pred_proba = self.model.predict_proba(x_test_scaled)[:, 1]
        y_pred = self.model.predict(x_test_scaled)

        # Basic classification metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Precision at different thresholds
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

        # Precision@K metrics (top K% of users by risk score)
        precision_at_k = {}
        for k in [5, 10, 20]:
            threshold_k = np.percentile(y_pred_proba, 100 - k)
            y_pred_k = (y_pred_proba >= threshold_k).astype(int)
            if y_pred_k.sum() > 0:
                precision_at_k[f"precision_at_{k}%"] = precision_score(
                    y_test, y_pred_k, zero_division=0
                )
            else:
                precision_at_k[f"precision_at_{k}%"] = 0

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Compile results
        results = {
            "test_samples": len(y_test),
            "positive_rate": y_test.mean(),
            "auc_roc": auc_score,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            **precision_at_k,
        }

        # Add calibration metrics
        if calibration_curve is not None:
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, y_pred_proba, n_bins=10, strategy="quantile"
                )
                results["calibration_slope"] = np.corrcoef(
                    fraction_of_positives, mean_predicted_value
                )[0, 1]
            except Exception:
                results["calibration_slope"] = np.nan
        else:
            results["calibration_slope"] = np.nan

        logger.info(
            f"Test AUC: {auc_score:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )

        return results

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance based on logistic regression coefficients.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")

        # Get coefficients and calculate importance (absolute value)
        coefficients = self.model.coef_[0]
        importance_scores = np.abs(coefficients)

        # Create importance dataframe
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "coefficient": coefficients,
                "importance": importance_scores,
            }
        ).sort_values("importance", ascending=False)

        return importance_df.head(top_n)

    def cross_validate(
        self,
        features_df: pd.DataFrame,
        target_col: str = "is_churned",
        reference_date_col: str = "reference_date",
        n_splits: int = 3,
    ) -> dict:
        """
        Perform time-series cross-validation to assess model stability.

        Args:
            features_df: DataFrame with features and target
            target_col: Name of the target column
            reference_date_col: Name of the reference date column
            n_splits: Number of cross-validation splits

        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Performing {n_splits}-fold time-series cross-validation")

        # Prepare features
        x, y = self.prepare_features(features_df, target_col)

        # Sort by reference date for temporal splits
        sort_idx = features_df[reference_date_col].argsort()
        x = x.iloc[sort_idx]
        y = y.iloc[sort_idx]

        # Perform time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(x)):
            x_train_fold, x_val_fold = x.iloc[train_idx], x.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Train fold model
            scaler_fold = StandardScaler()
            x_train_scaled = scaler_fold.fit_transform(x_train_fold)
            x_val_scaled = scaler_fold.transform(x_val_fold)

            model_fold = LogisticRegression(
                random_state=self.random_state, class_weight="balanced", max_iter=1000
            )
            model_fold.fit(x_train_scaled, y_train_fold)

            # Evaluate fold
            y_val_pred_proba = model_fold.predict_proba(x_val_scaled)[:, 1]
            fold_auc = roc_auc_score(y_val_fold, y_val_pred_proba)
            cv_scores.append(fold_auc)

            logger.info(f"Fold {fold + 1} AUC: {fold_auc:.4f}")

        results = {
            "cv_scores": cv_scores,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "cv_min": np.min(cv_scores),
            "cv_max": np.max(cv_scores),
        }

        logger.info(
            f"Cross-validation AUC: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}"
        )

        return results

    def train_and_evaluate(
        self,
        features_df: pd.DataFrame,
        target_col: str = "is_churned",
        reference_date_col: str = "reference_date",
    ) -> dict:
        """
        Complete training and evaluation pipeline.

        Args:
            features_df: DataFrame with features, target, and reference date
            target_col: Name of the target column
            reference_date_col: Name of the reference date column

        Returns:
            Dictionary with complete results including training, evaluation, and cross-validation
        """
        logger.info("Starting complete baseline model training and evaluation")

        # Create temporal splits
        train_df, val_df, test_df = self.create_temporal_splits(
            features_df, reference_date_col
        )

        # Prepare features for each split
        x_train, y_train = self.prepare_features(train_df, target_col)
        x_val, y_val = self.prepare_features(val_df, target_col)
        x_test, y_test = self.prepare_features(test_df, target_col)

        # Train the model
        training_results = self.train(x_train, y_train, x_val, y_val)

        # Evaluate on test set
        test_results = self.evaluate(x_test, y_test)

        # Get feature importance
        feature_importance = self.get_feature_importance()

        # Perform cross-validation
        cv_results = self.cross_validate(train_df, target_col, reference_date_col)

        # Compile all results
        self.results = {
            "training": training_results,
            "test_evaluation": test_results,
            "cross_validation": cv_results,
            "feature_importance": feature_importance.to_dict("records"),
            "metadata": {
                "model_type": "logistic_regression",
                "training_date": datetime.now().isoformat(),
                "n_features": len(self.feature_names),
                "feature_names": self.feature_names[
                    :10
                ],  # Store first 10 for reference
                **self.training_metadata,
            },
        }

        # Log key results
        logger.info("=" * 60)
        logger.info("BASELINE MODEL RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Test AUC-ROC: {test_results['auc_roc']:.4f}")
        logger.info(f"Test F1 Score: {test_results['f1_score']:.4f}")
        logger.info(f"Test Precision: {test_results['precision']:.4f}")
        logger.info(f"Test Recall: {test_results['recall']:.4f}")
        logger.info(f"Precision@10%: {test_results.get('precision_at_10%', 'N/A')}")
        logger.info(
            f"Cross-val AUC: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}"
        )
        logger.info(f"Features used: {len(self.feature_names)}")
        logger.info("=" * 60)

        return self.results

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """
        Generate prediction probabilities for new data.

        Args:
            x: Features dataframe

        Returns:
            Array of prediction probabilities
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained before making predictions")

        # Select and scale features
        x_features = x[self.feature_names]
        x_scaled = self.scaler.transform(x_features)

        return self.model.predict_proba(x_scaled)[:, 1]

    def save_results_summary(self, file_path: str | None = None) -> str:
        """
        Save a summary of results to a text file.

        Args:
            file_path: Optional file path, defaults to timestamped file

        Returns:
            Path to the saved file
        """
        if not self.results:
            raise ValueError("No results to save. Run train_and_evaluate() first.")

        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"baseline_model_results_{timestamp}.txt"

        with open(file_path, "w") as f:
            f.write("CHURN PREDICTION BASELINE MODEL RESULTS\n")
            f.write("=" * 50 + "\n\n")

            # Test results
            test_results = self.results["test_evaluation"]
            f.write("TEST SET PERFORMANCE:\n")
            f.write(f"  AUC-ROC: {test_results['auc_roc']:.4f}\n")
            f.write(f"  F1 Score: {test_results['f1_score']:.4f}\n")
            f.write(f"  Precision: {test_results['precision']:.4f}\n")
            f.write(f"  Recall: {test_results['recall']:.4f}\n")
            f.write(
                f"  Precision@10%: {test_results.get('precision_at_10%', 'N/A')}\n\n"
            )

            # Cross-validation
            cv_results = self.results["cross_validation"]
            f.write("CROSS-VALIDATION RESULTS:\n")
            f.write(f"  Mean AUC: {cv_results['cv_mean']:.4f}\n")
            f.write(f"  Std AUC: {cv_results['cv_std']:.4f}\n")
            f.write(
                f"  Range: {cv_results['cv_min']:.4f} - {cv_results['cv_max']:.4f}\n\n"
            )

            # Feature importance
            f.write("TOP 10 FEATURES:\n")
            for i, feat in enumerate(self.results["feature_importance"][:10]):
                f.write(f"  {i+1}. {feat['feature']}: {feat['importance']:.4f}\n")

        logger.info(f"Results summary saved to: {file_path}")
        return file_path


if __name__ == "__main__":
    print("Baseline Churn Prediction Model")
    print("This module provides a logistic regression baseline for churn prediction.")
    print(
        "Use ChurnBaselineModel.train_and_evaluate() for complete model training and evaluation."
    )
