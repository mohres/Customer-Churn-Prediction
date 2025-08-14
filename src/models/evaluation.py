"""
Model Evaluation Module for Churn Prediction

This module provides comprehensive evaluation tools for churn prediction models,
including various metrics, visualizations, and analysis utilities.

Key functions:
- evaluate_classification_model(): Complete classification evaluation
- plot_model_performance(): Generate performance visualization plots
- calculate_business_metrics(): Business-focused evaluation metrics
- compare_models(): Compare multiple models side by side
- generate_evaluation_report(): Create comprehensive evaluation report

Usage:
    from models.evaluation import evaluate_classification_model

    results = evaluate_classification_model(y_true, y_pred, y_pred_proba)
"""

import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from sklearn.calibration import calibration_curve
except ImportError:
    # Fallback for older sklearn versions
    calibration_curve = None
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_classification_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Comprehensive evaluation of a binary classification model.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_pred_proba: Predicted probabilities for positive class
        model_name: Name of the model for reporting

    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    logger.info(f"Evaluating {model_name} performance")

    # Basic classification metrics
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # Additional metrics
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    brier_score = brier_score_loss(y_true, y_pred_proba)

    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Derived metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

    # Precision@K metrics (top K% of users by risk score)
    precision_at_k = {}
    recall_at_k = {}

    for k in [5, 10, 15, 20]:
        threshold_k = np.percentile(y_pred_proba, 100 - k)
        y_pred_k = (y_pred_proba >= threshold_k).astype(int)

        if y_pred_k.sum() > 0:
            precision_at_k[f"precision_at_{k}%"] = precision_score(
                y_true, y_pred_k, zero_division=0
            )
            recall_at_k[f"recall_at_{k}%"] = recall_score(
                y_true, y_pred_k, zero_division=0
            )
        else:
            precision_at_k[f"precision_at_{k}%"] = 0
            recall_at_k[f"recall_at_{k}%"] = 0

    # Lift calculation at different percentiles
    lift_at_k = {}
    baseline_rate = y_true.mean()

    for k in [10, 20]:
        threshold_k = np.percentile(y_pred_proba, 100 - k)
        y_pred_k = (y_pred_proba >= threshold_k).astype(int)

        if y_pred_k.sum() > 0:
            precision_k = precision_score(y_true, y_pred_k, zero_division=0)
            lift_at_k[f"lift_at_{k}%"] = (
                precision_k / baseline_rate if baseline_rate > 0 else 0
            )
        else:
            lift_at_k[f"lift_at_{k}%"] = 0

    # Compile comprehensive results
    results = {
        "model_name": model_name,
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
        "positive_rate": float(y_true.mean()),
        # Core metrics
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "npv": float(npv),
        # Additional metrics
        "matthews_corrcoef": float(mcc),
        "cohen_kappa": float(kappa),
        "brier_score": float(brier_score),
        # Confusion matrix
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "fpr": float(fpr),
        "fnr": float(fnr),
        # Precision/Recall at K
        **precision_at_k,
        **recall_at_k,
        **lift_at_k,
        # Metadata
        "evaluation_date": datetime.now().isoformat(),
    }

    logger.info(f"{model_name} evaluation completed:")
    logger.info(f"  AUC-ROC: {auc_roc:.4f}")
    logger.info(f"  AUC-PR: {auc_pr:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Precision@10%: {precision_at_k.get('precision_at_10%', 0):.4f}")
    logger.info(f"  Lift@10%: {lift_at_k.get('lift_at_10%', 0):.2f}x")

    return results


def plot_model_performance(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    figsize: tuple[int, int] = (15, 12),
) -> plt.Figure:
    """
    Generate comprehensive performance visualization plots.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        model_name: Name of the model for plot titles
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f"{model_name} - Performance Analysis", fontsize=16, fontweight="bold")

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_roc = roc_auc_score(y_true, y_pred_proba)

    axes[0, 0].plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_roc:.3f})", linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], "k--", label="Random Classifier")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    baseline_precision = y_true.mean()

    axes[0, 1].plot(
        recall, precision, label=f"PR Curve (AUC = {auc_pr:.3f})", linewidth=2
    )
    axes[0, 1].axhline(
        y=baseline_precision,
        color="k",
        linestyle="--",
        label=f"Baseline = {baseline_precision:.3f}",
    )
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Prediction Distribution
    axes[0, 2].hist(
        y_pred_proba[y_true == 0], bins=30, alpha=0.7, label="Non-Churned", density=True
    )
    axes[0, 2].hist(
        y_pred_proba[y_true == 1], bins=30, alpha=0.7, label="Churned", density=True
    )
    axes[0, 2].set_xlabel("Predicted Probability")
    axes[0, 2].set_ylabel("Density")
    axes[0, 2].set_title("Prediction Distribution")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Calibration Plot
    if calibration_curve is not None:
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10, strategy="quantile"
            )
            axes[1, 0].plot(
                mean_predicted_value,
                fraction_of_positives,
                "s-",
                label="Model",
                linewidth=2,
                markersize=6,
            )
            axes[1, 0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            axes[1, 0].set_xlabel("Mean Predicted Probability")
            axes[1, 0].set_ylabel("Fraction of Positives")
            axes[1, 0].set_title("Calibration Plot")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        except Exception:
            axes[1, 0].text(
                0.5,
                0.5,
                "Calibration plot\nnot available",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set_title("Calibration Plot")
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "Calibration plot\nnot available",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_title("Calibration Plot")

    # 5. Precision/Recall at different thresholds
    percentiles = [5, 10, 15, 20, 25, 30]
    precisions_at_k = []
    recalls_at_k = []

    for k in percentiles:
        threshold_k = np.percentile(y_pred_proba, 100 - k)
        y_pred_k = (y_pred_proba >= threshold_k).astype(int)

        if y_pred_k.sum() > 0:
            precisions_at_k.append(precision_score(y_true, y_pred_k, zero_division=0))
            recalls_at_k.append(recall_score(y_true, y_pred_k, zero_division=0))
        else:
            precisions_at_k.append(0)
            recalls_at_k.append(0)

    axes[1, 1].plot(
        percentiles,
        precisions_at_k,
        "o-",
        label="Precision@K",
        linewidth=2,
        markersize=6,
    )
    axes[1, 1].plot(
        percentiles, recalls_at_k, "s-", label="Recall@K", linewidth=2, markersize=6
    )
    axes[1, 1].set_xlabel("Top K% of Users")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("Precision/Recall at K")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1.05)

    # 6. Lift Chart
    lifts = []
    baseline_rate = y_true.mean()

    for k in percentiles:
        threshold_k = np.percentile(y_pred_proba, 100 - k)
        y_pred_k = (y_pred_proba >= threshold_k).astype(int)

        if y_pred_k.sum() > 0:
            precision_k = precision_score(y_true, y_pred_k, zero_division=0)
            lift = precision_k / baseline_rate if baseline_rate > 0 else 0
            lifts.append(lift)
        else:
            lifts.append(0)

    axes[1, 2].plot(percentiles, lifts, "o-", linewidth=2, markersize=6, color="red")
    axes[1, 2].axhline(y=1, color="k", linestyle="--", alpha=0.5, label="Baseline")
    axes[1, 2].set_xlabel("Top K% of Users")
    axes[1, 2].set_ylabel("Lift")
    axes[1, 2].set_title("Lift Chart")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(0, max(lifts) * 1.1 if lifts else 1)

    plt.tight_layout()
    return fig


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    intervention_cost: float = 10.0,
    churn_value_loss: float = 100.0,
    intervention_success_rate: float = 0.3,
) -> dict:
    """
    Calculate business-focused evaluation metrics for churn prediction.

    Args:
        y_true: True binary labels (1 = churned)
        y_pred_proba: Predicted churn probabilities
        intervention_cost: Cost of intervention per user
        churn_value_loss: Value lost when a customer churns
        intervention_success_rate: Success rate of intervention (retention rate)

    Returns:
        Dictionary with business metrics at different targeting levels
    """
    logger.info("Calculating business metrics")

    business_results = {
        "parameters": {
            "intervention_cost": intervention_cost,
            "churn_value_loss": churn_value_loss,
            "intervention_success_rate": intervention_success_rate,
        },
        "scenarios": {},
    }

    # Calculate baseline (no model) scenario
    total_users = len(y_true)
    actual_churns = y_true.sum()
    baseline_loss = actual_churns * churn_value_loss

    business_results["baseline"] = {
        "total_users": total_users,
        "actual_churns": int(actual_churns),
        "total_loss": baseline_loss,
        "intervention_cost": 0,
        "net_loss": baseline_loss,
    }

    # Calculate metrics for different targeting percentiles
    for target_pct in [5, 10, 15, 20, 25]:
        threshold = np.percentile(y_pred_proba, 100 - target_pct)
        targeted_users = y_pred_proba >= threshold
        n_targeted = targeted_users.sum()

        if n_targeted == 0:
            continue

        # True positives (correctly identified churners)
        tp = (targeted_users & (y_true == 1)).sum()

        # Calculate business impact
        intervention_costs = n_targeted * intervention_cost

        # Prevented churns (assuming intervention success rate)
        prevented_churns = tp * intervention_success_rate
        prevented_loss = prevented_churns * churn_value_loss

        # Remaining churns (those not prevented)
        remaining_churns = actual_churns - prevented_churns
        remaining_loss = remaining_churns * churn_value_loss

        # Net impact
        total_costs = intervention_costs
        total_savings = prevented_loss
        net_impact = total_savings - total_costs

        business_results["scenarios"][f"target_top_{target_pct}%"] = {
            "targeted_users": int(n_targeted),
            "correctly_identified_churners": int(tp),
            "precision": float(tp / n_targeted) if n_targeted > 0 else 0,
            "recall": float(tp / actual_churns) if actual_churns > 0 else 0,
            "intervention_costs": intervention_costs,
            "prevented_churns": prevented_churns,
            "prevented_loss": prevented_loss,
            "remaining_churns": remaining_churns,
            "remaining_loss": remaining_loss,
            "net_impact": net_impact,
            "roi": (net_impact / intervention_costs) if intervention_costs > 0 else 0,
        }

    # Find optimal targeting percentage
    best_scenario = max(
        business_results["scenarios"].items(), key=lambda x: x[1]["net_impact"]
    )

    business_results["optimal_scenario"] = {
        "targeting_percentage": best_scenario[0],
        **best_scenario[1],
    }

    logger.info(
        f"Business metrics calculated for {len(business_results['scenarios'])} targeting scenarios"
    )
    logger.info(
        f"Optimal targeting: {best_scenario[0]} with net impact: ${best_scenario[1]['net_impact']:.0f}"
    )

    return business_results


def compare_models(
    model_results: list[dict], metrics: list[str] | None = None
) -> pd.DataFrame:
    """
    Compare multiple model evaluation results side by side.

    Args:
        model_results: List of evaluation result dictionaries
        metrics: List of metrics to include in comparison

    Returns:
        DataFrame with model comparison
    """
    if metrics is None:
        metrics = [
            "auc_roc",
            "auc_pr",
            "f1_score",
            "precision",
            "recall",
            "precision_at_10%",
            "precision_at_20%",
            "lift_at_10%",
        ]

    comparison_data = []

    for result in model_results:
        model_data = {"model_name": result.get("model_name", "Unknown")}

        for metric in metrics:
            model_data[metric] = result.get(metric, np.nan)

        comparison_data.append(model_data)

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by AUC-ROC (descending)
    if "auc_roc" in comparison_df.columns:
        comparison_df = comparison_df.sort_values("auc_roc", ascending=False)

    logger.info(f"Model comparison generated for {len(comparison_df)} models")

    return comparison_df


def generate_evaluation_report(
    results: dict, business_metrics: dict | None = None, save_path: str | None = None
) -> str | None:
    """
    Generate a comprehensive evaluation report.

    Args:
        results: Evaluation results dictionary
        business_metrics: Business metrics dictionary (optional)
        save_path: Path to save the report (optional)

    Returns:
        Report text content
    """
    model_name = results.get("model_name", "Model")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        "CHURN PREDICTION MODEL EVALUATION REPORT",
        f"Model: {model_name}",
        f"Generated: {timestamp}",
        "=" * 60,
        "",
        "DATASET SUMMARY:",
        f"  Total samples: {results['n_samples']:,}",
        f"  Positive samples: {results['n_positive']:,} ({results['positive_rate']:.1%})",
        f"  Negative samples: {results['n_negative']:,}",
        "",
        "CORE PERFORMANCE METRICS:",
        f"  AUC-ROC: {results['auc_roc']:.4f}",
        f"  AUC-PR: {results['auc_pr']:.4f}",
        f"  F1 Score: {results['f1_score']:.4f}",
        f"  Precision: {results['precision']:.4f}",
        f"  Recall: {results['recall']:.4f}",
        f"  Specificity: {results['specificity']:.4f}",
        "",
        "CONFUSION MATRIX:",
        f"  True Positives: {results['true_positives']:,}",
        f"  False Positives: {results['false_positives']:,}",
        f"  True Negatives: {results['true_negatives']:,}",
        f"  False Negatives: {results['false_negatives']:,}",
        "",
        "PRECISION AT TOP K% USERS:",
        f"  Top 5%: {results.get('precision_at_5%', 0):.4f}",
        f"  Top 10%: {results.get('precision_at_10%', 0):.4f}",
        f"  Top 15%: {results.get('precision_at_15%', 0):.4f}",
        f"  Top 20%: {results.get('precision_at_20%', 0):.4f}",
        "",
        "LIFT ANALYSIS:",
        f"  Lift at 10%: {results.get('lift_at_10%', 0):.2f}x",
        f"  Lift at 20%: {results.get('lift_at_20%', 0):.2f}x",
    ]

    # Add business metrics if provided
    if business_metrics:
        report_lines.extend(
            [
                "",
                "BUSINESS IMPACT ANALYSIS:",
                f"  Intervention cost: ${business_metrics['parameters']['intervention_cost']:.2f} per user",
                f"  Churn value loss: ${business_metrics['parameters']['churn_value_loss']:.2f} per user",
                f"  Intervention success rate: {business_metrics['parameters']['intervention_success_rate']:.1%}",
                "",
                "OPTIMAL TARGETING SCENARIO:",
                f"  Strategy: {business_metrics['optimal_scenario']['targeting_percentage']}",
                f"  Users targeted: {business_metrics['optimal_scenario']['targeted_users']:,}",
                f"  Churners identified: {business_metrics['optimal_scenario']['correctly_identified_churners']:,}",
                f"  Net impact: ${business_metrics['optimal_scenario']['net_impact']:,.0f}",
                f"  ROI: {business_metrics['optimal_scenario']['roi']:.1%}",
            ]
        )

    report_text = "\n".join(report_lines)

    # Save report if path provided
    if save_path:
        with open(save_path, "w") as f:
            f.write(report_text)
        logger.info(f"Evaluation report saved to: {save_path}")

    return report_text


if __name__ == "__main__":
    print("Model Evaluation Module for Churn Prediction")
    print(
        "This module provides comprehensive evaluation tools for classification models."
    )
    print("Use evaluate_classification_model() for complete model evaluation.")
