import json
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple

from sklearn.base import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from matplotlib import pyplot as plt
import shap
import numpy as np

logger = logging.getLogger(__name__)

def preprocess_new_data(
    new_data: pd.DataFrame,
    label_encoders: Dict[str, Any],
    feature_columns: list
) -> pd.DataFrame:
    """Preprocess new data for prediction"""
    logger.info("Preprocessing new data for prediction")
    
    processed_data = new_data.copy()
    
    for col, encoder in label_encoders.items():
        if col in processed_data.columns:
            processed_data[col] = encoder.transform(processed_data[col].astype(str))
    
    missing_cols = set(feature_columns) - set(processed_data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return processed_data[feature_columns]

def make_predictions(
    model: Any,
    scaler: Any,
    processed_data: pd.DataFrame,
    original_data: pd.DataFrame
) -> pd.DataFrame:
    """Make predictions using the trained model"""
    logger.info("Making predictions")
    
    # Scale features
    scaled_data = scaler.transform(processed_data)
    
    # Generate predictions
    predictions = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)[:, 1]
    
    # Return results with original IDs
    return pd.DataFrame({
        "student_id": original_data["student_id"],
        "predicted_dropout_risk": predictions,
        "dropout_probability": probabilities
    })
    
    
def validate_predictions(
    predictions: pd.DataFrame,
    actuals: pd.Series = None,
    threshold: float = 0.5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate model predictions against business rules and optional ground truth.
    
    Args:
        predictions: DataFrame with prediction results
        actuals: Optional Series of true labels for validation
        threshold: Probability threshold for high-risk classification
        
    Returns:
        Tuple of (validated_predictions, validation_metrics)
    """
    validation_metrics = {
        "run_timestamp": pd.Timestamp.now().isoformat(),
        "threshold_used": threshold,
        "total_predictions": len(predictions),
        "high_risk_count": int((predictions["dropout_probability"] > threshold).sum()),
        "missing_values": predictions.isna().sum().to_dict()
    }
    
    # Calculate validation metrics if actuals provided
    if actuals is not None:
        validation_metrics.update({
            "accuracy": float(accuracy_score(actuals, predictions["predicted_dropout_risk"])),
            "precision": float(precision_score(actuals, predictions["predicted_dropout_risk"])),
            "recall": float(recall_score(actuals, predictions["predicted_dropout_risk"])),
            "f1": float(f1_score(actuals, predictions["predicted_dropout_risk"])),
            "roc_auc": float(roc_auc_score(actuals, predictions["dropout_probability"]))
        })
    
    # Add risk flags to predictions
    validated_predictions = predictions.copy()
    validated_predictions["high_risk_flag"] = validated_predictions["dropout_probability"] > threshold
    validated_predictions["risk_category"] = pd.cut(
        validated_predictions["dropout_probability"],
        bins=[0, 0.3, 0.7, 1],
        labels=["low", "medium", "high"]
    )
    
    return validated_predictions, validation_metrics


def explain_predictions(
    model: Any,
    features: pd.DataFrame,
    feature_names: List[str],
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Generate SHAP explanations for model predictions.
    
    Args:
        model: Trained model object
        features: DataFrame of preprocessed features
        feature_names: List of feature names
        sample_size: Number of samples to use for explanation
        
    Returns:
        Dictionary containing explanation artifacts
    """
    # Sample data if large dataset
    if len(features) > sample_size:
        features = features.sample(sample_size, random_state=42)
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    
    # Generate plots
    plt.switch_backend('Agg')  # Prevent GUI window from opening
    plot_dir = Path("data/08_reporting/shap_plots")
    plot_dir.mkdir(exist_ok=True)
    
    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, features, feature_names=feature_names, show=False)
    plt.tight_layout()
    summary_path = str(plot_dir / "shap_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Force plot for first prediction
    plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_values[0][0,:],
        features.iloc[0,:],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    force_path = str(plot_dir / "shap_force.png")
    plt.savefig(force_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        "shap_values": shap_values,
        "expected_value": float(explainer.expected_value),
        "feature_importance": dict(zip(
            feature_names,
            np.abs(shap_values).mean(0).tolist()
        )),
        "plot_paths": {
            "summary_plot": summary_path,
            "force_plot": force_path
        }
    }
    
    

def generate_prediction_report(
    predictions: pd.DataFrame,
    validation_metrics: Dict[str, Any],
    explanations: Dict[str, Any],
    report_dir: str = "data/08_reporting"
) -> Dict[str, Any]:
    """
    Generate comprehensive prediction report with metrics and explanations.
    
    Args:
        predictions: Validated predictions DataFrame
        validation_metrics: Dictionary of validation metrics
        explanations: Dictionary of SHAP explanations
        report_dir: Directory to save report artifacts
        
    Returns:
        Dictionary containing report data and paths
    """
    # Create report directory
    report_path = Path(report_dir)
    report_path.mkdir(exist_ok=True)
    
    # Generate summary statistics
    risk_distribution = predictions["risk_category"].value_counts(normalize=True).to_dict()
    prob_stats = predictions["dropout_probability"].describe().to_dict()
    
    # Create report dictionary
    report = {
        "metadata": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "model_version": "1.0",
            "report_id": f"report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        },
        "statistics": {
            "risk_distribution": risk_distribution,
            "probability_stats": prob_stats,
            "validation_metrics": validation_metrics
        },
        "explanations": {
            "top_features": dict(sorted(
                explanations["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            "expected_value": explanations["expected_value"],
            "plot_paths": explanations["plot_paths"]
        },
        "sample_predictions": predictions.head(10).to_dict("records")
    }
    
    # Save report files
    report_json_path = report_path / "prediction_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Save sample predictions as CSV
    sample_csv_path = report_path / "sample_predictions.csv"
    predictions.head(100).to_csv(sample_csv_path, index=False)
    
    return {
        "report_data": report,
        "report_paths": {
            "json_report": str(report_json_path),
            "sample_csv": str(sample_csv_path),
            "shap_plots": explanations["plot_paths"]
        }
    }