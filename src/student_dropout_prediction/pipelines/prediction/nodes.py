import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_model_artifacts(model_artifacts: Dict[str, Any]) -> tuple:
    """Unpack model artifacts for prediction"""
    return (
        model_artifacts['model'],
        model_artifacts['scaler'],
        model_artifacts['label_encoders'],
        model_artifacts['feature_columns']
    )

def preprocess_new_data(
    new_data: pd.DataFrame,
    label_encoders: Dict[str, Any],
    feature_columns: list
) -> pd.DataFrame:
    """Preprocess new data for prediction"""
    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in new_data.columns:
            new_data[col] = encoder.transform(new_data[col].astype(str))
    
    # Ensure all required features are present
    missing_cols = set(feature_columns) - set(new_data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return new_data[feature_columns]

def make_predictions(
    model: Any,
    scaler: Any,
    processed_data: pd.DataFrame,
    original_data: pd.DataFrame
) -> pd.DataFrame:
    """Make predictions using the trained model"""
    # Scale features
    X_scaled = scaler.transform(processed_data)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Return results with original data
    return pd.DataFrame({
        "student_id": original_data.get("student_id", range(len(predictions))),
        "dropout_risk": predictions,
        "dropout_probability": probabilities
    })

def save_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Save predictions with timestamp"""
    predictions['prediction_time'] = pd.Timestamp.now()
    return predictions