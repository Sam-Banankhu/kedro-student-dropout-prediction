import pandas as pd
import logging
from typing import Dict, Any

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