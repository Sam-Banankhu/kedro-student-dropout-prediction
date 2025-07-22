from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Train and evaluate a Random Forest model"""
    logger.info("Starting model training")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Return as tuple matching the output names
    return model, scaler, X_test, y_test