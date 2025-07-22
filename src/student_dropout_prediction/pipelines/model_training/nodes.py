import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import json
import logging

logger = logging.getLogger(__name__)

def train_models(X: pd.DataFrame, y: pd.Series, feature_columns: list) -> dict:
    """Train multiple models and find the best one"""
    logger.info("Starting model training...")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models and their hyperparameters
    model_configs = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [3, 5, 10, 15],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 4, 8]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    }
    
    # Train and evaluate models
    results = []
    trained_models = {}
    
    for name, config in model_configs.items():
        logger.info(f"Training {name}...")
        
        # Grid search for hyperparameter tuning
        grid = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid.fit(X_scaled, y)
        best_model = grid.best_estimator_
        
        # Cross-validation metrics
        cv_scores = {
            'accuracy': cross_val_score(best_model, X_scaled, y, cv=5, scoring='accuracy'),
            'precision': cross_val_score(best_model, X_scaled, y, cv=5, scoring='precision'),
            'recall': cross_val_score(best_model, X_scaled, y, cv=5, scoring='recall'),
            'f1': cross_val_score(best_model, X_scaled, y, cv=5, scoring='f1'),
            'roc_auc': cross_val_score(best_model, X_scaled, y, cv=5, scoring='roc_auc')
        }
        
        # Store model and metrics
        trained_models[name] = best_model
        
        # Calculate mean metrics
        mean_metrics = {
            'model_name': name,
            'accuracy': round(cv_scores['accuracy'].mean(), 4),
            'precision': round(cv_scores['precision'].mean(), 4),
            'recall': round(cv_scores['recall'].mean(), 4),
            'f1_score': round(cv_scores['f1'].mean(), 4),
            'roc_auc': round(cv_scores['roc_auc'].mean(), 4),
            'best_params': grid.best_params_
        }
        
        results.append(mean_metrics)
        logger.info(f"{name} - {mean_metrics}")
    
    # Choose the best model based on F1 Score
    best_model_info = max(results, key=lambda x: x['f1_score'])
    best_model_name = best_model_info['model_name']
    best_model = trained_models[best_model_name]
    
    logger.info(f"Best Model: {best_model_name} with F1 Score: {best_model_info['f1_score']}")
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'model_metrics': best_model_info,
        'all_models': trained_models,
        'scaler': scaler,
        'feature_columns': feature_columns
    }

def save_model_artifacts(
    best_model: object,
    scaler: StandardScaler,
    label_encoders: dict,
    feature_columns: list,
    model_metrics: dict
) -> None:
    """Save model artifacts to disk"""
    artifacts = {
        'model': best_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns,
        'model_metrics': model_metrics
    }
    return artifacts