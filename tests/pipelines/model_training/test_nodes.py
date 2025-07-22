import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from student_dropout_prediction.pipelines.model_training.nodes import (
    train_models,
    save_model_artifacts
)

@pytest.fixture
def sample_training_data():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y)
    feature_columns = X.columns.tolist()
    return X, y, feature_columns

def test_train_models(sample_training_data):
    X, y, feature_columns = sample_training_data
    results = train_models(X, y, feature_columns)
    
    assert 'best_model' in results
    assert 'best_model_name' in results
    assert 'model_metrics' in results
    assert results['model_metrics']['f1_score'] > 0 

def test_save_model_artifacts():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    test_artifacts = {
        'model': LogisticRegression(),
        'scaler': StandardScaler(),
        'label_encoders': {'test': None},
        'feature_columns': ['test'],
        'model_metrics': {'test': 1}
    }
    
    result = save_model_artifacts(**test_artifacts)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(test_artifacts.keys())