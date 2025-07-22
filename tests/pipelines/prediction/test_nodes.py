import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from student_dropout_prediction.pipelines.prediction.nodes import (
    load_model_artifacts,
    preprocess_new_data,
    make_predictions,
    save_predictions
)

@pytest.fixture
def sample_model_artifacts():
    model = LogisticRegression()
    model.coef_ = np.array([[1, 1]]) 
    model.intercept_ = np.array([0])
    model.classes_ = np.array([0, 1])
    
    return {
        'model': model,
        'scaler': StandardScaler(),
        'label_encoders': {'test_feature': LabelEncoder().fit(['a', 'b'])},
        'feature_columns': ['numeric_feature', 'test_feature'],
        'model_metrics': {}
    }

@pytest.fixture
def sample_new_data():
    return pd.DataFrame({
        'student_id': [1, 2],
        'numeric_feature': [1.0, 2.0],
        'test_feature': ['a', 'b'],
        'other_feature': ['x', 'y']
    })

def test_load_model_artifacts(sample_model_artifacts):
    result = load_model_artifacts(sample_model_artifacts)
    assert len(result) == 4

def test_preprocess_new_data(sample_new_data, sample_model_artifacts):
    _, _, label_encoders, feature_columns = load_model_artifacts(sample_model_artifacts)
    processed = preprocess_new_data(sample_new_data, label_encoders, feature_columns)
    assert set(feature_columns).issubset(processed.columns)

def test_make_predictions(sample_new_data, sample_model_artifacts):
    model, scaler, label_encoders, feature_columns = load_model_artifacts(sample_model_artifacts)
    processed = preprocess_new_data(sample_new_data, label_encoders, feature_columns)
    scaler.fit(processed) 
    
    predictions = make_predictions(model, scaler, processed, sample_new_data)
    assert len(predictions) == len(sample_new_data)
    assert 'dropout_risk' in predictions.columns

def test_save_predictions():
    test_preds = pd.DataFrame({'student_id': [1], 'dropout_risk': [0], 'dropout_probability': [0.2]})
    result = save_predictions(test_preds)
    assert 'prediction_time' in result.columns