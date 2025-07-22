import pytest
import pandas as pd
import numpy as np
from student_dropout_prediction.pipelines.data_processing.nodes import (
    load_and_preprocess_data,
    _assign_dropout_risk,
    prepare_features
)

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'student_id': [1, 1, 2, 2],
        'english_score': [80, 85, 0, 75],
        'present_days': [90, 95, 80, 85],
        'absent_days': [10, 5, 20, 15],
        'bullying_reported': [0, 1, 3, 0],
        'class_repetitions': [0, 0, 1, 1],
        'distance_to_school': [5, 5, 10, 10],
        'household_income': ['low', 'low', 'medium', 'medium'],
        'orphan_status': ['no', 'no', 'yes', 'yes'],
        'standard': [4, 4, 5, 5],
        'age': [10, 10, 12, 12],
        'gender': ['M', 'M', 'F', 'F']
    })

def test_load_and_preprocess_data(sample_data, tmp_path):
    """Test loading and preprocessing data"""
    file_path = tmp_path / "test_data.csv"
    sample_data.to_csv(file_path, index=False)
    
    # Test function
    result = load_and_preprocess_data(str(file_path))
    
    # Assertions
    assert isinstance(result, pd.DataFrame)
    assert 'dropout_risk' in result.columns
    assert result['english_score'].min() > 0  
    assert len(result) == 2  

def test_assign_dropout_risk():
    test_row = pd.Series({
        'term_avg_score': 250,
        'school_attendance_rate': 0.7,
        'class_repetitions': 1,
        'household_income': 'low',
        'bullying_incidents_total': 6,
        'age': 13,
        'standard': 5
    })
    assert _assign_dropout_risk(test_row) == 1
    
    test_row['term_avg_score'] = 400
    assert _assign_dropout_risk(test_row) == 0

def test_prepare_features(sample_data):
    processed_data = load_and_preprocess_data(sample_data)
    X, y, label_encoders, feature_columns = prepare_features(processed_data)
    
    assert X.shape[0] == y.shape[0]
    assert len(feature_columns) == X.shape[1]
    assert set(['household_income', 'orphan_status', 'gender']).issubset(label_encoders.keys())