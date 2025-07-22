import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

def load_and_preprocess_data(primary_school_data: pd.DataFrame) -> pd.DataFrame:
    """Load and preprocess the dataset"""
    logger.info("Starting data preprocessing")
    
    df = primary_school_data.copy()  # Work with a copy of the input data
    
    # Handle invalid scores (replace zeros with median)
    score_columns = ['english_score', 'chichewa_score', 'maths_score', 
                    'primary_science_score', 'social_religious_score', 
                    'life_skills_expressive_arts_score']
    
    for col in score_columns:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan).fillna(df[col].median())

    df['term_avg_score'] = df[score_columns].sum(axis=1)
    df['school_attendance_rate'] = df['present_days'] / (df['present_days'] + df['absent_days'])

    df['bullying_incidents_total'] = df.groupby('student_id')['bullying_reported'].transform('sum')

    agg_dict = {
        'term_avg_score': 'mean',
        'school_attendance_rate': 'mean',
        'bullying_incidents_total': 'max',
        'class_repetitions': 'max',
        'distance_to_school': 'mean',
        'special_learning': 'max',
        'household_income': 'first',
        'orphan_status': 'first',
        'standard': 'max',
        'age': 'mean',
        'gender': 'first'
    }
    
    student_df = df.groupby('student_id').agg(agg_dict).reset_index()
    
    # Create target variable (dropout_risk)
    student_df['dropout_risk'] = student_df.apply(_assign_dropout_risk, axis=1)
    
    return student_df

def _assign_dropout_risk(row) -> int:
    """Assign dropout risk based on multiple conditions"""
    conditions = [
        row['term_avg_score'] < 300,
        row['school_attendance_rate'] < 0.8,
        row['class_repetitions'] > 0,
        row['household_income'] == 'low',
        row['bullying_incidents_total'] > 5,
        row['age'] > (6 + row['standard'] + row['class_repetitions'])
    ]
    return 1 if sum(conditions) >= 3 else 0

def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features for model training"""
    categorical_cols = ['household_income', 'orphan_status', 'gender']
    label_encoders = {}
    
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))
    
    feature_columns = ['term_avg_score', 'school_attendance_rate', 
                      'bullying_incidents_total', 'class_repetitions', 
                      'distance_to_school', 'special_learning', 
                      'household_income', 'orphan_status', 
                      'standard', 'age', 'gender']
    
    X = df[feature_columns]
    y = df['dropout_risk']
    
    return X, y, label_encoders, feature_columns