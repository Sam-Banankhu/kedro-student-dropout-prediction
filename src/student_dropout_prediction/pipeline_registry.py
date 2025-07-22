from typing import Dict
from kedro.pipeline import Pipeline

from student_dropout_prediction.pipelines.data_processing import pipeline as dp_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines."""
    data_processing_pipeline = dp_pipeline.create_pipeline()
    
    return {
        "__default__": data_processing_pipeline,
        "data_processing": data_processing_pipeline,
    }