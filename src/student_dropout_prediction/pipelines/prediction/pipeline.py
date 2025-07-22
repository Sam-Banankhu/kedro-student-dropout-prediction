from kedro.pipeline import Pipeline, node
from .nodes import (
    preprocess_new_data,
    make_predictions,
    validate_predictions,
    explain_predictions,
    generate_prediction_report
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_new_data,
                inputs=["new_student_data", "label_encoders", "feature_columns"],
                outputs="processed_prediction_data",
                name="preprocess_new_data_node",
            ),
            node(
                func=make_predictions,
                inputs=["trained_model", "feature_scaler", "processed_prediction_data", "new_student_data"],
                outputs="student_predictions",
                name="make_predictions_node",
            ),
            node(
                func=validate_predictions,
                inputs=["student_predictions"],
                outputs="validated_predictions",
                name="validate_predictions_node",
            ),
            node(
                func=explain_predictions,
                inputs=['model', 'features', 'feature_names', 'sample_size'],
                outputs="explanation_reports",
                name="explain_predictions_node",
            ),
            node(
                func=generate_prediction_report,
                inputs=['predictions', 'validation_metrics', 'explanations', 'report_dir'],
                outputs="prediction_reports",
                name="generate_prediction_report_node",
            )
        ]
    )