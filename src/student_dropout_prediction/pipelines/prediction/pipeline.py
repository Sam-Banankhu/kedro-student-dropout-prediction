from kedro.pipeline import Pipeline, node
from .nodes import preprocess_new_data, make_predictions

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
            )
        ]
    )