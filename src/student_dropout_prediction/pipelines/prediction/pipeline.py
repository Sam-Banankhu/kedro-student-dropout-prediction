from kedro.pipeline import Pipeline, node
from .nodes import (
    load_model_artifacts,
    preprocess_new_data,
    make_predictions,
    save_predictions
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=load_model_artifacts,
                inputs="model_artifacts",
                outputs=["prediction_model", "prediction_scaler", 
                        "prediction_label_encoders", "prediction_feature_columns"],
                name="load_model_artifacts_node",
            ),
            node(
                func=preprocess_new_data,
                inputs=["new_data", "prediction_label_encoders", "prediction_feature_columns"],
                outputs="processed_prediction_data",
                name="preprocess_new_data_node",
            ),
            node(
                func=make_predictions,
                inputs=["prediction_model", "prediction_scaler", 
                       "processed_prediction_data", "new_data"],
                outputs="raw_predictions",
                name="make_predictions_node",
            ),
            node(
                func=save_predictions,
                inputs="raw_predictions",
                outputs="final_predictions",
                name="save_predictions_node",
            ),
        ]
    )