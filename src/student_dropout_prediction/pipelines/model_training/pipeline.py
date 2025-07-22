from kedro.pipeline import Pipeline, node
from .nodes import train_models, save_model_artifacts

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_models,
                inputs=["X_features", "y_target", "feature_columns"],
                outputs="model_training_results",
                name="train_models_node",
            ),
            node(
                func=save_model_artifacts,
                inputs=[
                    "model_training_results#best_model",
                    "model_training_results#scaler",
                    "label_encoders",
                    "model_training_results#feature_columns",
                    "model_training_results#model_metrics"
                ],
                outputs="model_artifacts",
                name="save_model_artifacts_node",
            ),
        ]
    )