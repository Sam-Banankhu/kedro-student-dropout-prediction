from kedro.pipeline import Pipeline, node
from .nodes import train_model

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_model,
                inputs=["X_features", "y_target"],
                outputs=["trained_model", "feature_scaler", "test_features", "test_targets"],
                name="train_model_node",
            )
        ]
    )