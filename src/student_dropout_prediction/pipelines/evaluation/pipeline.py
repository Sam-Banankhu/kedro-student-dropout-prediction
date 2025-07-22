from kedro.pipeline import Pipeline, node
from .nodes import evaluate_model

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=evaluate_model,
                inputs=["trained_model", "test_features", "test_targets"],
                outputs="model_metrics",
                name="evaluate_model_node",
            )
        ]
    )