from kedro.pipeline import Pipeline, node
from .nodes import load_and_preprocess_data, prepare_features

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=load_and_preprocess_data,
                inputs="primary_school_data",  # This comes from catalog
                outputs="preprocessed_student_data",
                name="load_and_preprocess_data_node",
            ),
            node(
                func=prepare_features,
                inputs="preprocessed_student_data",
                outputs=["X_features", "y_target", "label_encoders", "feature_columns"],
                name="prepare_features_node",
            ),
        ]
    )