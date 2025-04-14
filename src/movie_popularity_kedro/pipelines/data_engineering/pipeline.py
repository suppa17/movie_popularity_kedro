from kedro.pipeline import Pipeline, node
from .nodes import preprocess_movies, merge_data, train_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=preprocess_movies,
            inputs="movies",
            outputs="clean_movies",
            name="preprocess_movies_node"
        ),
        node(
            func=merge_data,
            inputs=["clean_movies", "ratings"],
            outputs="merged_data",
            name="merge_data_node"
        ),
        node(
            func=train_model,
            inputs="merged_data",
            outputs="model_metrics",
            name="train_model_node"
        )
    ])
