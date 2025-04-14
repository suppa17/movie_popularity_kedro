from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_movie_features, generate_recommendations

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_movie_features,
                inputs="merged_data",
                outputs="movie_features",
                name="create_movie_features_node",
            ),
            node(
                func=generate_recommendations,
                inputs=["movie_features", "ratings"],
                outputs="movie_recommendations",
                name="generate_recommendations_node",
            ),
        ]
    ) 