from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_movie_features, generate_recommendations, compute_cosine_similarity

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
                func=compute_cosine_similarity,
                inputs="movie_features",
                outputs="cosine_sim",
                name="compute_cosine_similarity_node"
            ),
            node(
                func=generate_recommendations,
                inputs=["params:movie_id", "movie_features", "cosine_sim"],
                outputs="recommendations",
                name="generate_recommendations_node"
            )
        ]
    ) 