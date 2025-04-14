"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

from movie_popularity_kedro.pipelines import data_engineering, recommendation

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_engineering_pipeline = data_engineering.create_pipeline()
    recommendation_pipeline = recommendation.create_pipeline()
    
    # Create a combined pipeline that runs data engineering first, then recommendation
    combined_pipeline = data_engineering_pipeline + recommendation_pipeline
    
    return {
        "data_engineering": data_engineering_pipeline,
        "recommendation": recommendation_pipeline,
        "__default__": combined_pipeline,  # This will run both pipelines in sequence
    }
