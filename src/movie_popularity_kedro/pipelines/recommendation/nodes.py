import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


import numpy as np

def create_movie_features(merged_data: pd.DataFrame) -> pd.DataFrame:
    """Create movie features for recommendation."""
    # Get unique movies with their features
    movies = merged_data[['movieId', 'title', 'genres', 'year']].drop_duplicates()
    
    # Create a text feature combining title, genres, and year
    movies['features'] = movies['title'] + ' ' + movies['genres'] + ' ' + movies['year'].astype(str)
    
    return movies

def generate_recommendations(movie_id: int, movies: pd.DataFrame, cosine_sim: np.ndarray) -> pd.DataFrame:
    """Generate movie recommendations for a user."""
    # Ensure the movie_id exists
    if movie_id not in movies["movieId"].values:
        raise ValueError(f"Movie ID {movie_id} not found in dataset.")

    # Get the row index for the given movieId
    idx_series = movies[movies["movieId"] == movie_id].index
    if idx_series.empty:
        raise ValueError(f"No index found for movie ID {movie_id}")
    idx = idx_series[0]

    # Check if index is in bounds for cosine_sim matrix
    if idx >= cosine_sim.shape[0]:
        raise ValueError(f"Index {idx} is out of bounds for cosine_sim (shape={cosine_sim.shape})")

    # Proceed with similarity ranking
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:11]]  # Get top 10 similar movies (excluding itself)

    # Get the recommended movies
    recommended_movies = movies.iloc[movie_indices][['movieId', 'title', 'genres', 'year']]
    
    return recommended_movies

from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(movies: pd.DataFrame) -> np.ndarray:
    """Compute cosine similarity between movies based on their genres."""
    # Create genre features
    genre_features = movies["genres"].str.get_dummies(sep="|")
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(genre_features)
    
    return cosine_sim
