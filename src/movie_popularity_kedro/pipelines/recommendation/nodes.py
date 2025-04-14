import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def create_movie_features(merged_data: pd.DataFrame) -> pd.DataFrame:
    """Create movie features for recommendation."""
    # Get unique movies with their features
    movies = merged_data[['movieId', 'title', 'genres', 'year']].drop_duplicates()
    
    # Create a text feature combining title, genres, and year
    movies['features'] = movies['title'] + ' ' + movies['genres'] + ' ' + movies['year'].astype(str)
    
    return movies

def generate_recommendations(movies: pd.DataFrame, user_ratings: pd.DataFrame) -> pd.DataFrame:
    """Generate movie recommendations for a user."""
    # Create TF-IDF features
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['features'])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get movies the user has rated highly (>= 4.0)
    user_liked_movies = user_ratings[user_ratings['rating'] >= 4.0]['movieId'].unique()
    
    # Get recommendations
    recommendations = []
    for movie_id in user_liked_movies:
        # Get the index of the movie
        idx = movies[movies['movieId'] == movie_id].index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort movies by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top 5 similar movies
        sim_scores = sim_scores[1:6]  # Skip the movie itself
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Add recommendations
        for idx in movie_indices:
            recommendations.append({
                'movieId': movies.iloc[idx]['movieId'],
                'title': movies.iloc[idx]['title'],
                'genres': movies.iloc[idx]['genres'],
                'year': movies.iloc[idx]['year']
            })
    
    # Convert to DataFrame and remove duplicates
    recommendations_df = pd.DataFrame(recommendations).drop_duplicates()
    
    return recommendations_df 