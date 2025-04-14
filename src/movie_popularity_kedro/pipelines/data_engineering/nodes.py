import pandas as pd

def preprocess_movies(movies: pd.DataFrame) -> pd.DataFrame:
    movies[['title_clean', 'year']] = movies['title'].str.extract(r'^(.*) \((\d{4})\)$')
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
    return movies

def merge_data(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(ratings, movies, on="movieId", how="left")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model(merged_data: pd.DataFrame) -> dict:
    movie_stats = merged_data.groupby("movieId").agg({
        "rating": ["mean", "count"],
        "year": "first"
    })
    movie_stats.columns = ["avg_rating", "rating_count", "year"]
    movie_stats = movie_stats.dropna()

    # Label: Popular = 1 if avg_rating >= 4.0
    movie_stats["popular"] = (movie_stats["avg_rating"] >= 4.0).astype(int)

    # Features
    X = movie_stats[["rating_count", "year"]]
    y = movie_stats["popular"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    return {"accuracy": acc}
