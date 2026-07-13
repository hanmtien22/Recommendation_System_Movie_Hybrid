from pathlib import Path
import pickle
import pandas as pd


def load_data(data_dir="data"):
    data_dir = Path(data_dir)
    movie_meta = pd.read_csv(data_dir / "movie_meta_processed.csv")
    ratings = pd.read_csv(data_dir / "final_ratings_processed.csv")
    movies = pd.read_csv(data_dir / "movies.csv") if (data_dir / "movies.csv").exists() else None
    links = pd.read_csv(data_dir / "links.csv") if (data_dir / "links.csv").exists() else None
    return movie_meta, ratings, movies, links


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_models(models_dir="models"):
    models_dir = Path(models_dir)
    tfidf = load_pickle(models_dir / "tfidf_vectorizer.pkl")
    content_features = load_pickle(models_dir / "content_features.pkl")
    svd_model = load_pickle(models_dir / "svd_model.pkl")
    mappings = load_pickle(models_dir / "mappings.pkl")
    hybrid_config = load_pickle(models_dir / "hybrid_config.pkl") if (models_dir / "hybrid_config.pkl").exists() else {}
    popularity_table = load_pickle(models_dir / "popularity_table.pkl") if (models_dir / "popularity_table.pkl").exists() else None
    blend_model = load_pickle(models_dir / "blend_model.pkl") if (models_dir / "blend_model.pkl").exists() else None
    return tfidf, content_features, svd_model, mappings, hybrid_config, popularity_table, blend_model
