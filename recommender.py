import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, movie_meta, ratings, content_features, svd_model, mappings, hybrid_config=None, popularity_table=None, blend_model=None):
        self.movie_meta = movie_meta.reset_index(drop=True).copy()
        self.ratings = ratings.copy()
        self.content_features = content_features
        self.svd_model = svd_model
        self.mappings = mappings
        self.hybrid_config = hybrid_config or {}
        self.popularity_table = popularity_table
        self.blend_model = blend_model

        self.P = svd_model["P"]
        self.Q = svd_model["Q"]
        self.bu = svd_model.get("bu")
        self.bi = svd_model.get("bi")
        self.mu = svd_model.get("mu", 0.0)
        self.user_to_idx = svd_model["user_to_idx"]
        self.movie_to_idx = svd_model["movie_to_idx"]
        self.idx_to_movie = svd_model["idx_to_movie"]
        self.rating_min = svd_model.get("rating_min", 0.5)
        self.rating_max = svd_model.get("rating_max", 5.0)

        self.movieid_to_content_idx = mappings.get(
            "movieid_to_content_idx",
            {mid: idx for idx, mid in enumerate(self.movie_meta["movieId"])},
        )

    @staticmethod
    def normalize_series(series):
        series = pd.Series(series).astype(float)
        min_val = series.min()
        max_val = series.max()
        if pd.isna(min_val) or pd.isna(max_val):
            return pd.Series([0.0] * len(series), index=series.index)
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)

    def predict_rating(self, user_id, movie_id):
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return self.mu
        u = self.user_to_idx[user_id]
        i = self.movie_to_idx[movie_id]
        pred = self.mu
        if self.bu is not None:
            pred += self.bu[u]
        if self.bi is not None:
            pred += self.bi[i]
        pred += np.dot(self.P[u], self.Q[i])
        return float(np.clip(pred, self.rating_min, self.rating_max))

    def get_content_based_recommendations(self, movie_title, top_n=10):
        movie_title = str(movie_title).strip().lower()
        matches = self.movie_meta[
            self.movie_meta["title"].str.lower().str.contains(movie_title, na=False, regex=False)
        ]
        if matches.empty:
            return pd.DataFrame()
        idx = matches.index[0]
        sims = cosine_similarity(self.content_features[idx].reshape(1, -1), self.content_features).ravel()
        sim_scores = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[1 : top_n + 1]
        movie_indices = [i for i, _ in sim_scores]
        scores = [s for _, s in sim_scores]
        recs = self.movie_meta.iloc[movie_indices][["movieId", "id", "title"]].copy()
        recs["similarity_score"] = scores
        recs["rank"] = range(1, len(recs) + 1)
        return recs[["rank", "movieId", "id", "title", "similarity_score"]]

    def get_mf_recommendations(self, user_id, top_n=10, ratings_data=None):
        ratings_data = self.ratings if ratings_data is None else ratings_data
        if user_id not in self.user_to_idx:
            return pd.DataFrame()
        rated_movies = set(ratings_data[ratings_data["userId"] == user_id]["movieId"].values)
        rows = [
            {"movieId": movie_id, "predicted_rating": self.predict_rating(user_id, movie_id)}
            for movie_id in self.movie_to_idx.keys()
            if movie_id not in rated_movies
        ]
        recs = pd.DataFrame(rows)
        if recs.empty:
            return recs
        recs = recs.sort_values("predicted_rating", ascending=False).head(top_n)
        recs = recs.merge(self.movie_meta[["movieId", "id", "title"]], on="movieId", how="left")
        recs["rank"] = range(1, len(recs) + 1)
        return recs[["rank", "movieId", "id", "title", "predicted_rating"]]

    def build_user_content_profile(self, user_id, ratings_data=None, rating_threshold=3.5, negative_threshold=2.5, negative_weight=0.25):
        ratings_data = self.ratings if ratings_data is None else ratings_data
        positive = ratings_data[
            (ratings_data["userId"] == user_id) & (ratings_data["rating"] >= rating_threshold)
        ]
        positive = positive[positive["movieId"].isin(self.movieid_to_content_idx)]
        if positive.empty:
            return None
        positive_indices = [self.movieid_to_content_idx[mid] for mid in positive["movieId"].values]
        positive_profile = np.average(self.content_features[positive_indices], axis=0, weights=positive["rating"].values)

        negative = ratings_data[
            (ratings_data["userId"] == user_id) & (ratings_data["rating"] <= negative_threshold)
        ]
        negative = negative[negative["movieId"].isin(self.movieid_to_content_idx)]
        if negative.empty:
            return positive_profile
        negative_indices = [self.movieid_to_content_idx[mid] for mid in negative["movieId"].values]
        negative_profile = np.average(self.content_features[negative_indices], axis=0, weights=(5.5 - negative["rating"]).values)
        return positive_profile - negative_weight * negative_profile

    def build_candidate_frame(self, user_id, ratings_data):
        rated_movies = set(ratings_data[ratings_data["userId"] == user_id]["movieId"].values)
        profile = self.build_user_content_profile(
            user_id,
            ratings_data,
            rating_threshold=self.hybrid_config.get("liked_threshold", 3.5),
        )
        popularity_lookup = {}
        if self.popularity_table is not None and not self.popularity_table.empty:
            popularity_lookup = self.popularity_table.set_index("movieId")["bayesian_popularity"].to_dict()
        rows = []
        for _, row in self.movie_meta.iterrows():
            movie_id = row["movieId"]
            if movie_id in rated_movies:
                continue
            cf_score = self.predict_rating(user_id, movie_id)
            if profile is not None and movie_id in self.movieid_to_content_idx:
                movie_idx = self.movieid_to_content_idx[movie_id]
                cbf_score = cosine_similarity(profile.reshape(1, -1), self.content_features[movie_idx].reshape(1, -1))[0][0]
            else:
                cbf_score = 0.0
            rows.append(
                {
                    "movieId": movie_id,
                    "id": row.get("id"),
                    "title": row["title"],
                    "cf_score": cf_score,
                    "cbf_score": cbf_score,
                    "popularity_score": popularity_lookup.get(movie_id, 0.0),
                }
            )
        return pd.DataFrame(rows)

    def get_hybrid_recommendations(self, user_id, top_n=10, alpha=0.6, ratings_data=None):
        ratings_data = self.ratings if ratings_data is None else ratings_data
        if user_id not in self.user_to_idx:
            return pd.DataFrame()
        recs = self.build_candidate_frame(user_id, ratings_data)
        if recs.empty:
            return recs
        if self.blend_model is not None and self.hybrid_config.get("serving_mode") == "blender":
            features = self.hybrid_config.get("blender_features", ["cf_score", "cbf_score", "popularity_score"])
            recs["hybrid_score"] = self.blend_model.predict_proba(recs[features])[:, 1]
            recs = recs.sort_values("hybrid_score", ascending=False).head(top_n).copy()
            recs["rank"] = range(1, len(recs) + 1)
            return recs[["rank", "movieId", "id", "title", "cf_score", "cbf_score", "popularity_score", "hybrid_score"]]
        recs["norm_cf"] = self.normalize_series(recs["cf_score"])
        recs["norm_cbf"] = self.normalize_series(recs["cbf_score"])
        recs["norm_popularity"] = self.normalize_series(recs["popularity_score"])
        if self.hybrid_config.get("weights"):
            w_cf, w_cbf, w_pop = self.hybrid_config["weights"]
            recs["hybrid_score"] = w_cf * recs["norm_cf"] + w_cbf * recs["norm_cbf"] + w_pop * recs["norm_popularity"]
        else:
            recs["hybrid_score"] = alpha * recs["norm_cf"] + (1 - alpha) * recs["norm_cbf"]
        recs = recs.sort_values("hybrid_score", ascending=False).head(top_n).copy()
        recs["rank"] = range(1, len(recs) + 1)
        return recs[
            [
                "rank",
                "movieId",
                "id",
                "title",
                "cf_score",
                "cbf_score",
                "popularity_score",
                "norm_cf",
                "norm_cbf",
                "norm_popularity",
                "hybrid_score",
            ]
        ]
