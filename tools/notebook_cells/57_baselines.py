def build_popularity_table(ratings_df, min_ratings=20):
    stats = ratings_df.groupby("movieId").agg(
        avg_rating=("rating", "mean"), num_ratings=("rating", "count")
    ).reset_index()
    stats = stats[stats["num_ratings"] >= min_ratings].copy()
    stats["bayesian_popularity"] = (
        stats["avg_rating"] * stats["num_ratings"] + ratings_df["rating"].mean() * min_ratings
    ) / (stats["num_ratings"] + min_ratings)
    return stats.sort_values(["bayesian_popularity", "num_ratings"], ascending=False)


def get_popular_recommendations(popularity_table, ratings_df, user_id, top_n=10):
    rated_movies = set(ratings_df[ratings_df["userId"] == user_id]["movieId"].values)
    return [m for m in popularity_table["movieId"].tolist() if m not in rated_movies][:top_n]


def get_random_recommendations(ratings_df, user_id, candidate_movie_ids, top_n=10, random_state=RANDOM_STATE):
    rated_movies = set(ratings_df[ratings_df["userId"] == user_id]["movieId"].values)
    candidates = sorted(set(candidate_movie_ids) - rated_movies)
    if not candidates:
        return []
    rng = np.random.default_rng(random_state + int(user_id))
    return rng.choice(candidates, size=min(top_n, len(candidates)), replace=False).tolist()


def fit_bias_only_model(ratings_df, reg=10.0, n_iters=10):
    mu = ratings_df["rating"].mean()
    user_bias = pd.Series(0.0, index=ratings_df["userId"].unique())
    item_bias = pd.Series(0.0, index=ratings_df["movieId"].unique())
    for _ in range(n_iters):
        residual_u = ratings_df.assign(
            resid=ratings_df["rating"] - mu - ratings_df["movieId"].map(item_bias).fillna(0.0)
        )
        user_bias = residual_u.groupby("userId")["resid"].sum() / (
            reg + residual_u.groupby("userId").size()
        )
        residual_i = ratings_df.assign(
            resid=ratings_df["rating"] - mu - ratings_df["userId"].map(user_bias).fillna(0.0)
        )
        item_bias = residual_i.groupby("movieId")["resid"].sum() / (
            reg + residual_i.groupby("movieId").size()
        )
    return {"mu": mu, "user_bias": user_bias.to_dict(), "item_bias": item_bias.to_dict()}


def get_bias_only_recommendations(model, ratings_df, user_id, candidate_movie_ids, top_n=10):
    rated_movies = set(ratings_df[ratings_df["userId"] == user_id]["movieId"].values)
    bu = model["user_bias"].get(user_id, 0.0)
    rows = [
        (movie_id, model["mu"] + bu + model["item_bias"].get(movie_id, 0.0))
        for movie_id in candidate_movie_ids
        if movie_id not in rated_movies
    ]
    return [movie_id for movie_id, _ in sorted(rows, key=lambda x: x[1], reverse=True)[:top_n]]


def build_user_item_matrix(ratings_df, threshold=LIKED_THRESHOLD):
    implicit_df = ratings_df.copy()
    implicit_df["interaction"] = (implicit_df["rating"] >= threshold).astype(float)
    users = sorted(implicit_df["userId"].unique())
    movies = sorted(implicit_df["movieId"].unique())
    u2i = {u: i for i, u in enumerate(users)}
    m2i = {m: i for i, m in enumerate(movies)}
    matrix = np.zeros((len(users), len(movies)), dtype=np.float32)
    for row in implicit_df.itertuples(index=False):
        matrix[u2i[row.userId], m2i[row.movieId]] = row.interaction
    return matrix, u2i, m2i, {i: m for m, i in m2i.items()}


def fit_item_knn(ratings_df):
    matrix, u2i, m2i, i2m = build_user_item_matrix(ratings_df)
    sim = cosine_similarity(matrix.T)
    np.fill_diagonal(sim, 0.0)
    return {"matrix": matrix, "user_to_idx": u2i, "movie_to_idx": m2i, "idx_to_movie": i2m, "similarity": sim}


def get_item_knn_recommendations(model, ratings_df, user_id, top_n=10):
    if user_id not in model["user_to_idx"]:
        return []
    u = model["user_to_idx"][user_id]
    scores = model["matrix"][u].dot(model["similarity"])
    rated_movies = set(ratings_df[ratings_df["userId"] == user_id]["movieId"].values)
    rows = [
        (model["idx_to_movie"][idx], score)
        for idx, score in enumerate(scores)
        if model["idx_to_movie"][idx] not in rated_movies
    ]
    return [movie_id for movie_id, _ in sorted(rows, key=lambda x: x[1], reverse=True)[:top_n]]


def fit_user_knn(ratings_df):
    matrix, u2i, m2i, i2m = build_user_item_matrix(ratings_df)
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, 0.0)
    return {"matrix": matrix, "user_to_idx": u2i, "movie_to_idx": m2i, "idx_to_movie": i2m, "similarity": sim}


def get_user_knn_recommendations(model, ratings_df, user_id, top_n=10):
    if user_id not in model["user_to_idx"]:
        return []
    u = model["user_to_idx"][user_id]
    scores = model["similarity"][u].dot(model["matrix"])
    rated_movies = set(ratings_df[ratings_df["userId"] == user_id]["movieId"].values)
    rows = [
        (model["idx_to_movie"][idx], score)
        for idx, score in enumerate(scores)
        if model["idx_to_movie"][idx] not in rated_movies
    ]
    return [movie_id for movie_id, _ in sorted(rows, key=lambda x: x[1], reverse=True)[:top_n]]


popularity_table_train = build_popularity_table(train_ratings_v2)
popularity_table_dev = build_popularity_table(development_ratings)
bias_only_train = fit_bias_only_model(train_ratings_v2)
bias_only_dev = fit_bias_only_model(development_ratings)
item_knn_train = fit_item_knn(train_ratings_v2)
item_knn_dev = fit_item_knn(development_ratings)
user_knn_train = fit_user_knn(train_ratings_v2)
user_knn_dev = fit_user_knn(development_ratings)
all_movie_ids = movie_meta["movieId"].tolist()
