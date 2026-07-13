def prepare_cf_data(ratings_df):
    cf_df = ratings_df[["userId", "movieId", "rating"]].copy()
    users = cf_df["userId"].unique()
    movies = cf_df["movieId"].unique()
    u2i = {user_id: idx for idx, user_id in enumerate(users)}
    m2i = {movie_id: idx for idx, movie_id in enumerate(movies)}
    i2u = {idx: user_id for user_id, idx in u2i.items()}
    i2m = {idx: movie_id for movie_id, idx in m2i.items()}
    cf_df["user_idx"] = cf_df["userId"].map(u2i)
    cf_df["movie_idx"] = cf_df["movieId"].map(m2i)
    return cf_df, u2i, m2i, i2u, i2m


def train_mf_sgd(ratings_df, num_factors=40, learning_rate=0.005, reg=0.01, epochs=30, random_state=RANDOM_STATE):
    cf_df, u2i, m2i, i2u, i2m = prepare_cf_data(ratings_df)
    rng = np.random.default_rng(random_state)
    P_local = rng.normal(0, 0.1, size=(len(u2i), num_factors))
    Q_local = rng.normal(0, 0.1, size=(len(m2i), num_factors))
    bu_local = np.zeros(len(u2i))
    bi_local = np.zeros(len(m2i))
    mu_local = cf_df["rating"].mean()
    training_rmse = []
    for epoch in range(epochs):
        shuffled = cf_df.sample(frac=1, random_state=random_state + epoch)
        total_squared_error = 0.0
        for row in shuffled.itertuples(index=False):
            u, i, r_ui = row.user_idx, row.movie_idx, row.rating
            pred = mu_local + bu_local[u] + bi_local[i] + np.dot(P_local[u], Q_local[i])
            err = r_ui - pred
            old_pu = P_local[u].copy()
            old_qi = Q_local[i].copy()
            bu_local[u] += learning_rate * (err - reg * bu_local[u])
            bi_local[i] += learning_rate * (err - reg * bi_local[i])
            P_local[u] += learning_rate * (err * old_qi - reg * old_pu)
            Q_local[i] += learning_rate * (err * old_pu - reg * old_qi)
            total_squared_error += err ** 2
        training_rmse.append(float(np.sqrt(total_squared_error / len(cf_df))))
    return {
        "model_type": "mf_explicit",
        "P": P_local,
        "Q": Q_local,
        "bu": bu_local,
        "bi": bi_local,
        "mu": mu_local,
        "user_to_idx": u2i,
        "movie_to_idx": m2i,
        "idx_to_user": i2u,
        "idx_to_movie": i2m,
        "num_factors": num_factors,
        "rating_min": 0.5,
        "rating_max": 5.0,
        "training_rmse": training_rmse,
        "hyperparams": {
            "num_factors": num_factors,
            "learning_rate": learning_rate,
            "reg": reg,
            "epochs": epochs,
        },
    }


def predict_rating_from_model(model, user_id, movie_id):
    if user_id not in model["user_to_idx"] or movie_id not in model["movie_to_idx"]:
        return model.get("mu", 0.0)
    u = model["user_to_idx"][user_id]
    i = model["movie_to_idx"][movie_id]
    bu = model.get("bu")
    bi = model.get("bi")
    pred = model.get("mu", 0.0)
    if bu is not None:
        pred += bu[u]
    if bi is not None:
        pred += bi[i]
    pred += np.dot(model["P"][u], model["Q"][i])
    return float(np.clip(pred, model.get("rating_min", -np.inf), model.get("rating_max", np.inf)))


def get_model_recommendations(model, ratings_df, user_id, top_n=10):
    if user_id not in model["user_to_idx"]:
        return pd.DataFrame(columns=["movieId", "score"])
    rated_movies = set(ratings_df[ratings_df["userId"] == user_id]["movieId"].values)
    rows = [
        {"movieId": movie_id, "score": predict_rating_from_model(model, user_id, movie_id)}
        for movie_id in model["movie_to_idx"].keys()
        if movie_id not in rated_movies
    ]
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_n)


mf_param_grid = [
    {"num_factors": 20, "learning_rate": 0.005, "reg": 0.01, "epochs": 20},
    {"num_factors": 40, "learning_rate": 0.005, "reg": 0.01, "epochs": 25},
    {"num_factors": 60, "learning_rate": 0.005, "reg": 0.02, "epochs": 25},
]
val_eval_users = get_eval_users(val_ratings_v2)
trained_mf_models, mf_tuning_rows = [], []
for params in mf_param_grid:
    candidate = train_mf_sgd(train_ratings_v2, **params)
    summary, _ = evaluate_recommender(
        "MF candidate",
        lambda user_id, k, model=candidate: get_model_recommendations(model, train_ratings_v2, user_id, k),
        train_ratings_v2,
        val_ratings_v2,
        val_eval_users,
        k=K,
    )
    trained_mf_models.append(candidate)
    mf_tuning_rows.append({**params, **summary})
mf_tuning_results = pd.DataFrame(mf_tuning_rows).sort_values(
    ["recall_at_k", "ndcg_at_k", "precision_at_k"], ascending=False
).reset_index(drop=True)
display(mf_tuning_results)
best_mf_params = mf_tuning_results.iloc[0][["num_factors", "learning_rate", "reg", "epochs"]].to_dict()
best_mf_params["num_factors"] = int(best_mf_params["num_factors"])
best_mf_params["epochs"] = int(best_mf_params["epochs"])
best_mf_train_model = train_mf_sgd(train_ratings_v2, **best_mf_params)
final_mf_model = train_mf_sgd(development_ratings, **best_mf_params)
