def prepare_implicit_data(ratings_df, threshold=LIKED_THRESHOLD):
    positives = ratings_df[ratings_df["rating"] >= threshold][["userId", "movieId"]].drop_duplicates()
    users = positives["userId"].unique()
    movies = ratings_df["movieId"].unique()
    u2i = {u: idx for idx, u in enumerate(users)}
    m2i = {m: idx for idx, m in enumerate(movies)}
    i2m = {idx: m for m, idx in m2i.items()}
    user_positives = {
        u2i[u]: set(group["movieId"].map(m2i))
        for u, group in positives.groupby("userId")
        if u in u2i
    }
    return positives, u2i, m2i, i2m, user_positives


def train_bpr_mf(ratings_df, num_factors=40, learning_rate=0.03, reg=0.002, epochs=20, random_state=RANDOM_STATE):
    positives, u2i, m2i, i2m, user_positives = prepare_implicit_data(ratings_df)
    rng = np.random.default_rng(random_state)
    P_local = rng.normal(0, 0.1, size=(len(u2i), num_factors))
    Q_local = rng.normal(0, 0.1, size=(len(m2i), num_factors))
    all_items = np.arange(len(m2i))
    positive_pairs = [(u2i[row.userId], m2i[row.movieId]) for row in positives.itertuples(index=False)]
    for _ in range(epochs):
        for _ in range(len(positive_pairs)):
            u, i = positive_pairs[rng.integers(len(positive_pairs))]
            j = int(rng.choice(all_items))
            while j in user_positives[u]:
                j = int(rng.choice(all_items))
            x_uij = np.dot(P_local[u], Q_local[i] - Q_local[j])
            sigmoid = 1.0 / (1.0 + np.exp(x_uij))
            pu_old = P_local[u].copy()
            qi_old = Q_local[i].copy()
            qj_old = Q_local[j].copy()
            P_local[u] += learning_rate * (sigmoid * (qi_old - qj_old) - reg * pu_old)
            Q_local[i] += learning_rate * (sigmoid * pu_old - reg * qi_old)
            Q_local[j] += learning_rate * (-sigmoid * pu_old - reg * qj_old)
    return {
        "model_type": "bpr_mf",
        "P": P_local,
        "Q": Q_local,
        "user_to_idx": u2i,
        "movie_to_idx": m2i,
        "idx_to_movie": i2m,
        "mu": 0.0,
        "rating_min": -np.inf,
        "rating_max": np.inf,
        "hyperparams": {
            "num_factors": num_factors,
            "learning_rate": learning_rate,
            "reg": reg,
            "epochs": epochs,
        },
    }


def train_implicit_als(ratings_df, num_factors=20, alpha=10.0, reg=0.1, epochs=5, random_state=RANDOM_STATE):
    _, u2i, m2i, i2m, _ = prepare_implicit_data(ratings_df)
    rng = np.random.default_rng(random_state)
    R = np.zeros((len(u2i), len(m2i)), dtype=np.float32)
    for row in ratings_df.itertuples(index=False):
        if row.userId in u2i and row.movieId in m2i and row.rating >= LIKED_THRESHOLD:
            R[u2i[row.userId], m2i[row.movieId]] = 1.0
    X = rng.normal(0, 0.1, size=(len(u2i), num_factors))
    Y = rng.normal(0, 0.1, size=(len(m2i), num_factors))
    eye = np.eye(num_factors)
    user_positive_items = {u: np.flatnonzero(R[u]) for u in range(R.shape[0])}
    item_positive_users = {i: np.flatnonzero(R[:, i]) for i in range(R.shape[1])}
    for _ in range(epochs):
        YTY = Y.T @ Y
        for u in range(len(u2i)):
            items = user_positive_items[u]
            if len(items) == 0:
                continue
            Y_u = Y[items]
            A = YTY + alpha * (Y_u.T @ Y_u) + reg * eye
            b = (1.0 + alpha) * Y_u.sum(axis=0)
            X[u] = np.linalg.solve(A, b)
        XTX = X.T @ X
        for i in range(len(m2i)):
            users = item_positive_users[i]
            if len(users) == 0:
                continue
            X_i = X[users]
            A = XTX + alpha * (X_i.T @ X_i) + reg * eye
            b = (1.0 + alpha) * X_i.sum(axis=0)
            Y[i] = np.linalg.solve(A, b)
    return {
        "model_type": "als_implicit",
        "P": X,
        "Q": Y,
        "user_to_idx": u2i,
        "movie_to_idx": m2i,
        "idx_to_movie": i2m,
        "mu": 0.0,
        "rating_min": -np.inf,
        "rating_max": np.inf,
        "hyperparams": {"num_factors": num_factors, "alpha": alpha, "reg": reg, "epochs": epochs},
    }


bpr_param_grid = [
    {"num_factors": 20, "learning_rate": 0.03, "reg": 0.002, "epochs": 15},
    {"num_factors": 40, "learning_rate": 0.03, "reg": 0.002, "epochs": 20},
]
bpr_rows, trained_bpr_models = [], []
for params in bpr_param_grid:
    candidate = train_bpr_mf(train_ratings_v2, **params)
    summary, _ = evaluate_recommender(
        "BPR candidate",
        lambda user_id, k, model=candidate: get_model_recommendations(model, train_ratings_v2, user_id, k),
        train_ratings_v2,
        val_ratings_v2,
        val_eval_users,
        k=K,
    )
    bpr_rows.append({**params, **summary})
    trained_bpr_models.append(candidate)
bpr_tuning_results = pd.DataFrame(bpr_rows).sort_values(
    ["recall_at_k", "ndcg_at_k", "precision_at_k"], ascending=False
).reset_index(drop=True)
display(bpr_tuning_results)
best_bpr_params = bpr_tuning_results.iloc[0][["num_factors", "learning_rate", "reg", "epochs"]].to_dict()
best_bpr_params["num_factors"] = int(best_bpr_params["num_factors"])
best_bpr_params["epochs"] = int(best_bpr_params["epochs"])
best_bpr_train_model = train_bpr_mf(train_ratings_v2, **best_bpr_params)
final_bpr_model = train_bpr_mf(development_ratings, **best_bpr_params)

als_param_grid = [{"num_factors": 20, "alpha": 10.0, "reg": 0.1, "epochs": 5}]
als_rows, trained_als_models = [], []
for params in als_param_grid:
    candidate = train_implicit_als(train_ratings_v2, **params)
    summary, _ = evaluate_recommender(
        "ALS candidate",
        lambda user_id, k, model=candidate: get_model_recommendations(model, train_ratings_v2, user_id, k),
        train_ratings_v2,
        val_ratings_v2,
        val_eval_users,
        k=K,
    )
    als_rows.append({**params, **summary})
    trained_als_models.append(candidate)
als_tuning_results = pd.DataFrame(als_rows).sort_values(
    ["recall_at_k", "ndcg_at_k", "precision_at_k"], ascending=False
).reset_index(drop=True)
display(als_tuning_results)
final_als_model = train_implicit_als(development_ratings, **als_param_grid[0])
