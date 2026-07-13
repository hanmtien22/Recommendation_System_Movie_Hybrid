def precision_at_k(recommended, relevant):
    recommended = list(recommended)
    relevant = set(relevant)
    return len(set(recommended) & relevant) / len(recommended) if recommended else 0.0


def recall_at_k(recommended, relevant):
    relevant = set(relevant)
    return len(set(recommended) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended, relevant):
    relevant = set(relevant)
    dcg = sum((1 / log2(idx + 2)) for idx, item in enumerate(recommended) if item in relevant)
    ideal_hits = min(len(relevant), len(recommended))
    idcg = sum((1 / log2(idx + 2)) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def get_eval_users(test_df, sample_size=EVAL_SAMPLE_SIZE, rating_threshold=LIKED_THRESHOLD, random_state=RANDOM_STATE):
    eligible_users = test_df[test_df["rating"] >= rating_threshold]["userId"].drop_duplicates()
    if sample_size is None or sample_size >= len(eligible_users):
        return eligible_users.tolist()
    return eligible_users.sample(n=sample_size, random_state=random_state).tolist()


def validate_recommendations(recommended, rated_movies, k=K):
    recommended = list(recommended)
    assert len(recommended) <= k
    assert len(recommended) == len(set(recommended))
    assert not (set(recommended) & set(rated_movies))


def evaluate_recommender(model_name, recommend_fn, train_df, test_df, eval_users, k=K, rating_threshold=LIKED_THRESHOLD):
    rows = []
    for user_id in eval_users:
        actual_liked = set(
            test_df[(test_df["userId"] == user_id) & (test_df["rating"] >= rating_threshold)]["movieId"].values
        )
        if not actual_liked:
            continue
        recommended = recommend_fn(user_id, k)
        if isinstance(recommended, pd.DataFrame):
            recommended = recommended["movieId"].tolist()
        recommended = list(recommended)[:k]
        if not recommended:
            continue
        rated_movies = train_df[train_df["userId"] == user_id]["movieId"].tolist()
        validate_recommendations(recommended, rated_movies, k=k)
        rows.append(
            {
                "model": model_name,
                "userId": user_id,
                "train_count": len(rated_movies),
                "precision_at_k": precision_at_k(recommended, actual_liked),
                "recall_at_k": recall_at_k(recommended, actual_liked),
                "ndcg_at_k": ndcg_at_k(recommended, actual_liked),
            }
        )
    detail_df = pd.DataFrame(rows)
    summary = {
        "model": model_name,
        "users_evaluated": len(detail_df),
        "precision_at_k": detail_df["precision_at_k"].mean() if not detail_df.empty else 0.0,
        "recall_at_k": detail_df["recall_at_k"].mean() if not detail_df.empty else 0.0,
        "ndcg_at_k": detail_df["ndcg_at_k"].mean() if not detail_df.empty else 0.0,
    }
    return summary, detail_df
