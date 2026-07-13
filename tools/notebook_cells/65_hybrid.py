def normalize_scores(series, method="minmax"):
    values = pd.Series(series).astype(float)
    if method == "minmax":
        min_val, max_val = values.min(), values.max()
        return pd.Series(0.5, index=values.index) if max_val == min_val else (values - min_val) / (max_val - min_val)
    if method == "zscore":
        std = values.std(ddof=0)
        return pd.Series(0.0, index=values.index) if std == 0 else (values - values.mean()) / std
    if method == "rank":
        return values.rank(method="average", pct=True)
    raise ValueError(f"Unknown normalization method: {method}")


def build_candidate_frame(user_id, ranking_model, ratings_df, popularity_table, content_matrix=content_features_v2):
    rated_movies = set(ratings_df[ratings_df["userId"] == user_id]["movieId"].values)
    candidate_meta = movie_meta[~movie_meta["movieId"].isin(rated_movies)][["movieId", "id", "title"]].copy()
    candidate_meta["cf_score"] = [predict_rating_from_model(ranking_model, user_id, mid) for mid in candidate_meta["movieId"]]
    profile = build_weighted_profile(user_id, ratings_df, content_matrix)
    if profile is not None:
        all_cbf = cosine_similarity(profile.reshape(1, -1), content_matrix).ravel()
        candidate_meta["cbf_score"] = [all_cbf[movieid_to_content_idx[mid]] for mid in candidate_meta["movieId"]]
    else:
        candidate_meta["cbf_score"] = 0.0
    popularity_lookup = popularity_table.set_index("movieId")["bayesian_popularity"].to_dict()
    candidate_meta["popularity_score"] = candidate_meta["movieId"].map(popularity_lookup).fillna(0.0)
    return candidate_meta


def score_hybrid_candidates(candidates, weights=(0.5, 0.3, 0.2), normalization="rank", top_n=10):
    scored = candidates.copy()
    scored["norm_cf"] = normalize_scores(scored["cf_score"], normalization)
    scored["norm_cbf"] = normalize_scores(scored["cbf_score"], normalization)
    scored["norm_popularity"] = normalize_scores(scored["popularity_score"], normalization)
    w_cf, w_cbf, w_pop = weights
    scored["hybrid_score"] = (
        w_cf * scored["norm_cf"] + w_cbf * scored["norm_cbf"] + w_pop * scored["norm_popularity"]
    )
    return scored.sort_values("hybrid_score", ascending=False).head(top_n)


def get_hybrid_recommendations_v2(user_id, ranking_model, ratings_df, popularity_table, top_n=10, weights=(0.5, 0.3, 0.2), normalization="rank"):
    return score_hybrid_candidates(
        build_candidate_frame(user_id, ranking_model, ratings_df, popularity_table),
        weights,
        normalization,
        top_n,
    )


ranking_model_candidates = {
    "MF tuned": best_mf_train_model,
    "BPR tuned": best_bpr_train_model,
    "ALS implicit": trained_als_models[0],
}
best_validation_ranking_name = max(
    [
        ("MF tuned", mf_tuning_results.iloc[0]["recall_at_k"]),
        ("BPR tuned", bpr_tuning_results.iloc[0]["recall_at_k"]),
        ("ALS implicit", als_tuning_results.iloc[0]["recall_at_k"]),
    ],
    key=lambda x: x[1],
)[0]
best_validation_ranking_model = ranking_model_candidates[best_validation_ranking_name]

weight_grid = [(0.6, 0.4, 0.0), (0.6, 0.0, 0.4), (0.5, 0.3, 0.2), (0.4, 0.2, 0.4), (0.3, 0.3, 0.4)]
normalization_grid = ["minmax", "rank"]
validation_candidate_cache = {
    u: build_candidate_frame(u, best_validation_ranking_model, train_ratings_v2, popularity_table_train)
    for u in val_eval_users
}
hybrid_rows = []
for normalization in normalization_grid:
    for weights in weight_grid:
        summary, _ = evaluate_recommender(
            "Hybrid candidate",
            lambda user_id, k, w=weights, n=normalization: score_hybrid_candidates(validation_candidate_cache[user_id], w, n, k),
            train_ratings_v2,
            val_ratings_v2,
            val_eval_users,
            k=K,
        )
        hybrid_rows.append({"weights": weights, "normalization": normalization, **summary})
hybrid_tuning_results = pd.DataFrame(hybrid_rows).sort_values(
    ["recall_at_k", "ndcg_at_k", "precision_at_k"], ascending=False
).reset_index(drop=True)
display(hybrid_tuning_results)
best_hybrid_weights = tuple(hybrid_tuning_results.iloc[0]["weights"])
best_hybrid_normalization = hybrid_tuning_results.iloc[0]["normalization"]


def build_blender_training_frame(candidate_cache, test_df, eval_users):
    frames = []
    for user_id in eval_users:
        frame = candidate_cache[user_id].copy()
        relevant = set(test_df[(test_df["userId"] == user_id) & (test_df["rating"] >= LIKED_THRESHOLD)]["movieId"].values)
        frame["label"] = frame["movieId"].isin(relevant).astype(int)
        frame["userId"] = user_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


blender_train = build_blender_training_frame(validation_candidate_cache, val_ratings_v2, val_eval_users)
blender_features = ["cf_score", "cbf_score", "popularity_score"]
logistic_blender = LogisticRegression(class_weight="balanced", max_iter=500, random_state=RANDOM_STATE)
logistic_blender.fit(blender_train[blender_features], blender_train["label"])


def get_blended_recommendations(user_id, ranking_model, ratings_df, popularity_table, top_n=10):
    candidates = build_candidate_frame(user_id, ranking_model, ratings_df, popularity_table)
    candidates["blend_score"] = logistic_blender.predict_proba(candidates[blender_features])[:, 1]
    return candidates.sort_values("blend_score", ascending=False).head(top_n)
