def build_content_features_for_dim(n_components):
    model = TruncatedSVD(n_components=min(n_components, tfidf_matrix.shape[1] - 1), random_state=RANDOM_STATE)
    return model.fit_transform(tfidf_matrix), float(model.explained_variance_ratio_.sum())


def build_weighted_profile(user_id, ratings_df, content_matrix, positive_threshold=3.5, negative_threshold=2.5, negative_weight=0.25):
    user_rows = ratings_df[ratings_df["userId"] == user_id]
    positive_rows = user_rows[user_rows["rating"] >= positive_threshold]
    negative_rows = user_rows[user_rows["rating"] <= negative_threshold]
    positive_valid = positive_rows[positive_rows["movieId"].isin(movieid_to_content_idx)]
    negative_valid = negative_rows[negative_rows["movieId"].isin(movieid_to_content_idx)]
    if positive_valid.empty:
        return None
    positive_indices = [movieid_to_content_idx[mid] for mid in positive_valid["movieId"].values]
    positive_profile = np.average(content_matrix[positive_indices], axis=0, weights=positive_valid["rating"].values)
    if negative_valid.empty:
        return positive_profile
    negative_indices = [movieid_to_content_idx[mid] for mid in negative_valid["movieId"].values]
    negative_profile = np.average(content_matrix[negative_indices], axis=0, weights=(5.5 - negative_valid["rating"]).values)
    return positive_profile - negative_weight * negative_profile


def get_cbf_recommendations_with_matrix(user_id, ratings_df, content_matrix, top_n=10):
    profile = build_weighted_profile(user_id, ratings_df, content_matrix)
    rated_movies = set(ratings_df[ratings_df["userId"] == user_id]["movieId"].values)
    if profile is None:
        return []
    scores = cosine_similarity(profile.reshape(1, -1), content_matrix).ravel()
    rows = [(mid, scores[movieid_to_content_idx[mid]]) for mid in movie_meta["movieId"].tolist() if mid not in rated_movies]
    return [movie_id for movie_id, _ in sorted(rows, key=lambda x: x[1], reverse=True)[:top_n]]


content_dim_grid = [50, 100, 200, 300, 500]
content_dim_rows, content_feature_candidates = [], {}
for dim in content_dim_grid:
    features_dim, explained = build_content_features_for_dim(dim)
    summary, _ = evaluate_recommender(
        f"CBF dim={dim}",
        lambda user_id, k, matrix=features_dim: get_cbf_recommendations_with_matrix(user_id, train_ratings_v2, matrix, k),
        train_ratings_v2,
        val_ratings_v2,
        val_eval_users,
        k=K,
    )
    content_dim_rows.append({"n_components": dim, "explained_variance": explained, **summary})
    content_feature_candidates[dim] = features_dim
content_dim_results = pd.DataFrame(content_dim_rows).sort_values(
    ["recall_at_k", "ndcg_at_k", "precision_at_k"], ascending=False
).reset_index(drop=True)
display(content_dim_results)
best_content_dim = int(content_dim_results.iloc[0]["n_components"])
content_features_v2 = content_feature_candidates[best_content_dim]
print("Best content dimension:", best_content_dim)
