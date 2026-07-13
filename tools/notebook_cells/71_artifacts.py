selected_model_name = benchmark_results.iloc[0]["model"]
if selected_model_name in {"Hybrid tuned", "Logistic blender"}:
    selected_ranking_backbone = best_validation_ranking_name
else:
    selected_ranking_backbone = selected_model_name

if selected_ranking_backbone == "BPR tuned":
    production_ranking_model = train_bpr_mf(final_ratings, **best_bpr_params)
elif selected_ranking_backbone == "ALS implicit":
    production_ranking_model = train_implicit_als(final_ratings, **als_param_grid[0])
else:
    production_ranking_model = train_mf_sgd(final_ratings, **best_mf_params)

production_popularity = build_popularity_table(final_ratings)

with open(MODELS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open(MODELS_DIR / "content_features.pkl", "wb") as f:
    pickle.dump(content_features_v2, f)
with open(MODELS_DIR / "svd_model.pkl", "wb") as f:
    pickle.dump(production_ranking_model, f)
with open(MODELS_DIR / "popularity_table.pkl", "wb") as f:
    pickle.dump(production_popularity, f)
with open(MODELS_DIR / "blend_model.pkl", "wb") as f:
    pickle.dump(logistic_blender, f)

mappings_v2 = {
    "movieid_to_content_idx": movieid_to_content_idx,
    "user_to_idx": production_ranking_model["user_to_idx"],
    "movie_to_idx": production_ranking_model["movie_to_idx"],
    "idx_to_movie": production_ranking_model["idx_to_movie"],
}
with open(MODELS_DIR / "mappings.pkl", "wb") as f:
    pickle.dump(mappings_v2, f)

hybrid_config = {
    "selected_model_name": selected_model_name,
    "selected_ranking_backbone": selected_ranking_backbone,
    "serving_mode": "blender" if selected_model_name == "Logistic blender" else "hybrid",
    "best_validation_ranking_name": best_validation_ranking_name,
    "weights": best_hybrid_weights,
    "normalization": best_hybrid_normalization,
    "liked_threshold": LIKED_THRESHOLD,
    "k": K,
    "content_n_components": best_content_dim,
    "blender_features": blender_features,
}
with open(MODELS_DIR / "hybrid_config.pkl", "wb") as f:
    pickle.dump(hybrid_config, f)

print("Đã lưu artifacts production:")
for file_name in [
    "tfidf_vectorizer.pkl",
    "content_features.pkl",
    "svd_model.pkl",
    "mappings.pkl",
    "popularity_table.pkl",
    "blend_model.pkl",
    "hybrid_config.pkl",
]:
    print("   -", MODELS_DIR / file_name)
