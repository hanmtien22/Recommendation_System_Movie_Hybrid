temporal_eval_users = get_eval_users(temporal_test_ratings)
temporal_popularity = build_popularity_table(temporal_train_ratings)
temporal_bias = fit_bias_only_model(temporal_train_ratings)
temporal_item_knn = fit_item_knn(temporal_train_ratings)
temporal_user_knn = fit_user_knn(temporal_train_ratings)
if best_validation_ranking_name == "BPR tuned":
    temporal_ranking_model = train_bpr_mf(temporal_train_ratings, **best_bpr_params)
elif best_validation_ranking_name == "ALS implicit":
    temporal_ranking_model = train_implicit_als(temporal_train_ratings, **als_param_grid[0])
else:
    temporal_ranking_model = train_mf_sgd(temporal_train_ratings, **best_mf_params)

temporal_specs = [
    ("Popular temporal", lambda user_id, k: get_popular_recommendations(temporal_popularity, temporal_train_ratings, user_id, k)),
    ("Bias-only temporal", lambda user_id, k: get_bias_only_recommendations(temporal_bias, temporal_train_ratings, user_id, all_movie_ids, k)),
    ("Item-KNN temporal", lambda user_id, k: get_item_knn_recommendations(temporal_item_knn, temporal_train_ratings, user_id, k)),
    ("User-KNN temporal", lambda user_id, k: get_user_knn_recommendations(temporal_user_knn, temporal_train_ratings, user_id, k)),
    ("Best ranking temporal", lambda user_id, k: get_model_recommendations(temporal_ranking_model, temporal_train_ratings, user_id, k)),
    ("Hybrid temporal", lambda user_id, k: get_hybrid_recommendations_v2(user_id, temporal_ranking_model, temporal_train_ratings, temporal_popularity, k, best_hybrid_weights, best_hybrid_normalization)),
]
temporal_summaries = []
for model_name, recommend_fn in temporal_specs:
    summary, _ = evaluate_recommender(model_name, recommend_fn, temporal_train_ratings, temporal_test_ratings, temporal_eval_users, k=K)
    temporal_summaries.append(summary)
display(pd.DataFrame(temporal_summaries).sort_values(["recall_at_k", "ndcg_at_k"], ascending=False))
