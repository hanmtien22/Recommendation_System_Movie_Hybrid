test_eval_users = get_eval_users(test_ratings_v2)
final_ranking_models = {"MF tuned": final_mf_model, "BPR tuned": final_bpr_model, "ALS implicit": final_als_model}
final_best_ranking_model = final_ranking_models[best_validation_ranking_name]

benchmark_specs = [
    ("Random baseline", lambda user_id, k: get_random_recommendations(development_ratings, user_id, all_movie_ids, k)),
    ("Popular baseline", lambda user_id, k: get_popular_recommendations(popularity_table_dev, development_ratings, user_id, k)),
    ("Bias-only baseline", lambda user_id, k: get_bias_only_recommendations(bias_only_dev, development_ratings, user_id, all_movie_ids, k)),
    ("Item-KNN baseline", lambda user_id, k: get_item_knn_recommendations(item_knn_dev, development_ratings, user_id, k)),
    ("User-KNN baseline", lambda user_id, k: get_user_knn_recommendations(user_knn_dev, development_ratings, user_id, k)),
    ("MF tuned", lambda user_id, k: get_model_recommendations(final_mf_model, development_ratings, user_id, k)),
    ("BPR tuned", lambda user_id, k: get_model_recommendations(final_bpr_model, development_ratings, user_id, k)),
    ("ALS implicit", lambda user_id, k: get_model_recommendations(final_als_model, development_ratings, user_id, k)),
    ("CBF weighted", lambda user_id, k: get_cbf_recommendations_with_matrix(user_id, development_ratings, content_features_v2, k)),
    ("Hybrid tuned", lambda user_id, k: get_hybrid_recommendations_v2(user_id, final_best_ranking_model, development_ratings, popularity_table_dev, k, best_hybrid_weights, best_hybrid_normalization)),
    ("Logistic blender", lambda user_id, k: get_blended_recommendations(user_id, final_best_ranking_model, development_ratings, popularity_table_dev, k)),
]
benchmark_summaries, benchmark_details = [], []
for model_name, recommend_fn in benchmark_specs:
    summary, detail = evaluate_recommender(model_name, recommend_fn, development_ratings, test_ratings_v2, test_eval_users, k=K)
    benchmark_summaries.append(summary)
    benchmark_details.append(detail)
benchmark_results = pd.DataFrame(benchmark_summaries).sort_values(
    ["recall_at_k", "ndcg_at_k", "precision_at_k"], ascending=False
).reset_index(drop=True)
display(benchmark_results)

all_benchmark_details = pd.concat(benchmark_details, ignore_index=True)
all_benchmark_details["activity_bucket"] = pd.qcut(all_benchmark_details["train_count"], q=3, duplicates="drop").astype(str)
segment_results = all_benchmark_details.groupby(["model", "activity_bucket"], observed=True)[
    ["precision_at_k", "recall_at_k", "ndcg_at_k"]
].mean().reset_index()
display(segment_results)

popular_recall = benchmark_results.loc[benchmark_results["model"] == "Popular baseline", "recall_at_k"].iloc[0]
best_model_row = benchmark_results.iloc[0]
print("Best test model:", best_model_row["model"])
print("Vượt Popular baseline:", bool(best_model_row["recall_at_k"] > popular_recall))
