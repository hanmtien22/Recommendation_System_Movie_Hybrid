import numpy as np


def evaluate_rmse(recommender, test_ratings):
    squared_errors = []
    skipped = 0
    for row in test_ratings.itertuples(index=False):
        user_id = row.userId
        movie_id = row.movieId
        actual = row.rating
        if user_id not in recommender.user_to_idx or movie_id not in recommender.movie_to_idx:
            skipped += 1
            continue
        pred = recommender.predict_rating(user_id, movie_id)
        squared_errors.append((actual - pred) ** 2)

    rmse = float(np.sqrt(np.mean(squared_errors))) if squared_errors else None
    coverage = len(squared_errors) / len(test_ratings) if len(test_ratings) else 0
    return {"rmse": rmse, "coverage": coverage, "skipped": skipped}
