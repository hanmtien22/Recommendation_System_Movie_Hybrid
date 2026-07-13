from itertools import product
from math import log2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
K = 40
LIKED_THRESHOLD = 4.0
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2
EVAL_SAMPLE_SIZE = 100

development_ratings, test_ratings_v2 = train_test_split(
    final_ratings, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
train_ratings_v2, val_ratings_v2 = train_test_split(
    development_ratings, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
)


def temporal_leave_last_out_split(ratings_df):
    ordered = ratings_df.sort_values(["userId", "timestamp"])
    test_idx = ordered.groupby("userId").tail(1).index
    train_df = ordered.drop(index=test_idx).copy()
    test_df = ordered.loc[test_idx].copy()
    return train_df, test_df


temporal_train_ratings, temporal_test_ratings = temporal_leave_last_out_split(final_ratings)

print("Random split mới:")
print(f"   Train: {len(train_ratings_v2):,} ratings")
print(f"   Validation: {len(val_ratings_v2):,} ratings")
print(f"   Test: {len(test_ratings_v2):,} ratings")
print("Temporal split:")
print(f"   Train: {len(temporal_train_ratings):,} ratings")
print(f"   Test: {len(temporal_test_ratings):,} ratings")
