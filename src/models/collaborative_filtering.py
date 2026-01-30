"""
Item-Item Collaborative Filtering Recommender.

Uses the user-item rating matrix to compute item-item similarities
and predict ratings via weighted neighborhood aggregation.

Item-item CF is chosen over user-user because:
1. Item similarity is more stable over time than user similarity.
2. Comparable matrix dimensions (items vs users) in this dataset.
3. Industry standard (Amazon's original approach).

Limitation with temporal split:
Items in the test set (books published 2002-2004) have no training ratings,
so they cannot be represented in the collaborative similarity matrix.
For these items, prediction falls back to user/global mean.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

from .similarity import (
    compute_cosine_similarity,
    compute_pearson_similarity,
    get_top_k_neighbors,
)


class ItemItemCFRecommender:
    """
    Item-Item Collaborative Filtering.

    Similarity Functions:
        - cosine: Cosine similarity on raw rating vectors.
          Standard baseline for CF. Treats missing ratings as zero.
        - pearson: Pearson correlation (mean-centered cosine).
          Accounts for item rating bias: items with consistently
          higher/lower ratings get centered. Preferred for explicit
          ratings where users have different rating scales.
    """

    def __init__(
        self,
        similarity_metric="cosine",
        n_neighbors=50,
        min_common_users=2,
    ):
        """
        Args:
            similarity_metric: "cosine" or "pearson"
            n_neighbors: Number of item neighbors to consider
            min_common_users: Minimum co-rated users for similarity
                              (not currently enforced for speed; relies on
                              sparsity naturally filtering weak similarities)
        """
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.min_common_users = min_common_users

        # Populated during fit
        self.item_neighbors = {}    # item_idx -> [(neighbor_idx, sim), ...]
        self.isbn_to_idx = {}
        self.idx_to_isbn = {}
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.rating_matrix = None   # sparse (n_items, n_users)
        self.user_profiles = {}     # user_id -> {isbn: rating}
        self.global_mean = 0.0
        self.item_means = {}
        self.user_means = {}
        self._all_train_isbns = set()

    def fit(self, train_df):
        """
        Build rating matrix and compute item-item similarities.

        Steps:
        1. Create user/item index mappings
        2. Build sparse item×user rating matrix
        3. Compute pairwise item similarity
        4. Store top-K neighbors per item
        5. Cache user profiles and means for prediction

        Args:
            train_df: Training DataFrame with user_id, isbn, rating
        """
        print(f"Fitting Item-Item CF ({self.similarity_metric})...")

        self.global_mean = train_df['rating'].mean()
        self.user_means = train_df.groupby('user_id')['rating'].mean().to_dict()
        self.item_means = train_df.groupby('isbn')['rating'].mean().to_dict()

        # Build index mappings
        unique_items = train_df['isbn'].unique()
        unique_users = train_df['user_id'].unique()
        self.isbn_to_idx = {isbn: i for i, isbn in enumerate(unique_items)}
        self.idx_to_isbn = {i: isbn for isbn, i in self.isbn_to_idx.items()}
        self.user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx_to_user = {i: uid for uid, i in self.user_to_idx.items()}
        self._all_train_isbns = set(unique_items)

        n_items = len(unique_items)
        n_users = len(unique_users)

        # Build sparse rating matrix (items x users)
        rows = [self.isbn_to_idx[isbn] for isbn in train_df['isbn']]
        cols = [self.user_to_idx[uid] for uid in train_df['user_id']]
        vals = train_df['rating'].values.astype(np.float32)
        self.rating_matrix = csr_matrix((vals, (rows, cols)), shape=(n_items, n_users))

        print(f"  Rating matrix: {n_items} items x {n_users} users")
        print(f"  Computing {self.similarity_metric} similarity...")

        # Compute similarity
        if self.similarity_metric == "pearson":
            sim_matrix = compute_pearson_similarity(self.rating_matrix)
        else:
            sim_matrix = compute_cosine_similarity(self.rating_matrix)

        print(f"  Extracting top-{self.n_neighbors} neighbors...")
        self.item_neighbors = get_top_k_neighbors(sim_matrix, self.n_neighbors)

        # Free full similarity matrix
        del sim_matrix

        # Store user profiles
        for user_id, group in train_df.groupby('user_id'):
            self.user_profiles[user_id] = dict(zip(group['isbn'], group['rating']))

        print(f"  Done. {n_items} items, {n_users} users.")
        return self

    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair.

        Formula (baseline-adjusted):
            pred(u, i) = μ_i + Σ_{j ∈ N(i,u)} sim(i,j) * (r(u,j) - μ_j)
                         / Σ_{j ∈ N(i,u)} |sim(i,j)|

        Where N(i,u) is the set of i's neighbors that user u has rated.

        Falls back to user mean or global mean if item is unknown or
        no neighbors found.
        """
        # Unknown item (cold-start) -> fallback
        if item_id not in self.isbn_to_idx:
            return self.user_means.get(user_id, self.global_mean)

        # Unknown user -> fallback
        if user_id not in self.user_profiles:
            return self.item_means.get(item_id, self.global_mean)

        item_idx = self.isbn_to_idx[item_id]
        user_rated = self.user_profiles[user_id]
        item_mean = self.item_means.get(item_id, self.global_mean)

        # Get neighbors of this item that the user has rated
        neighbors = self.item_neighbors.get(item_idx, [])
        if not neighbors:
            return self.user_means.get(user_id, self.global_mean)

        weighted_sum = 0.0
        sim_sum = 0.0

        for neighbor_idx, sim in neighbors:
            neighbor_isbn = self.idx_to_isbn[neighbor_idx]
            if neighbor_isbn in user_rated:
                neighbor_mean = self.item_means.get(neighbor_isbn, self.global_mean)
                rating = user_rated[neighbor_isbn]
                weighted_sum += sim * (rating - neighbor_mean)
                sim_sum += abs(sim)

        if sim_sum == 0:
            return self.user_means.get(user_id, self.global_mean)

        pred = item_mean + weighted_sum / sim_sum
        return float(np.clip(pred, 1, 10))

    def predict_batch(self, user_ids, item_ids):
        """Predict ratings for multiple user-item pairs."""
        return np.array([
            self.predict(u, i) for u, i in zip(user_ids, item_ids)
        ])

    def recommend(self, user_id, n, exclude_items):
        """
        Generate top-N recommendations.

        Strategy:
        1. Gather candidate items from neighbors of user's rated items
        2. Score each candidate using the CF prediction formula
        3. Exclude already-seen items, return top-N by predicted score

        Args:
            user_id: User identifier
            n: Number of recommendations
            exclude_items: Set of item IDs to exclude

        Returns:
            List of recommended item IDs
        """
        if user_id not in self.user_profiles:
            return []

        user_rated = self.user_profiles[user_id]

        # Gather candidate items: neighbors of items the user has rated
        candidates = set()
        for isbn in user_rated:
            if isbn not in self.isbn_to_idx:
                continue
            item_idx = self.isbn_to_idx[isbn]
            for neighbor_idx, _ in self.item_neighbors.get(item_idx, []):
                neighbor_isbn = self.idx_to_isbn[neighbor_idx]
                if neighbor_isbn not in exclude_items:
                    candidates.add(neighbor_isbn)

        if not candidates:
            return []

        # Score all candidates
        scored = []
        for isbn in candidates:
            score = self.predict(user_id, isbn)
            scored.append((isbn, score))

        # Sort by predicted score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return [isbn for isbn, _ in scored[:n]]
