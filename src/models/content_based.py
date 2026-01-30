"""
Content-Based Filtering Recommender.

Represents items using TF-IDF features derived from book metadata
(author, publisher) and recommends items similar to a user's profile.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from .similarity import compute_cosine_similarity, compute_jaccard_similarity


class ContentBasedRecommender:
    """
    Content-based recommender using book metadata features.

    Item Representation:
        - TF-IDF vectors from author names
        - TF-IDF vectors from publisher names
        - Normalized publication year
        Combined via horizontal stacking into a single sparse feature matrix.

    User Profile:
        Weighted centroid of feature vectors of items the user rated highly,
        where weights are the user's ratings.

    Similarity Functions:
        - cosine: Standard cosine similarity on TF-IDF vectors.
          Good for sparse high-dimensional spaces. Scale-invariant.
        - jaccard: Jaccard similarity on binarized features.
          Measures overlap of non-zero features. More interpretable
          (shared author/publisher = direct match).
    """

    def __init__(
        self,
        books_df,
        similarity_metric="cosine",
        n_neighbors=50,
        relevance_threshold=6.0,
    ):
        """
        Args:
            books_df: DataFrame with isbn, title, author, year, publisher
            similarity_metric: "cosine" or "jaccard"
            n_neighbors: Number of neighbors for rating prediction
            relevance_threshold: Minimum rating to consider an item "liked"
        """
        self.books_df = books_df.copy()
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.relevance_threshold = relevance_threshold

        # Populated during fit
        self.item_features = None       # sparse (n_items, n_features)
        self.isbn_to_idx = {}
        self.idx_to_isbn = {}
        self.user_profiles = {}         # user_id -> {isbn: rating}
        self.user_pref_vectors = {}     # user_id -> sparse vector
        self.global_mean = 0.0
        self.user_means = {}
        self._all_isbns = []
        self._author_vectorizer = None
        self._publisher_vectorizer = None

    def fit(self, train_df):
        """
        Build item features and user profiles.

        Features are built for ALL books (not just training items),
        which is the key advantage of content-based filtering:
        it can represent items not seen during training.

        Args:
            train_df: Training DataFrame with user_id, isbn, rating
        """
        self.global_mean = train_df['rating'].mean()
        self.user_means = train_df.groupby('user_id')['rating'].mean().to_dict()

        # Build item features for all books
        self._build_item_features()

        # Store user profiles (items rated in training)
        for user_id, group in train_df.groupby('user_id'):
            self.user_profiles[user_id] = dict(zip(group['isbn'], group['rating']))

        # Precompute user preference vectors
        self._build_user_preference_vectors()

        return self

    def _build_item_features(self):
        """
        Build sparse feature matrix from book metadata.

        Features:
        1. Author TF-IDF: Captures author similarity. TF-IDF is preferred
           over one-hot because it weights rare authors higher (higher signal
           when two books share an uncommon author).
        2. Publisher TF-IDF: Captures publishing house affinity.
        3. Year: Normalized publication year (decade proximity).
        """
        books = self.books_df.copy()

        # Clean text fields
        books['author_clean'] = books['author'].fillna('').astype(str).str.lower().str.strip()
        books['publisher_clean'] = books['publisher'].fillna('').astype(str).str.lower().str.strip()

        # Build ISBN index
        self._all_isbns = books['isbn'].tolist()
        self.isbn_to_idx = {isbn: i for i, isbn in enumerate(self._all_isbns)}
        self.idx_to_isbn = {i: isbn for isbn, i in self.isbn_to_idx.items()}

        # Author TF-IDF
        self._author_vectorizer = TfidfVectorizer(
            analyzer='word',
            max_features=5000,
            min_df=2,
            dtype=np.float32
        )
        author_features = self._author_vectorizer.fit_transform(books['author_clean'])

        # Publisher TF-IDF
        self._publisher_vectorizer = TfidfVectorizer(
            analyzer='word',
            max_features=2000,
            min_df=2,
            dtype=np.float32
        )
        publisher_features = self._publisher_vectorizer.fit_transform(books['publisher_clean'])

        # Year feature (normalized)
        years = books['year'].fillna(books['year'].median()).values.astype(np.float32)
        year_min = np.nanmin(years[years > 1900]) if np.any(years > 1900) else 1900
        year_max = np.nanmax(years[years <= 2004]) if np.any(years <= 2004) else 2004
        years = np.clip(years, year_min, year_max)
        years_norm = (years - year_min) / max(year_max - year_min, 1)
        year_features = csr_matrix(years_norm.reshape(-1, 1))

        # Combine features
        self.item_features = hstack([author_features, publisher_features, year_features]).tocsr()

        # For jaccard, binarize
        if self.similarity_metric == "jaccard":
            self.item_features_binary = self.item_features.copy()
            self.item_features_binary.data = np.ones_like(self.item_features_binary.data)
        else:
            self.item_features_binary = None

    def _build_user_preference_vectors(self):
        """
        Build user preference vector as weighted centroid of liked item features.

        For each user, the preference vector is the weighted average of
        feature vectors of items they rated >= relevance_threshold.
        Weights are the ratings themselves.
        """
        for user_id, item_ratings in self.user_profiles.items():
            # Select liked items
            liked_items = {
                isbn: rating for isbn, rating in item_ratings.items()
                if rating >= self.relevance_threshold and isbn in self.isbn_to_idx
            }

            if not liked_items:
                # Fall back to all rated items with positive weight
                liked_items = {
                    isbn: rating for isbn, rating in item_ratings.items()
                    if isbn in self.isbn_to_idx
                }

            if not liked_items:
                continue

            # Build weighted centroid
            indices = [self.isbn_to_idx[isbn] for isbn in liked_items]
            weights = np.array([liked_items[self.idx_to_isbn[i]] for i in indices], dtype=np.float32)
            weights = weights / weights.sum()

            feature_matrix = self.item_features[indices].toarray().astype(np.float64)
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            pref_vector = csr_matrix(weights.astype(np.float64) @ feature_matrix)
            # Clean any NaN/inf in preference vector
            pref_vector.data = np.nan_to_num(pref_vector.data, nan=0.0, posinf=0.0, neginf=0.0)
            self.user_pref_vectors[user_id] = pref_vector

    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair.

        Strategy:
        1. Find K most similar items to target item among user's rated items
        2. Weighted average of ratings, weighted by content similarity
        3. Fallback to user mean or global mean
        """
        if item_id not in self.isbn_to_idx:
            return self.user_means.get(user_id, self.global_mean)

        if user_id not in self.user_profiles:
            return self.global_mean

        rated_items = self.user_profiles[user_id]
        rated_isbns = [isbn for isbn in rated_items if isbn in self.isbn_to_idx]

        if not rated_isbns:
            return self.user_means.get(user_id, self.global_mean)

        # Compute similarity between target item and user's rated items
        target_idx = self.isbn_to_idx[item_id]
        target_vec = self.item_features[target_idx]

        rated_indices = [self.isbn_to_idx[isbn] for isbn in rated_isbns]
        rated_matrix = self.item_features[rated_indices]

        if self.similarity_metric == "jaccard" and self.item_features_binary is not None:
            target_vec_bin = self.item_features_binary[target_idx]
            rated_matrix_bin = self.item_features_binary[rated_indices]
            # Jaccard for single vector vs matrix
            intersection = (target_vec_bin @ rated_matrix_bin.T).toarray().flatten()
            target_sum = target_vec_bin.sum()
            rated_sums = np.array(rated_matrix_bin.sum(axis=1)).flatten()
            union = target_sum + rated_sums - intersection
            union[union == 0] = 1.0
            sims = intersection / union
        else:
            sims = cosine_similarity(target_vec, rated_matrix).flatten()

        # Select top-K neighbors
        if len(sims) > self.n_neighbors:
            top_k = np.argpartition(sims, -self.n_neighbors)[-self.n_neighbors:]
        else:
            top_k = np.arange(len(sims))

        top_sims = sims[top_k]
        top_isbns = [rated_isbns[i] for i in top_k]
        top_ratings = np.array([rated_items[isbn] for isbn in top_isbns])

        # Filter to positive similarities
        pos_mask = top_sims > 0
        if not pos_mask.any():
            return self.user_means.get(user_id, self.global_mean)

        top_sims = top_sims[pos_mask]
        top_ratings = top_ratings[pos_mask]

        # Weighted average
        pred = np.sum(top_sims * top_ratings) / np.sum(top_sims)
        return float(np.clip(pred, 1, 10))

    def predict_batch(self, user_ids, item_ids):
        """Predict ratings for multiple user-item pairs."""
        return np.array([
            self.predict(u, i) for u, i in zip(user_ids, item_ids)
        ])

    def recommend(self, user_id, n, exclude_items):
        """
        Generate top-N recommendations.

        Computes similarity between user's preference vector and all items,
        excluding items the user has already interacted with.

        Args:
            user_id: User identifier
            n: Number of recommendations
            exclude_items: Set of item IDs to exclude

        Returns:
            List of recommended item IDs
        """
        if user_id not in self.user_pref_vectors:
            # Fall back to most common items not in exclude set
            return []

        pref_vec = self.user_pref_vectors[user_id]

        if self.similarity_metric == "jaccard" and self.item_features_binary is not None:
            pref_binary = pref_vec.copy()
            pref_binary.data = np.ones_like(pref_binary.data)
            intersection = (pref_binary @ self.item_features_binary.T).toarray().flatten()
            pref_sum = pref_binary.sum()
            item_sums = np.array(self.item_features_binary.sum(axis=1)).flatten()
            union = pref_sum + item_sums - intersection
            union[union == 0] = 1.0
            scores = intersection / union
        else:
            scores = cosine_similarity(pref_vec, self.item_features).flatten()

        # Build candidate list: exclude seen items
        candidates = []
        for idx in np.argsort(scores)[::-1]:
            isbn = self.idx_to_isbn[idx]
            if isbn not in exclude_items:
                candidates.append(isbn)
                if len(candidates) >= n:
                    break

        return candidates
