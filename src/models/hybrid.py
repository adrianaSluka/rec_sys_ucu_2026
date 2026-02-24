"""
Hybrid Recommender Systems.

This module implements two complementary hybrid strategies:

1. WeightedHybrid — score-level blending of FunkSVD (collaborative) and
   ContentBasedRecommender (content-based) signals.

2. CascadeHybrid — candidate generation + reranking:
   - Stage 1 (generator): ContentBasedRecommender produces a large candidate set.
   - Stage 2 (ranker): FunkSVD re-scores and re-ranks the candidates.

Design rationale
----------------
The Book Crossing dataset is extremely sparse (99.997 % missing) and dominated
by cold users (81 % have ≤5 ratings).  This creates two complementary failure
modes:

* Collaborative filtering (FunkSVD / Item-CF) captures *taste* but collapses
  for cold items — books published after the training cut-off (2002-2004) have
  no co-rating history, so latent factors are meaningless.

* Content-based filtering captures *item attributes* (author, publisher, year)
  and works for new items, but it ignores social signals and suffers from
  over-specialization (users can only discover authors they already know).

Combining both signals lets each model compensate for the other's blind spots.

Who benefits
------------
* **WeightedHybrid** — benefits *all* users, but especially moderately active
  ones (5-20 ratings) where both signals are partially reliable.  The α weight
  controls the trade-off: higher α → more CF influence.

* **CascadeHybrid** — benefits users with *cold items* in their test set.
  Content-based recall is high for attribute-similar items (same author / series)
  even for books with no rating history; FunkSVD then surface-polishes by user
  taste.  This strategy is most effective when the item pool contains many items
  unseen during collaborative training.

Performance notes
-----------------
Both models precompute all CB and CF scores in fit() using batched matrix
operations, so recommend() is purely array indexing + a partial sort — no
similarity computation at query time.  The CB matrix is restricted to train
items (~9k) instead of the full catalogue (~271k), which is the key size
reduction that keeps precomputation fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Tuple

from .funksvd import FunkSVD, FunkSVDConfig
from .content_based import ContentBasedRecommender


# =============================================================================
# Weighted (Score-Level) Hybrid
# =============================================================================

@dataclass
class WeightedHybridConfig:
    """Configuration for the weighted hybrid recommender."""
    # Mixing weight: final_score = alpha * cf_score + (1 - alpha) * cb_score
    alpha: float = 0.6          # Weight on collaborative signal [0, 1]

    # FunkSVD sub-model config
    funksvd: FunkSVDConfig = field(default_factory=lambda: FunkSVDConfig(
        n_factors=50,
        n_epochs=20,
        lr=0.01,
        reg=0.02,
        early_stopping=True,
        patience=3,
    ))

    # Content-based sub-model config
    cb_similarity: str = "cosine"   # "cosine" or "jaccard"
    cb_n_neighbors: int = 50
    cb_relevance_threshold: float = 6.0

    # Candidate pool size for recommendation generation
    n_candidates: int = 200


class WeightedHybrid:
    """
    Weighted score blending of collaborative (FunkSVD) and content-based signals.

    For rating prediction:
        pred(u, i) = α * CF_pred(u, i) + (1-α) * CB_pred(u, i)

    For recommendation:
        Both models score the shared train-item pool using precomputed score
        arrays (computed once in fit()).  Scores are min-max normalised per
        model and blended before final ranking.  recommend() is O(n_items)
        with no similarity computation at query time.
    """

    def __init__(self, books_df: pd.DataFrame, config: WeightedHybridConfig):
        self.config = config
        self.cf_model = FunkSVD(config.funksvd)
        self.cb_model = ContentBasedRecommender(
            books_df=books_df,
            similarity_metric=config.cb_similarity,
            n_neighbors=config.cb_n_neighbors,
            relevance_threshold=config.cb_relevance_threshold,
        )
        self._item_popularity: List[str] = []
        self.global_mean: float = 0.0

        # Set after fit() — both aligned to the same item order
        self._score_isbns: List[str] = []          # shared item list (train only)
        self._isbn_to_pos: Dict[str, int] = {}     # isbn -> position in arrays
        self._cb_scores: np.ndarray = np.array([]) # (n_users, n_items) rows per user
        self._cf_scores: np.ndarray = np.array([]) # (n_users, n_items)
        self._user_to_row: Dict = {}               # user_id -> row index

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> "WeightedHybrid":
        print("Fitting WeightedHybrid...")
        self.global_mean = train_df["rating"].mean()
        train_isbns = list(train_df["isbn"].unique())

        item_counts = train_df.groupby("isbn")["rating"].count().sort_values(ascending=False)
        self._item_popularity = list(item_counts.index)

        print("  [1/2] Fitting FunkSVD (collaborative)...")
        self.cf_model.fit(train_df, val_df=val_df, verbose=verbose)

        print("  [2/2] Fitting ContentBased (content)...")
        self.cb_model.fit(train_df)

        print("  [3/3] Precomputing score tables (train items only)...")
        (self._score_isbns, self._isbn_to_pos,
         self._cb_scores, self._cf_scores,
         self._user_to_row) = _precompute_score_tables(
            self.cb_model, self.cf_model, train_isbns, self.global_mean
        )
        print("WeightedHybrid ready.")
        return self

    # ------------------------------------------------------------------
    # Rating prediction
    # ------------------------------------------------------------------

    def predict(self, user_id, item_id) -> float:
        cf_pred = self.cf_model.predict_one(user_id, item_id)
        cb_pred = self.cb_model.predict(user_id, item_id)
        return float(self.config.alpha * cf_pred + (1 - self.config.alpha) * cb_pred)

    def predict_batch(self, user_ids, item_ids) -> np.ndarray:
        return np.array([self.predict(u, i) for u, i in zip(user_ids, item_ids)])

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(self, user_id, n: int, exclude_items: Set) -> List[str]:
        """
        Top-N via blended scores.  All scores precomputed — no similarity
        computation at query time, just array ops.
        """
        row = self._user_to_row.get(user_id)
        if row is None:
            return [i for i in self._item_popularity if i not in exclude_items][:n]

        cb_scores = self._cb_scores[row]
        cf_scores = self._cf_scores[row]

        # Mask excluded items
        excl_pos = np.array([self._isbn_to_pos[i]
                              for i in exclude_items
                              if i in self._isbn_to_pos], dtype=np.intp)
        mask = np.ones(len(self._score_isbns), dtype=bool)
        if excl_pos.size:
            mask[excl_pos] = False

        cb_f = cb_scores[mask]
        cf_f = cf_scores[mask]
        isbns_f = [self._score_isbns[i] for i in range(len(self._score_isbns)) if mask[i]]

        if len(isbns_f) == 0:
            return [i for i in self._item_popularity if i not in exclude_items][:n]

        blended = self.config.alpha * _minmax_norm(cf_f) + \
                  (1 - self.config.alpha) * _minmax_norm(cb_f)

        k = min(n, len(blended))
        top = np.argpartition(blended, -k)[-k:]
        top = top[np.argsort(blended[top])[::-1]]
        return [isbns_f[i] for i in top]


# =============================================================================
# Cascade (Generate + Rerank) Hybrid
# =============================================================================

@dataclass
class CascadeHybridConfig:
    """Configuration for the cascade hybrid recommender."""
    n_candidates: int = 300

    funksvd: FunkSVDConfig = field(default_factory=lambda: FunkSVDConfig(
        n_factors=50,
        n_epochs=20,
        lr=0.01,
        reg=0.02,
        early_stopping=True,
        patience=3,
    ))

    cb_similarity: str = "cosine"
    cb_n_neighbors: int = 50
    cb_relevance_threshold: float = 6.0


class CascadeHybrid:
    """
    Two-stage cascade: content-based candidate generation + FunkSVD reranking.

    Stage 1 — Generator (ContentBasedRecommender):
        Selects top-n_candidates by precomputed CB score.

    Stage 2 — Ranker (FunkSVD):
        Re-scores those candidates with vectorised FunkSVD scoring
        (mu + bu + bi[cands] + Q[cands] @ P[u]).

    Who benefits: users reading newly-published books (cold items) that have no
    CF history; CB retrieves them by metadata, FunkSVD personalises the ranking.
    """

    def __init__(self, books_df: pd.DataFrame, config: CascadeHybridConfig):
        self.config = config
        self.cf_model = FunkSVD(config.funksvd)
        self.cb_model = ContentBasedRecommender(
            books_df=books_df,
            similarity_metric=config.cb_similarity,
            n_neighbors=config.cb_n_neighbors,
            relevance_threshold=config.cb_relevance_threshold,
        )
        self._item_popularity: List[str] = []
        self.global_mean: float = 0.0

        self._score_isbns: List[str] = []
        self._isbn_to_pos: Dict[str, int] = {}
        self._cb_scores: np.ndarray = np.array([])
        self._cf_scores: np.ndarray = np.array([])
        self._user_to_row: Dict = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> "CascadeHybrid":
        print("Fitting CascadeHybrid...")
        self.global_mean = train_df["rating"].mean()
        train_isbns = list(train_df["isbn"].unique())

        item_counts = train_df.groupby("isbn")["rating"].count().sort_values(ascending=False)
        self._item_popularity = list(item_counts.index)

        print("  [1/2] Fitting ContentBased (Stage 1 — generator)...")
        self.cb_model.fit(train_df)

        print("  [2/2] Fitting FunkSVD (Stage 2 — ranker)...")
        self.cf_model.fit(train_df, val_df=val_df, verbose=verbose)

        print("  [3/3] Precomputing score tables (train items only)...")
        (self._score_isbns, self._isbn_to_pos,
         self._cb_scores, self._cf_scores,
         self._user_to_row) = _precompute_score_tables(
            self.cb_model, self.cf_model, train_isbns, self.global_mean
        )
        print("CascadeHybrid ready.")
        return self

    # ------------------------------------------------------------------
    # Rating prediction
    # ------------------------------------------------------------------

    def predict(self, user_id, item_id) -> float:
        cf_pred = self.cf_model.predict_one(user_id, item_id)
        cb_pred = self.cb_model.predict(user_id, item_id)
        return float(0.7 * cf_pred + 0.3 * cb_pred)

    def predict_batch(self, user_ids, item_ids) -> np.ndarray:
        return np.array([self.predict(u, i) for u, i in zip(user_ids, item_ids)])

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(self, user_id, n: int, exclude_items: Set) -> List[str]:
        """
        Stage 1: top-n_candidates by CB score (array lookup, no recompute).
        Stage 2: rerank with CF scores (also precomputed).
        """
        row = self._user_to_row.get(user_id)
        if row is None:
            return [i for i in self._item_popularity if i not in exclude_items][:n]

        cb_scores = self._cb_scores[row]
        cf_scores = self._cf_scores[row]

        excl_pos = np.array([self._isbn_to_pos[i]
                              for i in exclude_items
                              if i in self._isbn_to_pos], dtype=np.intp)
        mask = np.ones(len(self._score_isbns), dtype=bool)
        if excl_pos.size:
            mask[excl_pos] = False

        avail_idx = np.where(mask)[0]
        if avail_idx.size == 0:
            return [i for i in self._item_popularity if i not in exclude_items][:n]

        # Stage 1: top-n_candidates by CB
        k1 = min(self.config.n_candidates, avail_idx.size)
        cb_avail = cb_scores[avail_idx]
        top1 = np.argpartition(cb_avail, -k1)[-k1:]

        # Stage 2: rerank those by CF
        cf_top1 = cf_scores[avail_idx[top1]]
        order = np.argsort(cf_top1)[::-1]

        selected = avail_idx[top1[order[:n]]]
        return [self._score_isbns[i] for i in selected]


# =============================================================================
# Shared precomputation utility
# =============================================================================

def _precompute_score_tables(
    cb_model: "ContentBasedRecommender",
    cf_model: "FunkSVD",
    train_isbns: List[str],
    global_mean: float,
) -> Tuple[List[str], Dict[str, int], np.ndarray, np.ndarray, Dict]:
    """
    Precompute CB and CF score arrays for all users over train items.

    Returns
    -------
    score_isbns   : list of length n_items (train items that appear in CB index)
    isbn_to_pos   : isbn -> column index in the score matrices
    cb_scores     : (n_users, n_items) float32 array — cosine similarity
    cf_scores     : (n_users, n_items) float32 array — FunkSVD score
    user_to_row   : user_id -> row index in score matrices

    Restricting to train_isbns (rather than the full 271k catalogue) is the key
    size reduction: 9k items vs 271k items → 30x smaller matrix, fits in RAM,
    computes in seconds.
    """
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    from scipy.sparse import vstack as sp_vstack

    # Items: intersection of train_isbns and CB feature index
    cb_idx = cb_model.isbn_to_idx
    score_isbns = [isbn for isbn in train_isbns if isbn in cb_idx]
    isbn_to_pos = {isbn: pos for pos, isbn in enumerate(score_isbns)}
    n_items = len(score_isbns)

    # Users with CB profiles (needed for cosine similarity)
    users = list(cb_model.user_pref_vectors.keys())
    n_users = len(users)
    user_to_row = {u: i for i, u in enumerate(users)}

    print(f"    Items: {n_items:,}  Users: {n_users:,}")

    # ---- CB scores: one big matrix multiply ----
    item_iidx = np.array([cb_idx[isbn] for isbn in score_isbns])
    item_features_sub = cb_model.item_features[item_iidx]  # (n_items, d) sparse

    pref_matrix = sp_vstack([cb_model.user_pref_vectors[u] for u in users])
    # (n_users, d) sparse  ×  (d, n_items) sparse  →  (n_users, n_items) dense
    cb_scores = sk_cosine(pref_matrix, item_features_sub).astype(np.float32)

    # ---- CF scores: vectorised per user ----
    # Build item index into CF model
    cf_item_iidx = np.array([cf_model.item2idx.get(isbn, -1) for isbn in score_isbns])
    known_cf = cf_item_iidx >= 0  # mask for items in CF

    cf_scores = np.full((n_users, n_items), global_mean, dtype=np.float32)

    known_cf_positions = np.where(known_cf)[0]          # columns where CF knows the item
    cf_iidx_known = cf_item_iidx[known_cf_positions]    # their CF internal indices

    if cf_iidx_known.size > 0:
        # bi for known items, shape (n_known,)
        bi_known = cf_model.bi[cf_iidx_known].astype(np.float32)
        # Q for known items, shape (n_known, n_factors)
        Q_known = cf_model.Q[cf_iidx_known].astype(np.float32)

        for row_idx, user_id in enumerate(users):
            uidx = cf_model.user2idx.get(user_id)
            if uidx is None:
                continue
            bu = float(cf_model.bu[uidx])
            user_vec = cf_model.P[uidx].astype(np.float32)
            # scores for known items: mu + bu + bi + Q @ user_vec
            raw = cf_model.mu + bu + bi_known + Q_known @ user_vec  # (n_known,)
            cf_scores[row_idx, known_cf_positions] = raw.astype(np.float32)

    return score_isbns, isbn_to_pos, cb_scores, cf_scores, user_to_row


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Scale arr to [0, 1]; returns zeros if range is 0."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)
