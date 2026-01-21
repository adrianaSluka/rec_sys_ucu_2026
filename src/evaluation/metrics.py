"""
Evaluation metrics for recommendation systems.

Implements:
1. Rating Prediction Metrics: RMSE, MAE
2. Ranking Metrics: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate@K
3. Coverage Metrics: Catalog Coverage, User Coverage
"""

import numpy as np
from typing import List, Dict, Optional, Set
from collections import defaultdict


# =============================================================================
# RATING PREDICTION METRICS
# =============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Primary metric for rating prediction tasks.
    Penalizes large errors more heavily than MAE.

    Args:
        y_true: Ground truth ratings
        y_pred: Predicted ratings

    Returns:
        RMSE value (lower is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Secondary metric for rating prediction.
    More interpretable than RMSE (average absolute deviation).

    Args:
        y_true: Ground truth ratings
        y_pred: Predicted ratings

    Returns:
        MAE value (lower is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


# =============================================================================
# RANKING METRICS
# =============================================================================

def precision_at_k(recommended: List, relevant: Set, k: int) -> float:
    """
    Precision@K: Fraction of recommended items that are relevant.

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of relevant (ground truth) item IDs
        k: Number of recommendations to consider

    Returns:
        Precision@K value in [0, 1] (higher is better)
    """
    if k <= 0:
        return 0.0

    recommended_k = recommended[:k]
    if len(recommended_k) == 0:
        return 0.0

    n_relevant = len(set(recommended_k) & relevant)
    return n_relevant / len(recommended_k)


def recall_at_k(recommended: List, relevant: Set, k: int) -> float:
    """
    Recall@K: Fraction of relevant items that are recommended.

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of relevant (ground truth) item IDs
        k: Number of recommendations to consider

    Returns:
        Recall@K value in [0, 1] (higher is better)
    """
    if len(relevant) == 0 or k <= 0:
        return 0.0

    recommended_k = recommended[:k]
    n_relevant = len(set(recommended_k) & relevant)
    return n_relevant / len(relevant)


def hit_rate_at_k(recommended: List, relevant: Set, k: int) -> float:
    """
    Hit Rate@K: Whether at least one relevant item appears in top-K.

    Also known as Hit@K or HR@K.

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of relevant (ground truth) item IDs
        k: Number of recommendations to consider

    Returns:
        1.0 if hit, 0.0 otherwise
    """
    if len(relevant) == 0 or k <= 0:
        return 0.0

    recommended_k = set(recommended[:k])
    return 1.0 if len(recommended_k & relevant) > 0 else 0.0


def dcg_at_k(recommended: List, relevant: Set, k: int) -> float:
    """
    Discounted Cumulative Gain at K.

    Uses binary relevance (1 if relevant, 0 otherwise).

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of relevant (ground truth) item IDs
        k: Number of recommendations to consider

    Returns:
        DCG@K value
    """
    if k <= 0:
        return 0.0

    recommended_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            # Using log2(i+2) so position 0 has discount log2(2)=1
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def ndcg_at_k(recommended: List, relevant: Set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    Primary ranking metric. Accounts for position of relevant items.
    Normalized by ideal DCG (all relevant items at top).

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of relevant (ground truth) item IDs
        k: Number of recommendations to consider

    Returns:
        NDCG@K value in [0, 1] (higher is better)
    """
    if len(relevant) == 0 or k <= 0:
        return 0.0

    # Actual DCG
    dcg = dcg_at_k(recommended, relevant, k)

    # Ideal DCG (all relevant items at top positions)
    ideal_k = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision_at_k(recommended: List, relevant: Set, k: int) -> float:
    """
    Average Precision at K.

    Computes precision at each relevant item position and averages.

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of relevant (ground truth) item IDs
        k: Number of recommendations to consider

    Returns:
        AP@K value in [0, 1] (higher is better)
    """
    if len(relevant) == 0 or k <= 0:
        return 0.0

    recommended_k = recommended[:k]
    score = 0.0
    n_relevant_found = 0

    for i, item in enumerate(recommended_k):
        if item in relevant:
            n_relevant_found += 1
            # Precision at this position
            score += n_relevant_found / (i + 1)

    # Normalize by total relevant items (not just those in top-k)
    return score / len(relevant)


def mrr_at_k(recommended: List, relevant: Set, k: int) -> float:
    """
    Mean Reciprocal Rank at K.

    Reciprocal of the rank of the first relevant item.

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of relevant (ground truth) item IDs
        k: Number of recommendations to consider

    Returns:
        MRR@K value in [0, 1] (higher is better)
    """
    if len(relevant) == 0 or k <= 0:
        return 0.0

    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            return 1.0 / (i + 1)

    return 0.0


# =============================================================================
# COVERAGE METRICS
# =============================================================================

def catalog_coverage(
    all_recommendations: List[List],
    total_items: int
) -> float:
    """
    Catalog Coverage: Fraction of items that are ever recommended.

    Measures diversity of recommendations across users.

    Args:
        all_recommendations: List of recommendation lists (one per user)
        total_items: Total number of items in catalog

    Returns:
        Coverage value in [0, 1] (higher means more diverse)
    """
    if total_items == 0:
        return 0.0

    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)

    return len(recommended_items) / total_items


def user_coverage(
    n_users_with_recs: int,
    total_users: int
) -> float:
    """
    User Coverage: Fraction of users who received recommendations.

    Args:
        n_users_with_recs: Number of users who got at least one recommendation
        total_users: Total number of users

    Returns:
        Coverage value in [0, 1]
    """
    if total_users == 0:
        return 0.0
    return n_users_with_recs / total_users


# =============================================================================
# AGGREGATED EVALUATION
# =============================================================================

class RankingEvaluator:
    """
    Evaluator for ranking-based recommendation evaluation.

    Computes multiple metrics across all users and aggregates results.
    """

    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Args:
            k_values: List of K values to evaluate at
        """
        self.k_values = k_values

    def evaluate_user(
        self,
        recommended: List,
        relevant: Set,
    ) -> Dict[str, float]:
        """
        Evaluate recommendations for a single user.

        Args:
            recommended: Ordered list of recommended item IDs
            relevant: Set of relevant (ground truth) item IDs

        Returns:
            Dictionary of metric_name -> value
        """
        results = {}

        for k in self.k_values:
            results[f'precision@{k}'] = precision_at_k(recommended, relevant, k)
            results[f'recall@{k}'] = recall_at_k(recommended, relevant, k)
            results[f'ndcg@{k}'] = ndcg_at_k(recommended, relevant, k)
            results[f'hit_rate@{k}'] = hit_rate_at_k(recommended, relevant, k)
            results[f'map@{k}'] = average_precision_at_k(recommended, relevant, k)
            results[f'mrr@{k}'] = mrr_at_k(recommended, relevant, k)

        return results

    def evaluate_all(
        self,
        user_recommendations: Dict[str, List],
        user_relevant: Dict[str, Set],
        total_items: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate recommendations for all users and aggregate.

        Args:
            user_recommendations: Dict of user_id -> recommended item list
            user_relevant: Dict of user_id -> set of relevant items
            total_items: Total items in catalog (for coverage)

        Returns:
            Dictionary of aggregated metrics
        """
        all_results = defaultdict(list)
        all_recs_flat = []

        users_evaluated = 0

        for user_id in user_relevant:
            if user_id not in user_recommendations:
                continue

            recommended = user_recommendations[user_id]
            relevant = user_relevant[user_id]

            if len(relevant) == 0:
                continue

            user_results = self.evaluate_user(recommended, relevant)
            for metric, value in user_results.items():
                all_results[metric].append(value)

            all_recs_flat.append(recommended)
            users_evaluated += 1

        # Aggregate (mean across users)
        aggregated = {}
        for metric, values in all_results.items():
            aggregated[metric] = np.mean(values) if values else 0.0

        # Add coverage metrics
        if total_items is not None:
            max_k = max(self.k_values)
            recs_at_max_k = [r[:max_k] for r in all_recs_flat]
            aggregated[f'catalog_coverage@{max_k}'] = catalog_coverage(
                recs_at_max_k, total_items
            )

        aggregated['users_evaluated'] = users_evaluated
        aggregated['user_coverage'] = user_coverage(
            users_evaluated, len(user_relevant)
        )

        return aggregated


class RatingEvaluator:
    """
    Evaluator for rating prediction tasks.
    """

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate rating predictions.

        Args:
            y_true: Ground truth ratings
            y_pred: Predicted ratings

        Returns:
            Dictionary of metric_name -> value
        """
        return {
            'rmse': rmse(y_true, y_pred),
            'mae': mae(y_true, y_pred),
            'n_predictions': len(y_true)
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def binarize_ratings(
    ratings: np.ndarray,
    threshold: float = 6.0
) -> np.ndarray:
    """
    Convert ratings to binary relevance.

    For Book Crossing dataset (1-10 scale), ratings >= 6 are typically
    considered "positive" interactions.

    Args:
        ratings: Array of ratings
        threshold: Threshold for positive relevance

    Returns:
        Binary array (1 for relevant, 0 otherwise)
    """
    return (np.asarray(ratings) >= threshold).astype(int)


def get_relevant_items(
    test_df,
    threshold: float = 6.0
) -> Dict[str, Set]:
    """
    Extract relevant items per user from test data.

    Args:
        test_df: Test DataFrame with user_id, isbn, rating
        threshold: Minimum rating to consider item relevant

    Returns:
        Dict of user_id -> set of relevant item IDs
    """
    relevant = test_df[test_df['rating'] >= threshold]
    user_relevant = relevant.groupby('user_id')['isbn'].apply(set).to_dict()
    return user_relevant


def print_evaluation_results(results: Dict[str, float], title: str = "Evaluation Results"):
    """
    Pretty print evaluation results.

    Args:
        results: Dictionary of metric -> value
        title: Title for the output
    """
    print("=" * 60)
    print(title)
    print("=" * 60)

    # Group metrics by type
    ranking_metrics = {}
    coverage_metrics = {}
    rating_metrics = {}
    other_metrics = {}

    for metric, value in results.items():
        if any(x in metric for x in ['precision', 'recall', 'ndcg', 'hit_rate', 'map', 'mrr']):
            ranking_metrics[metric] = value
        elif 'coverage' in metric:
            coverage_metrics[metric] = value
        elif metric in ['rmse', 'mae']:
            rating_metrics[metric] = value
        else:
            other_metrics[metric] = value

    if ranking_metrics:
        print("\nRanking Metrics:")
        print("-" * 40)
        for metric, value in sorted(ranking_metrics.items()):
            print(f"  {metric:<20}: {value:.4f}")

    if rating_metrics:
        print("\nRating Prediction Metrics:")
        print("-" * 40)
        for metric, value in sorted(rating_metrics.items()):
            print(f"  {metric:<20}: {value:.4f}")

    if coverage_metrics:
        print("\nCoverage Metrics:")
        print("-" * 40)
        for metric, value in sorted(coverage_metrics.items()):
            print(f"  {metric:<20}: {value:.4f}")

    if other_metrics:
        print("\nOther:")
        print("-" * 40)
        for metric, value in sorted(other_metrics.items()):
            if isinstance(value, float):
                print(f"  {metric:<20}: {value:.4f}")
            else:
                print(f"  {metric:<20}: {value}")
