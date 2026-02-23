"""
Evaluation pipeline for recommendation models.

Provides a unified interface for evaluating different types of recommenders:
- Rating prediction models
- Ranking/top-N models
- Hybrid models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional, Any, Protocol
from abc import ABC, abstractmethod

from .metrics import (
    RankingEvaluator,
    RatingEvaluator,
    get_relevant_items,
    print_evaluation_results
)


class RecommenderModel(Protocol):
    """Protocol defining the interface for recommender models."""

    def predict(self, user_id: str, item_id: str) -> float:
        """Predict rating for a user-item pair."""
        ...

    def recommend(self, user_id: str, n: int, exclude_items: Set) -> List[str]:
        """Generate top-N recommendations for a user."""
        ...


class BaselineModel(ABC):
    """Abstract base class for baseline models."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> 'BaselineModel':
        """Fit the model on training data."""
        pass

    @abstractmethod
    def predict(self, user_id: str, item_id: str) -> float:
        """Predict rating for user-item pair."""
        pass

    def predict_batch(self, user_ids: List[str], item_ids: List[str]) -> np.ndarray:
        """Predict ratings for multiple user-item pairs."""
        return np.array([
            self.predict(u, i) for u, i in zip(user_ids, item_ids)
        ])


class GlobalMeanBaseline(BaselineModel):
    """Predicts the global mean rating."""

    def __init__(self):
        self.global_mean = 0.0

    def fit(self, train_df: pd.DataFrame) -> 'GlobalMeanBaseline':
        self.global_mean = train_df['rating'].mean()
        return self

    def predict(self, user_id: str, item_id: str) -> float:
        return self.global_mean


class UserMeanBaseline(BaselineModel):
    """Predicts the user's mean rating (falls back to global mean)."""

    def __init__(self):
        self.global_mean = 0.0
        self.user_means = {}

    def fit(self, train_df: pd.DataFrame) -> 'UserMeanBaseline':
        self.global_mean = train_df['rating'].mean()
        self.user_means = train_df.groupby('user_id')['rating'].mean().to_dict()
        return self

    def predict(self, user_id: str, item_id: str) -> float:
        return self.user_means.get(user_id, self.global_mean)


class ItemMeanBaseline(BaselineModel):
    """Predicts the item's mean rating (falls back to global mean)."""

    def __init__(self):
        self.global_mean = 0.0
        self.item_means = {}

    def fit(self, train_df: pd.DataFrame) -> 'ItemMeanBaseline':
        self.global_mean = train_df['rating'].mean()
        self.item_means = train_df.groupby('isbn')['rating'].mean().to_dict()
        return self

    def predict(self, user_id: str, item_id: str) -> float:
        return self.item_means.get(item_id, self.global_mean)


class PopularityBaseline(BaselineModel):
    """
    Popularity-based baseline for ranking.

    Recommends most popular items (by rating count or average rating).
    """

    def __init__(self, by: str = 'count'):
        """
        Args:
            by: 'count' for most rated items, 'rating' for highest rated
        """
        self.by = by
        self.global_mean = 0.0
        self.item_popularity = []
        self.item_scores = {}

    def fit(self, train_df: pd.DataFrame) -> 'PopularityBaseline':
        self.global_mean = train_df['rating'].mean()

        if self.by == 'count':
            # Sort by number of ratings
            item_stats = train_df.groupby('isbn').agg({
                'rating': ['count', 'mean']
            })
            item_stats.columns = ['count', 'mean']
            item_stats = item_stats.sort_values('count', ascending=False)
        else:
            # Sort by average rating (with minimum support)
            item_stats = train_df.groupby('isbn').agg({
                'rating': ['count', 'mean']
            })
            item_stats.columns = ['count', 'mean']
            # Require minimum 5 ratings
            item_stats = item_stats[item_stats['count'] >= 5]
            item_stats = item_stats.sort_values('mean', ascending=False)

        self.item_popularity = list(item_stats.index)
        self.item_scores = item_stats['mean'].to_dict()

        return self

    def predict(self, user_id: str, item_id: str) -> float:
        return self.item_scores.get(item_id, self.global_mean)

    def recommend(self, user_id: str, n: int, exclude_items: Set) -> List[str]:
        """Recommend top-N popular items not in exclude set."""
        recommendations = []
        for item in self.item_popularity:
            if item not in exclude_items:
                recommendations.append(item)
                if len(recommendations) >= n:
                    break
        return recommendations


class EvaluationPipeline:
    """
    Main evaluation pipeline for recommendation models.

    Handles both rating prediction and ranking evaluation.
    """

    def __init__(
        self,
        k_values: List[int] = [5, 10, 20],
        relevance_threshold: float = 1.0
    ):
        """
        Args:
            k_values: K values for ranking metrics
            relevance_threshold: Minimum rating to consider relevant
        """
        self.k_values = k_values
        self.relevance_threshold = relevance_threshold
        self.ranking_evaluator = RankingEvaluator(k_values)
        self.rating_evaluator = RatingEvaluator()

    def evaluate_rating_prediction(
        self,
        model: BaselineModel,
        test_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate model on rating prediction task.

        Args:
            model: Trained model with predict method
            test_df: Test data

        Returns:
            Dictionary of rating prediction metrics
        """
        y_true = test_df['rating'].values
        y_pred = model.predict_batch(
            test_df['user_id'].tolist(),
            test_df['isbn'].tolist()
        )

        return self.rating_evaluator.evaluate(y_true, y_pred)

    def evaluate_ranking(
        self,
        model,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        n_recommendations: int = 20,
        user2idx = None,
        idx2item = None,
        item2idx = None
    ) -> Dict[str, float]:
        """
        Evaluate model on ranking task.

        For each user in test set:
        1. Get items they rated highly (relevant items)
        2. Generate recommendations (excluding training items)
        3. Compute ranking metrics

        Args:
            model: Trained model with recommend method
            test_df: Test data
            train_df: Training data (to exclude from recommendations)
            n_recommendations: Number of recommendations to generate

        Returns:
            Dictionary of ranking metrics
        """
        # Get relevant items per user (items rated >= threshold in test)
        user_relevant = get_relevant_items(test_df, self.relevance_threshold)

        # Get items each user has in training (to exclude)
        user_train_items = train_df.groupby('user_id')['isbn'].apply(set).to_dict()

        # Get all items for coverage calculation
        all_items = set(train_df['isbn'].unique())

        # Generate recommendations for each user
        user_recommendations = {}

        for user_id in user_relevant:
            exclude = user_train_items.get(user_id, set())

            if user2idx != None:
                u_idx = user2idx[user_id]
                exclude = [item2idx[i] for i in exclude]
                try:
                    recs, _ = model.recommend(u_idx, n_recommendations, exclude)
                    user_recommendations[user_id] = [idx2item[i] for i in recs]
                except Exception as e:
                    # Skip users that can't get recommendations
                    continue
            else:
                try:
                    recs = model.recommend(user_id, n_recommendations, exclude)
                    user_recommendations[user_id] = recs
                except Exception as e:
                    # Skip users that can't get recommendations
                    continue

        # Evaluate
        results = self.ranking_evaluator.evaluate_all(
            user_recommendations,
            user_relevant,
            total_items=len(all_items)
        )

        return results
    
    def evaluate_ranking_graph(
        self,
        recs_dict,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        n_recommendations: int = 20
    ) -> Dict[str, float]:
        """
        Evaluate model on ranking task.

        For each user in test set:
        1. Get items they rated highly (relevant items)
        2. Generate recommendations (excluding training items)
        3. Compute ranking metrics

        Args:
            model: Trained model with recommend method
            test_df: Test data
            train_df: Training data (to exclude from recommendations)
            n_recommendations: Number of recommendations to generate

        Returns:
            Dictionary of ranking metrics
        """
        # Get relevant items per user (items rated >= threshold in test)
        user_relevant = get_relevant_items(test_df, self.relevance_threshold)
        print("test_df len", len(test_df['user_id'].drop_duplicates()))
        print(self.relevance_threshold)
        print("user_relevant len", len(user_relevant))


        # Get items each user has in training (to exclude)
        user_train_items = train_df.groupby('user_id')['isbn'].apply(set).to_dict()

        # Get all items for coverage calculation
        all_items = set(train_df['isbn'].unique())

        # Generate recommendations for each user
        user_recommendations = {}

        for user_id in user_relevant:
            try:
                user_recommendations[user_id] = recs_dict[user_id]
            except Exception as e:
                print("Except")
                # Skip users that can't get recommendations
                continue

        # Evaluate
        results = self.ranking_evaluator.evaluate_all(
            user_recommendations,
            user_relevant,
            total_items=len(all_items)
        )

        return results
    
    
    def evaluate_ranking_heuristic(
        self,
        recs_ranked,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        n_recommendations: int = 20
    ) -> Dict[str, float]:
        """
        Evaluate model on ranking task.

        For each user in test set:
        1. Get items they rated highly (relevant items)
        2. Generate recommendations (excluding training items)
        3. Compute ranking metrics

        Args:
            model: Trained model with recommend method
            test_df: Test data
            train_df: Training data (to exclude from recommendations)
            n_recommendations: Number of recommendations to generate

        Returns:
            Dictionary of ranking metrics
        """
        # Get relevant items per user (items rated >= threshold in test)
        user_relevant = get_relevant_items(test_df, self.relevance_threshold)
        print("test_df len", len(test_df['user_id'].drop_duplicates()))
        print(self.relevance_threshold)
        print("user_relevant len", len(user_relevant))


        # Get items each user has in training (to exclude)
        user_train_items = train_df.groupby('user_id')['isbn'].apply(set).to_dict()

        # Get all items for coverage calculation
        all_items = set(train_df['isbn'].unique())

        # Generate recommendations for each user
        user_recommendations = {}

        for user_id in user_relevant:
            exclude = user_train_items.get(user_id, set())
            try:
                recs = [i for i in recs_ranked if i not in exclude][:n_recommendations]
                user_recommendations[user_id] = recs
            except Exception as e:
                print("Except")
                # Skip users that can't get recommendations
                continue

        # Evaluate
        results = self.ranking_evaluator.evaluate_all(
            user_recommendations,
            user_relevant,
            total_items=len(all_items)
        )

        return results
    
    


    def evaluate_full(
        self,
        model,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Run full evaluation (both rating prediction and ranking).

        Args:
            model: Trained model
            test_df: Test data
            train_df: Training data
            model_name: Name for printing

        Returns:
            Combined dictionary of all metrics
        """
        results = {}

        # Rating prediction
        if hasattr(model, 'predict'):
            rating_results = self.evaluate_rating_prediction(model, test_df)
            results.update(rating_results)

        # Ranking
        if hasattr(model, 'recommend'):
            ranking_results = self.evaluate_ranking(model, test_df, train_df)
            results.update(ranking_results)

        print_evaluation_results(results, f"Evaluation Results: {model_name}")

        return results


def run_baseline_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, Dict[str, float]]:
    """
    Run evaluation for all baseline models.

    Args:
        train_df: Training data
        test_df: Test data
        k_values: K values for ranking metrics

    Returns:
        Dictionary of model_name -> metrics
    """
    pipeline = EvaluationPipeline(k_values=k_values)
    results = {}

    # Rating prediction baselines
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)

    baselines = [
        ("Global Mean", GlobalMeanBaseline()),
        ("User Mean", UserMeanBaseline()),
        ("Item Mean", ItemMeanBaseline()),
    ]

    for name, model in baselines:
        print(f"\nEvaluating {name}...")
        model.fit(train_df)
        rating_results = pipeline.evaluate_rating_prediction(model, test_df)
        results[name] = rating_results
        print(f"  RMSE: {rating_results['rmse']:.4f}")
        print(f"  MAE:  {rating_results['mae']:.4f}")

    # Popularity baseline (ranking)
    print(f"\nEvaluating Popularity Baseline...")
    pop_model = PopularityBaseline(by='count')
    pop_model.fit(train_df)

    pop_results = pipeline.evaluate_rating_prediction(pop_model, test_df)
    ranking_results = pipeline.evaluate_ranking(pop_model, test_df, train_df)
    pop_results.update(ranking_results)
    results["Popularity"] = pop_results

    print(f"  RMSE: {pop_results['rmse']:.4f}")
    print(f"  NDCG@10: {pop_results.get('ndcg@10', 0):.4f}")
    print(f"  Hit Rate@10: {pop_results.get('hit_rate@10', 0):.4f}")

    return results
