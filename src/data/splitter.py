"""
Data splitting utilities for offline evaluation.

Implements temporal splitting strategy based on book publication year.
Since the Book Crossing dataset lacks explicit timestamps, we use publication
year as a proxy for temporal ordering.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    # Year thresholds for temporal split
    train_max_year: int = 1999      # Train: books published <= 1999
    val_max_year: int = 2001        # Validation: books published 2000-2001
    # Test: books published 2002-2004

    # Minimum interactions thresholds
    min_user_interactions: int = 5   # Users must have at least this many ratings
    min_item_interactions: int = 5   # Items must have at least this many ratings

    # Whether to use only explicit ratings
    explicit_only: bool = True       # Filter out implicit (rating=0) interactions

    # Random seed for reproducibility
    random_seed: int = 42


def filter_data(
    ratings: pd.DataFrame,
    books: pd.DataFrame,
    config: SplitConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter ratings and books based on configuration.

    Applies:
    1. Explicit rating filter (optional)
    2. Valid publication year filter
    3. Minimum user interaction filter
    4. Minimum item interaction filter

    Args:
        ratings: DataFrame with user_id, isbn, rating
        books: DataFrame with isbn, year, etc.
        config: Split configuration

    Returns:
        Filtered (ratings, books) DataFrames
    """
    df = ratings.copy()

    # Filter explicit ratings only
    if config.explicit_only:
        df = df[df['rating'] > 0]
        print(f"After explicit filter: {len(df):,} ratings")

    # Merge with books to get publication year
    df = df.merge(books[['isbn', 'year']], on='isbn', how='inner')

    # Filter valid publication years (1900-2004, dataset collected in 2004)
    df = df[(df['year'] >= 1900) & (df['year'] <= 2004)]
    print(f"After year filter (1900-2004): {len(df):,} ratings")

    # Iteratively filter users and items until stable
    prev_len = 0
    iteration = 0
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1

        # Filter users with minimum interactions
        user_counts = df.groupby('user_id').size()
        valid_users = user_counts[user_counts >= config.min_user_interactions].index
        df = df[df['user_id'].isin(valid_users)]

        # Filter items with minimum interactions
        item_counts = df.groupby('isbn').size()
        valid_items = item_counts[item_counts >= config.min_item_interactions].index
        df = df[df['isbn'].isin(valid_items)]

        print(f"Iteration {iteration}: {len(df):,} ratings, "
              f"{df['user_id'].nunique():,} users, {df['isbn'].nunique():,} items")

    # Filter books to only those in ratings
    filtered_books = books[books['isbn'].isin(df['isbn'].unique())]

    return df, filtered_books


def temporal_split(
    ratings: pd.DataFrame,
    config: SplitConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally based on book publication year.

    Strategy:
    - Train: ratings for books published <= train_max_year
    - Validation: ratings for books published in (train_max_year, val_max_year]
    - Test: ratings for books published > val_max_year

    This simulates a realistic scenario where we train on historical data
    and evaluate on newer releases.

    Args:
        ratings: Filtered DataFrame with user_id, isbn, rating, year
        config: Split configuration

    Returns:
        (train_df, val_df, test_df) DataFrames
    """
    # Temporal split based on publication year
    train_mask = ratings['year'] <= config.train_max_year
    val_mask = (ratings['year'] > config.train_max_year) & (ratings['year'] <= config.val_max_year)
    test_mask = ratings['year'] > config.val_max_year

    train_df = ratings[train_mask].copy()
    val_df = ratings[val_mask].copy()
    test_df = ratings[test_mask].copy()

    return train_df, val_df, test_df


def ensure_test_users_in_train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_train_interactions: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ensure users in val/test sets have sufficient history in training set.

    This is critical for collaborative filtering evaluation - we can only
    evaluate users for whom we have some training data.

    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        min_train_interactions: Minimum interactions required in training set

    Returns:
        Filtered (train_df, val_df, test_df)
    """
    # Find users with sufficient training history
    train_user_counts = train_df.groupby('user_id').size()
    valid_users = set(train_user_counts[train_user_counts >= min_train_interactions].index)

    # Filter val and test to only include these users
    val_df = val_df[val_df['user_id'].isin(valid_users)].copy()
    test_df = test_df[test_df['user_id'].isin(valid_users)].copy()

    return train_df, val_df, test_df


def create_temporal_split(
    ratings: pd.DataFrame,
    books: pd.DataFrame,
    config: Optional[SplitConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Main function to create temporal train/validation/test split.

    Args:
        ratings: Raw ratings DataFrame
        books: Books DataFrame with publication years
        config: Split configuration (uses defaults if None)

    Returns:
        (train_df, val_df, test_df, split_info)
    """
    if config is None:
        config = SplitConfig()

    print("=" * 60)
    print("CREATING TEMPORAL SPLIT")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Train: books published <= {config.train_max_year}")
    print(f"  Validation: books published {config.train_max_year+1}-{config.val_max_year}")
    print(f"  Test: books published {config.val_max_year+1}-2004")
    print(f"  Min user interactions: {config.min_user_interactions}")
    print(f"  Min item interactions: {config.min_item_interactions}")
    print(f"  Explicit ratings only: {config.explicit_only}")

    # Step 1: Filter data
    print(f"\n--- Step 1: Filtering data ---")
    print(f"Original ratings: {len(ratings):,}")
    filtered_ratings, filtered_books = filter_data(ratings, books, config)

    # Step 2: Temporal split
    print(f"\n--- Step 2: Temporal split ---")
    train_df, val_df, test_df = temporal_split(filtered_ratings, config)
    print(f"Initial split sizes:")
    print(f"  Train: {len(train_df):,} ratings")
    print(f"  Val: {len(val_df):,} ratings")
    print(f"  Test: {len(test_df):,} ratings")

    # Step 3: Ensure test users have training history
    print(f"\n--- Step 3: Ensuring test users have training history ---")
    train_df, val_df, test_df = ensure_test_users_in_train(
        train_df, val_df, test_df, min_train_interactions=1
    )

    # Compile split information
    split_info = {
        'config': config,
        'train': {
            'n_ratings': len(train_df),
            'n_users': train_df['user_id'].nunique(),
            'n_items': train_df['isbn'].nunique(),
            'year_range': (int(train_df['year'].min()), int(train_df['year'].max())),
            'rating_mean': train_df['rating'].mean(),
        },
        'val': {
            'n_ratings': len(val_df),
            'n_users': val_df['user_id'].nunique(),
            'n_items': val_df['isbn'].nunique(),
            'year_range': (int(val_df['year'].min()), int(val_df['year'].max())) if len(val_df) > 0 else (0, 0),
            'rating_mean': val_df['rating'].mean() if len(val_df) > 0 else 0,
        },
        'test': {
            'n_ratings': len(test_df),
            'n_users': test_df['user_id'].nunique(),
            'n_items': test_df['isbn'].nunique(),
            'year_range': (int(test_df['year'].min()), int(test_df['year'].max())) if len(test_df) > 0 else (0, 0),
            'rating_mean': test_df['rating'].mean() if len(test_df) > 0 else 0,
        },
    }

    # Print summary
    print(f"\n--- Final Split Summary ---")
    print(f"{'Set':<10} {'Ratings':>12} {'Users':>10} {'Items':>10} {'Years':>15} {'Avg Rating':>12}")
    print("-" * 70)
    for split_name in ['train', 'val', 'test']:
        info = split_info[split_name]
        year_str = f"{info['year_range'][0]}-{info['year_range'][1]}"
        print(f"{split_name:<10} {info['n_ratings']:>12,} {info['n_users']:>10,} "
              f"{info['n_items']:>10,} {year_str:>15} {info['rating_mean']:>12.2f}")

    return train_df, val_df, test_df, split_info


def create_user_item_mappings(
    train_df: pd.DataFrame
) -> Tuple[dict, dict, dict, dict]:
    """
    Create mappings between original IDs and integer indices.

    Required for matrix-based methods (matrix factorization, etc.)

    Args:
        train_df: Training data

    Returns:
        (user2idx, idx2user, item2idx, idx2item) dictionaries
    """
    unique_users = train_df['user_id'].unique()
    unique_items = train_df['isbn'].unique()

    user2idx = {user: idx for idx, user in enumerate(unique_users)}
    idx2user = {idx: user for user, idx in user2idx.items()}

    item2idx = {item: idx for idx, item in enumerate(unique_items)}
    idx2item = {idx: item for item, idx in item2idx.items()}

    return user2idx, idx2user, item2idx, idx2item
