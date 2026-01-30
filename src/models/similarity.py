"""
Shared similarity computation utilities.

Provides cosine, Jaccard, and Pearson similarity functions
used by both content-based and collaborative filtering models.
"""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(matrix):
    """
    Compute pairwise cosine similarity.

    Args:
        matrix: Dense or sparse matrix (n_items, n_features)

    Returns:
        Dense ndarray of shape (n_items, n_items)
    """
    return cosine_similarity(matrix)


def compute_jaccard_similarity(binary_matrix):
    """
    Compute pairwise Jaccard similarity from a binary matrix.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        binary_matrix: Binary sparse or dense matrix (n_items, n_features)

    Returns:
        Dense ndarray of shape (n_items, n_items)
    """
    if not issparse(binary_matrix):
        binary_matrix = csr_matrix(binary_matrix)

    binary_matrix = binary_matrix.astype(float)
    intersection = (binary_matrix @ binary_matrix.T).toarray()
    row_sums = np.array(binary_matrix.sum(axis=1)).flatten()
    union = row_sums[:, None] + row_sums[None, :] - intersection
    union[union == 0] = 1.0
    return intersection / union


def compute_pearson_similarity(rating_matrix):
    """
    Compute pairwise Pearson correlation from a sparse rating matrix.

    Mean-centers each row (item), then computes cosine similarity on
    the centered matrix. Zero entries are treated as missing.

    Args:
        rating_matrix: Sparse matrix (n_items, n_users), zeros = missing

    Returns:
        Dense ndarray of shape (n_items, n_items)
    """
    if issparse(rating_matrix):
        dense = rating_matrix.toarray().astype(np.float64)
    else:
        dense = np.array(rating_matrix, dtype=np.float64)

    # Compute mean over non-zero entries per row
    mask = dense != 0
    row_sums = np.sum(dense, axis=1)
    row_counts = np.sum(mask, axis=1)
    row_counts[row_counts == 0] = 1
    row_means = row_sums / row_counts

    # Center non-zero entries
    centered = dense.copy()
    centered[mask] -= np.broadcast_to(row_means[:, None], centered.shape)[mask]
    # Keep zeros as zeros (missing, not centered)
    centered[~mask] = 0

    centered_sparse = csr_matrix(centered)
    return cosine_similarity(centered_sparse)


def get_top_k_neighbors(sim_matrix, k):
    """
    For each item, extract the top-K most similar neighbors.

    Args:
        sim_matrix: Dense similarity matrix (n, n)
        k: Number of neighbors to keep

    Returns:
        Dict mapping row index -> list of (neighbor_idx, similarity) tuples,
        sorted by descending similarity. Self-similarity excluded.
    """
    n = sim_matrix.shape[0]
    neighbors = {}

    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # exclude self

        # Get top-k indices
        if k < n - 1:
            top_idx = np.argpartition(sims, -k)[-k:]
        else:
            top_idx = np.arange(n)
            top_idx = top_idx[top_idx != i]

        # Sort by descending similarity
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        # Filter out non-positive similarities
        neighbors[i] = [(int(j), float(sims[j])) for j in top_idx if sims[j] > 0]

    return neighbors
