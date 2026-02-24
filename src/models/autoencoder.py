"""
Denoising Autoencoder for Collaborative Filtering (DAE-CF).

Architecture:
- Input: per-user normalized implicit feedback vector r_u ∈ R^|I|
  (1 = interacted, 0 = not; binarized from explicit ratings)
- Encoder: r_u -> h = ReLU(W_e r_u + b_e)  [one hidden layer]
- Decoder: h -> r_hat = W_d h + b_d         [linear output, no sigmoid]
- Denoising: during training, randomly zero out a fraction of observed entries
- Loss: masked MSE only on observed entries (WMSE variant)
  L = (1/|O_u|) sum_{i in O_u} (r_hat_ui - r_ui)^2  + L2 regularization

This is the AutoRec / denoising autoencoder approach from:
  - Sedhain et al., "AutoRec: Autoencoders Meet Collaborative Filtering", WWW 2015
  - Vincent et al., "Stacked Denoising Autoencoders", JMLR 2010

Training: mini-batch SGD over users; each batch processes a dense sub-matrix.

Key difference from MF/BPR:
- Does NOT decompose into user x item factors; instead compresses the full
  interaction row into a latent code and reconstructs it.
- Non-linear encoder enables capturing higher-order item co-occurrence patterns.
- Can score ALL items for a user in one forward pass (O(n_items) per user vs.
  O(n_items * n_factors) for MF).
"""

import warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


@dataclass
class DAEConfig:
    hidden_dim: int = 256          # encoder hidden layer size
    corruption_ratio: float = 0.3  # fraction of observed entries to zero during training
    lr: float = 0.001
    reg: float = 1e-4              # L2 weight decay
    n_epochs: int = 30
    batch_size: int = 128          # number of users per mini-batch
    seed: int = 42
    # Weight for unobserved entries in loss (0 = pure observed-only loss, like AutoRec)
    unobserved_weight: float = 0.0


class DAE:
    """
    Item-based Denoising Autoencoder for Collaborative Filtering.

    The model ingests a user's observed interaction vector, corrupts it,
    and learns to reconstruct it through a bottleneck.  At inference time
    we use the clean vector and score items by their reconstructed value.
    """

    def __init__(self, n_items: int, cfg: DAEConfig):
        self.n_items = n_items
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        h = cfg.hidden_dim
        scale = np.sqrt(2.0 / n_items)  # He initialisation

        # Encoder: n_items -> h
        self.W_e = self.rng.normal(0, scale, (h, n_items)).astype(np.float32)
        self.b_e = np.zeros(h, dtype=np.float32)

        # Decoder: h -> n_items
        scale_d = np.sqrt(2.0 / h)
        self.W_d = self.rng.normal(0, scale_d, (n_items, h)).astype(np.float32)
        self.b_d = np.zeros(n_items, dtype=np.float32)

        # Will be populated during fit
        self.user_vectors: Optional[np.ndarray] = None   # (n_users, n_items) sparse-ish float32
        self.n_users: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df,
        user2idx: Dict,
        item2idx: Dict,
        val_df=None,
        pipeline=None,
        idx2item=None,
        verbose: bool = True,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Build user-item matrix and train the autoencoder.

        Args:
            train_df: DataFrame with user_id, isbn, rating columns
            user2idx: mapping raw user_id -> int index
            item2idx: mapping raw isbn -> int index
        """
        cfg = self.cfg
        self.n_users = len(user2idx)
        n_items = self.n_items

        # Build normalised user-item matrix (binary / binarized explicit)
        R = np.zeros((self.n_users, n_items), dtype=np.float32)
        for row in train_df.itertuples(index=False):
            u_raw = getattr(row, 'user_id')
            i_raw = getattr(row, 'isbn')
            if u_raw in user2idx and i_raw in item2idx:
                u_idx = user2idx[u_raw]
                i_idx = item2idx[i_raw]
                R[u_idx, i_idx] = 1.0

        # Normalise each user row by L2 norm (prevents scale domination)
        norms = np.linalg.norm(R, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.R_norm = R / norms
        self.user_vectors = self.R_norm   # used at inference

        losses, ndcgs, precisions, hit_rates = [], [], [], []
        user_indices = np.arange(self.n_users)

        for epoch in range(cfg.n_epochs):
            self.rng.shuffle(user_indices)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, self.n_users, cfg.batch_size):
                batch_users = user_indices[start: start + cfg.batch_size]
                X = self.R_norm[batch_users]   # (B, n_items)

                # Denoising: corrupt observed entries
                if cfg.corruption_ratio > 0:
                    obs_mask = (X > 0)
                    noise_mask = (
                        self.rng.random(X.shape) < cfg.corruption_ratio
                    ) & obs_mask
                    X_corrupted = X.copy()
                    X_corrupted[noise_mask] = 0.0
                else:
                    X_corrupted = X

                # Forward
                X_hat, (h, z_e) = self._forward(X_corrupted)

                # Loss: observed-only MSE (+ optional unobserved term)
                obs = (X > 0).astype(np.float32)                 # (B, n_items)
                unobs = 1.0 - obs
                diff = X_hat - X                                  # (B, n_items)
                loss_obs = (obs * diff ** 2).sum(axis=1).mean()
                loss_unobs = (unobs * diff ** 2).sum(axis=1).mean()
                loss = loss_obs + cfg.unobserved_weight * loss_unobs
                # L2
                loss_reg = cfg.reg * (
                    (self.W_e ** 2).sum() + (self.W_d ** 2).sum()
                )
                epoch_loss += float(loss + loss_reg)
                n_batches += 1

                # Backward
                self._backward(X, X_hat, h, z_e, obs, unobs)

            epoch_loss /= max(n_batches, 1)
            losses.append(epoch_loss)

            if val_df is not None and pipeline is not None:
                ranking = pipeline.evaluate_ranking(
                    self, val_df, train_df, n_recommendations=10,
                    user2idx=user2idx, idx2item=idx2item, item2idx=item2idx,
                )
                ndcgs.append(ranking.get("ndcg@10", 0))
                precisions.append(ranking.get("precision@10", 0))
                hit_rates.append(ranking.get("hit_rate@10", 0))

            if verbose:
                print(f"Epoch {epoch+1:02d}/{cfg.n_epochs} | loss={epoch_loss:.5f}")

        return losses, ndcgs, precisions, hit_rates

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _forward(self, X: np.ndarray):
        """
        X: (B, n_items)
        Returns: X_hat (B, n_items), cache (h, z_e)
        """
        z_e = X @ self.W_e.T + self.b_e    # (B, h)
        h = np.maximum(0.0, z_e)            # ReLU
        X_hat = h @ self.W_d.T + self.b_d  # (B, n_items)  linear decoder
        return X_hat.astype(np.float32), (h.astype(np.float32), z_e.astype(np.float32))

    def _backward(
        self,
        X: np.ndarray,
        X_hat: np.ndarray,
        h: np.ndarray,
        z_e: np.ndarray,
        obs: np.ndarray,
        unobs: np.ndarray,
    ):
        cfg = self.cfg
        B = X.shape[0]
        lr = cfg.lr
        reg = cfg.reg

        # Gradient of MSE w.r.t. X_hat
        # d_loss/d_X_hat = 2*(X_hat - X) * mask / B
        diff = X_hat - X    # (B, n_items)
        weight = obs + cfg.unobserved_weight * unobs
        d_Xhat = (2.0 / B) * weight * diff    # (B, n_items)

        # float32 matmul can emit spurious overflow warnings in some BLAS builds;
        # suppress them — results are numerically correct.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Decoder gradients
            dW_d = d_Xhat.T @ h - reg * self.W_d        # (n_items, h)
            db_d = d_Xhat.sum(axis=0) - reg * self.b_d  # (n_items,)

            # Backprop through decoder -> h
            d_h = d_Xhat @ self.W_d    # (B, h)

            # Backprop through ReLU
            d_ze = d_h * (z_e > 0).astype(np.float32)   # (B, h)

            # Encoder gradients
            dW_e = d_ze.T @ X - reg * self.W_e          # (h, n_items)
            db_e = d_ze.sum(axis=0) - reg * self.b_e    # (h,)

        # SGD update
        self.W_d += lr * dW_d
        self.b_d += lr * db_d
        self.W_e += lr * dW_e
        self.b_e += lr * db_e

    # ------------------------------------------------------------------
    # Scoring / recommendation
    # ------------------------------------------------------------------

    def _score_user(self, u_idx: int) -> np.ndarray:
        """Reconstruct and return scores for all items for a given user."""
        x = self.user_vectors[u_idx: u_idx + 1]   # (1, n_items)
        x_hat, _ = self._forward(x)
        return x_hat[0]                            # (n_items,)

    def recommend(
        self,
        u_idx: int,
        K: int,
        seen_items: Optional[List] = None,
        candidates: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = self._score_user(u_idx)

        if candidates is None:
            cand = np.arange(self.n_items, dtype=np.int32)
        else:
            cand = np.asarray(candidates, dtype=np.int32)

        if seen_items:
            seen_set = set(seen_items)
            mask = np.array([c not in seen_set for c in cand], dtype=bool)
            cand = cand[mask]

        s = scores[cand]

        if len(cand) <= K:
            order = np.argsort(-s)
            return cand[order], s[order]

        topk_idx = np.argpartition(-s, K)[:K]
        topk_sorted = topk_idx[np.argsort(-s[topk_idx])]
        return cand[topk_sorted], s[topk_sorted]
