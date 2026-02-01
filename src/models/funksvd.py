from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class FunkSVDConfig:
    n_factors: int = 50
    n_epochs: int = 20
    lr: float = 0.01
    reg: float = 0.02          # L2 regularization for P, Q, bu, bi
    shuffle: bool = True
    seed: int = 42

    # optional rating clipping (Book-Crossing is typically 0..10)
    min_rating: Optional[float] = None
    max_rating: Optional[float] = None

    # early stopping
    early_stopping: bool = True
    patience: int = 3
    tol: float = 1e-4


class FunkSVD:
    """
    FunkSVD / biased MF trained with SGD:
      r_hat(u,i) = mu + bu[u] + bi[i] + <P[u], Q[i]>
    """

    def __init__(self, config: FunkSVDConfig):
        self.config = config
        self.mu: float = 0.0
        self.user2idx: Dict[int, int] = {}
        self.item2idx: Dict[str, int] = {}
        self.idx2item: Dict[int, str] = {}  # Fixed: maps index -> item_id
        self.P: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.bu: Optional[np.ndarray] = None
        self.bi: Optional[np.ndarray] = None

    def _build_index(self, df: pd.DataFrame, user_col: str, item_col: str) -> None:
        users = df[user_col].unique()
        items = df[item_col].unique()
        self.user2idx = {u: k for k, u in enumerate(users)}
        self.item2idx = {i: k for k, i in enumerate(items)}
        self.idx2item = {idx: item for item, idx in self.item2idx.items()}

    def _init_params(self) -> None:
        rng = np.random.default_rng(self.config.seed)
        n_users = len(self.user2idx)
        n_items = len(self.item2idx)
        k = self.config.n_factors

        # small random init
        self.P = 0.01 * rng.standard_normal((n_users, k))
        self.Q = 0.01 * rng.standard_normal((n_items, k))
        self.bu = np.zeros(n_users, dtype=np.float64)
        self.bi = np.zeros(n_items, dtype=np.float64)

    def _df_to_arrays(
        self, df: pd.DataFrame, user_col: str, item_col: str, rating_col: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = df[user_col].map(self.user2idx).to_numpy()
        i = df[item_col].map(self.item2idx).to_numpy()
        r = df[rating_col].to_numpy(dtype=np.float64)
        return u, i, r

    def fit(
        self,
        train_df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "isbn",
        rating_col: str = "rating",
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> "FunkSVD":
        # Build indices on TRAIN ONLY (important!)
        self._build_index(train_df, user_col, item_col)
        self._init_params()

        # global mean from train
        self.mu = float(train_df[rating_col].mean())

        # prepare training arrays
        u_train, i_train, r_train = self._df_to_arrays(train_df, user_col, item_col, rating_col)

        rng = np.random.default_rng(self.config.seed)
        best_val = np.inf
        bad_epochs = 0

        for epoch in range(1, self.config.n_epochs + 1):
            # shuffle
            if self.config.shuffle:
                idx = rng.permutation(len(r_train))
                u = u_train[idx]; it = i_train[idx]; r = r_train[idx]
            else:
                u = u_train; it = i_train; r = r_train

            # SGD loop
            P, Q, bu, bi = self.P, self.Q, self.bu, self.bi
            lr = self.config.lr
            reg = self.config.reg

            for uu, ii, rr in zip(u, it, r):
                # prediction
                pred = self.mu + bu[uu] + bi[ii] + np.dot(P[uu], Q[ii])

                # error
                err = rr - pred

                # updates (biased MF)
                bu[uu] += lr * (err - reg * bu[uu])
                bi[ii] += lr * (err - reg * bi[ii])

                Pu = P[uu].copy()
                P[uu] += lr * (err * Q[ii] - reg * P[uu])
                Q[ii] += lr * (err * Pu   - reg * Q[ii])

            # epoch metrics
            train_rmse = self.rmse(train_df, user_col, item_col, rating_col)

            if val_df is not None:
                val_rmse = self.rmse(val_df, user_col, item_col, rating_col)
                if verbose:
                    print(f"Epoch {epoch:02d}: train RMSE={train_rmse:.4f} | val RMSE={val_rmse:.4f}")

                if self.config.early_stopping:
                    if val_rmse + self.config.tol < best_val:
                        best_val = val_rmse
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                        if bad_epochs >= self.config.patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch} (best val RMSE={best_val:.4f})")
                            break
            else:
                if verbose:
                    print(f"Epoch {epoch:02d}: train RMSE={train_rmse:.4f}")

        return self

    def predict_one(self, user, item) -> float:
        pred = self.mu

        uidx = self.user2idx.get(user, None)
        iidx = self.item2idx.get(item, None)

        if uidx is not None and self.bu is not None:
            pred += float(self.bu[uidx])
        if iidx is not None and self.bi is not None:
            pred += float(self.bi[iidx])
        if uidx is not None and iidx is not None and self.P is not None and self.Q is not None:
            pred += float(np.dot(self.P[uidx], self.Q[iidx]))

        # optional clipping
        if self.config.min_rating is not None:
            pred = max(self.config.min_rating, pred)
        if self.config.max_rating is not None:
            pred = min(self.config.max_rating, pred)

        return pred

    def predict_batch(self, users, items) -> np.ndarray:
        return np.array([
            self.predict_one(u, i) for u, i in zip(users, items)
            ])
    

    def rmse(self, df: pd.DataFrame, user_col="user_id", item_col="isbn", rating_col="rating") -> float:
        preds = self.predict_batch(df[user_col], df[item_col])
        y = df[rating_col].to_numpy(dtype=np.float64)
        return float(np.sqrt(np.mean((y - preds) ** 2)))
    
    def recommend(self, user_id, n, exclude_items):
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id: User identifier
            n: Number of recommendations to generate
            exclude_items: Set of items to exclude from recommendations
            
        Returns:
            List of recommended item IDs
        """
        # Ensure exclude_items is a set
        if exclude_items is None:
            exclude_items = set()
        elif not isinstance(exclude_items, set):
            exclude_items = set(exclude_items)
            
        # If user not in train: fall back to popular items (bias-based)
        uidx = self.user2idx.get(user_id)
        if uidx is None:
            # rank by item bias only (mu + bi) as a simple fallback
            scores = self.mu + self.bi.copy()  # Copy to avoid modifying self.bi
            
            # Filter excluded items
            for it in exclude_items:
                iidx = self.item2idx.get(it)
                if iidx is not None:
                    scores[iidx] = -np.inf
            
            # Get top-K
            k_eff = min(n, np.sum(np.isfinite(scores)))
            if k_eff == 0:
                return []
            top = np.argpartition(-scores, k_eff-1)[:k_eff]
            top = top[np.argsort(-scores[top])]
            items = [self.idx2item[i] for i in top]
            return items  # Fixed: return consistent type (list, not tuple)

        # score all items: mu + bu[u] + bi + P[u] dot Q^T
        bu = self.bu[uidx]
        user_vec = self.P[uidx]                      # shape (f,)
        scores = self.mu + bu + self.bi + self.Q @ user_vec   # (n_items,)

        # filter seen items by setting score to -inf
        # Make a copy to avoid modifying the computation
        scores = scores.copy()
        for it in exclude_items:
            iidx = self.item2idx.get(it)
            if iidx is not None:
                scores[iidx] = -np.inf

        # top-K indices efficiently
        k_eff = min(n, np.sum(np.isfinite(scores)))
        if k_eff == 0:
            return []
        top = np.argpartition(-scores, k_eff-1)[:k_eff]
        top = top[np.argsort(-scores[top])]

        items = [self.idx2item[i] for i in top]
        return items

