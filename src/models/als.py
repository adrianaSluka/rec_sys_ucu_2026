from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

@dataclass
class ALSConfig:
    n_factors: int = 50
    n_iters: int = 15
    reg: float = 0.1
    seed: int = 42
    # optional mean-centering helps
    center_by_global_mean: bool = True


class ALS:
    def __init__(self, cfg: ALSConfig):
        self.cfg = cfg
        self.user2idx: Dict = {}
        self.item2idx: Dict = {}
        self.idx2item: Dict[int, str] = {}
        self.P: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.mu: float = 0.0

        # cached interactions
        self.user_items: List[np.ndarray] = []
        self.user_ratings: List[np.ndarray] = []
        self.item_users: List[np.ndarray] = []
        self.item_ratings: List[np.ndarray] = []

    def _build_mappings(self, df: pd.DataFrame, user_col: str, item_col: str):
        users = df[user_col].unique()
        items = df[item_col].unique()
        self.user2idx = {u: k for k, u in enumerate(users)}
        self.item2idx = {i: k for k, i in enumerate(items)}
        self.idx2item = {k: i for i, k in self.item2idx.items()}

    def _build_groups(self, df: pd.DataFrame, user_col: str, item_col: str, rating_col: str):
        n_users = len(self.user2idx)
        n_items = len(self.item2idx)

        # gather per-user and per-item lists
        user_items = [[] for _ in range(n_users)]
        user_ratings = [[] for _ in range(n_users)]
        item_users = [[] for _ in range(n_items)]
        item_ratings = [[] for _ in range(n_items)]

        for u, i, r in zip(df[user_col], df[item_col], df[rating_col]):
            uidx = self.user2idx[u]
            iidx = self.item2idx[i]
            user_items[uidx].append(iidx)
            user_ratings[uidx].append(float(r))
            item_users[iidx].append(uidx)
            item_ratings[iidx].append(float(r))

        self.user_items = [np.array(x, dtype=np.int32) for x in user_items]
        self.user_ratings = [np.array(x, dtype=np.float64) for x in user_ratings]
        self.item_users = [np.array(x, dtype=np.int32) for x in item_users]
        self.item_ratings = [np.array(x, dtype=np.float64) for x in item_ratings]

    def fit(self, train_df: pd.DataFrame, user_col="user_id", item_col="isbn", rating_col="rating", verbose=True):
        self._build_mappings(train_df, user_col, item_col)

        # global mean centering (recommended for explicit ratings)
        if self.cfg.center_by_global_mean:
            self.mu = float(train_df[rating_col].mean())
        else:
            self.mu = 0.0

        df = train_df.copy()
        df["_r"] = df[rating_col].astype(float) - self.mu

        self._build_groups(df, user_col, item_col, "_r")

        rng = np.random.default_rng(self.cfg.seed)
        n_users = len(self.user2idx)
        n_items = len(self.item2idx)
        k = self.cfg.n_factors

        self.P = 0.01 * rng.standard_normal((n_users, k))
        self.Q = 0.01 * rng.standard_normal((n_items, k))

        regI = self.cfg.reg * np.eye(k, dtype=np.float64)

        for it in range(1, self.cfg.n_iters + 1):
            # --- update P (users) ---
            for u in range(n_users):
                items = self.user_items[u]
                if items.size == 0:
                    continue
                Q_u = self.Q[items]                   # (m, k)
                r_u = self.user_ratings[u]            # (m,)
                A = Q_u.T @ Q_u + regI                # (k, k)
                b = Q_u.T @ r_u                       # (k,)
                self.P[u] = np.linalg.solve(A, b)

            # --- update Q (items) ---
            for i in range(n_items):
                users = self.item_users[i]
                if users.size == 0:
                    continue
                P_i = self.P[users]                   # (m, k)
                r_i = self.item_ratings[i]            # (m,)
                A = P_i.T @ P_i + regI
                b = P_i.T @ r_i
                self.Q[i] = np.linalg.solve(A, b)

            if verbose:
                print(f"ALS iter {it:02d} done")

        return self

    def predict_one(self, user, item) -> float:
        pred = self.mu
        uidx = self.user2idx.get(user)
        iidx = self.item2idx.get(item)
        if uidx is None or iidx is None or self.P is None or self.Q is None:
            return pred
        pred += float(self.P[uidx] @ self.Q[iidx])
        return pred

    def predict_batch(self, users, items) -> np.ndarray:
        return np.array([self.predict_one(u, i) for u, i in zip(users, items)], dtype=np.float64)
