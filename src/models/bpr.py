import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_id_maps(df: pd.DataFrame, user_col="user_id", item_col="isbn") -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
    """Map raw ids -> contiguous indices [0..n-1]."""
    users = df[user_col].unique()
    items = df[item_col].unique()
    user2idx = {u: k for k, u in enumerate(users)}
    item2idx = {i: k for k, i in enumerate(items)}
    idx2user = users
    idx2item = items
    return user2idx, item2idx, idx2user, idx2item


def build_user_positives(
    df: pd.DataFrame,
    user2idx: Dict,
    item2idx: Dict,
    user_col="user_id",
    item_col="isbn",
    rating_col="rating",
    min_rating_pos: int = 1,
) -> List[np.ndarray]:
    """
    For each user index, store an array of positive item indices.
    Positives: rating >= min_rating_pos (default: >0, i.e., explicit positives in Book-Crossing).
    """
    n_users = len(user2idx)
    pos_lists: List[List[int]] = [[] for _ in range(n_users)]
    sub = df[df[rating_col] >= min_rating_pos][[user_col, item_col]].drop_duplicates()

    for u, i in sub.itertuples(index=False):
        if u in user2idx and i in item2idx:
            pos_lists[user2idx[u]].append(item2idx[i])

    return [np.asarray(lst, dtype=np.int32) for lst in pos_lists]


def build_seen_sets(
    df: pd.DataFrame,
    user2idx: Dict,
    item2idx: Dict,
    user_col="userId",
    item_col="itemId",
) -> List[set]:
    """Items seen in train (for filtering during recommend)."""
    n_users = len(user2idx)
    seen = [set() for _ in range(n_users)]
    sub = df[[user_col, item_col]].drop_duplicates()
    for u, i in sub.itertuples(index=False):
        if u in user2idx and i in item2idx:
            seen[user2idx[u]].add(item2idx[i])
    return seen

class NegativeSampler:
    def sample(self, size: int) -> np.ndarray:
        raise NotImplementedError


class UniformNegativeSampler(NegativeSampler):
    def __init__(self, n_items: int, rng: np.random.Generator):
        self.n_items = n_items
        self.rng = rng

    def sample(self, size: int) -> np.ndarray:
        return self.rng.integers(0, self.n_items, size=size, dtype=np.int32)


class PopularityNegativeSampler(NegativeSampler):
    def __init__(self, item_pop_counts: np.ndarray, rng: np.random.Generator, alpha: float = 1.0):
        """
        item_pop_counts: shape (n_items,)
        alpha: >1 emphasizes head items more; <1 flattens.
        """
        self.rng = rng
        p = np.asarray(item_pop_counts, dtype=np.float64) ** alpha
        p = p / p.sum()
        self.p = p

    def sample(self, size: int) -> np.ndarray:
        return self.rng.choice(len(self.p), size=size, replace=True, p=self.p).astype(np.int32)


class MixedNegativeSampler(NegativeSampler):
    def __init__(self, sampler_a: NegativeSampler, sampler_b: NegativeSampler, p_a: float, rng: np.random.Generator):
        self.a = sampler_a
        self.b = sampler_b
        self.p_a = float(p_a)
        self.rng = rng

    def sample(self, size: int) -> np.ndarray:
        mask = self.rng.random(size) < self.p_a
        out = np.empty(size, dtype=np.int32)
        n_a = int(mask.sum())
        out[mask] = self.a.sample(n_a)
        out[~mask] = self.b.sample(size - n_a)
        return out


@dataclass
class BPRConfig:
    n_factors: int = 64
    lr: float = 0.05
    reg: float = 1e-4           # L2 on embeddings (and item bias if used)
    n_epochs: int = 20
    batch_size: int = 2048
    n_samples_per_epoch: int = 200_000  # number of (u,i,j) triples per epoch
    seed: int = 42
    use_item_bias: bool = True


class BPRMF:
    """
    BPR-OPT with MF scoring: s(u,i) = p_u^T q_i + b_i
    Optimizes: -log sigma(s(u,i) - s(u,j)) + L2 regularization
    """

    def __init__(self, n_users: int, n_items: int, cfg: BPRConfig):
        self.n_users = n_users
        self.n_items = n_items
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # init embeddings
        scale = 0.1
        self.P = (self.rng.normal(0, scale, size=(n_users, cfg.n_factors))).astype(np.float32)
        self.Q = (self.rng.normal(0, scale, size=(n_items, cfg.n_factors))).astype(np.float32)
        self.b = np.zeros(n_items, dtype=np.float32) if cfg.use_item_bias else None

    def score(self, u_idx: np.ndarray, i_idx: np.ndarray) -> np.ndarray:
        s = np.sum(self.P[u_idx] * self.Q[i_idx], axis=1)
        if self.b is not None:
            s = s + self.b[i_idx]
        return s

    def fit(
        self,
        user_pos: List[np.ndarray],
        neg_sampler: NegativeSampler,
        verbose: bool = True,
    ) -> List[float]:
        """
        user_pos: list length n_users, each an array of positive item indices
        neg_sampler: defines how we sample j from items
        """
        cfg = self.cfg
        losses = []

        # users with at least 1 positive
        eligible_users = np.array([u for u in range(self.n_users) if len(user_pos[u]) > 0], dtype=np.int32)
        if len(eligible_users) == 0:
            raise ValueError("No eligible users with positives. Check your min_rating_pos or data filtering.")

        for epoch in range(cfg.n_epochs):
            epoch_loss = 0.0
            n_done = 0

            # iterate in mini-batches over randomly sampled triples
            n_total = cfg.n_samples_per_epoch
            while n_done < n_total:
                bs = min(cfg.batch_size, n_total - n_done)

                # sample users
                u = self.rng.choice(eligible_users, size=bs, replace=True)

                # sample a positive i for each user u
                i = np.empty(bs, dtype=np.int32)
                for k, uu in enumerate(u):
                    pos_items = user_pos[uu]
                    i[k] = pos_items[self.rng.integers(0, len(pos_items))]

                # sample negatives j (and resample if accidentally positive for that user)
                j = neg_sampler.sample(bs)

                # reject positives (simple loop; ok for typical data; optimize if needed)
                for k, uu in enumerate(u):
                    pos_set = user_pos[uu]
                    # fast containment if we convert to set per user; but array is ok if small.
                    # We'll do a small while with np.any on array.
                    while np.any(pos_set == j[k]):
                        j[k] = int(neg_sampler.sample(1)[0])

                # scores
                x_ui = self.score(u, i)
                x_uj = self.score(u, j)
                x_uij = x_ui - x_uj

                # BPR loss = -log sigmoid(x_uij)
                # gradient factor = sigmoid(-x_uij) = 1 - sigmoid(x_uij)
                s = sigmoid(x_uij).astype(np.float32)
                g = (1.0 - s)  # = sigmoid(-x_uij)

                # Fetch embeddings
                Pu = self.P[u]          # (bs, f)
                Qi = self.Q[i]
                Qj = self.Q[j]

                # Gradients (vectorized)
                # d/dPu:  g[:,None]*(Qi - Qj) - reg*Pu
                # d/dQi:  g[:,None]*Pu       - reg*Qi
                # d/dQj: -g[:,None]*Pu       - reg*Qj
                reg = cfg.reg
                lr = cfg.lr

                dPu = (g[:, None] * (Qi - Qj)) - reg * Pu
                dQi = (g[:, None] * Pu) - reg * Qi
                dQj = (-g[:, None] * Pu) - reg * Qj

                # Apply updates (SGD)
                # Note: repeated indices exist; numpy fancy assignment won't accumulate.
                # We do per-row updates via np.add.at for correctness.
                np.add.at(self.P, u, lr * dPu)
                np.add.at(self.Q, i, lr * dQi)
                np.add.at(self.Q, j, lr * dQj)

                if self.b is not None:
                    # d/db_i:  g - reg*b_i ; d/db_j: -g - reg*b_j
                    db_i = g - reg * self.b[i]
                    db_j = -g - reg * self.b[j]
                    np.add.at(self.b, i, lr * db_i)
                    np.add.at(self.b, j, lr * db_j)

                # batch loss value
                batch_loss = -np.log(np.clip(s, 1e-8, 1.0)).mean()
                epoch_loss += batch_loss * bs
                n_done += bs

            epoch_loss /= n_total
            losses.append(epoch_loss)
            if verbose:
                print(f"Epoch {epoch+1:02d}/{cfg.n_epochs} | train_bpr_loss={epoch_loss:.5f}")

        return losses

    def recommend(
        self,
        u_idx: int,
        K: int,
        seen_items: Optional[set] = None,
        candidates: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (item_indices, scores) for top-K items.
        If candidates is None: scores all items (can be slow for huge catalogs).
        Filters seen_items if provided.
        """
        if candidates is None:
            cand = np.arange(self.n_items, dtype=np.int32)
        else:
            cand = candidates.astype(np.int32)

        if seen_items:
            mask = np.array([i not in seen_items for i in cand], dtype=bool)
            cand = cand[mask]

        # score
        u_arr = np.full(len(cand), u_idx, dtype=np.int32)
        s = self.score(u_arr, cand)

        if len(cand) <= K:
            order = np.argsort(-s)
            return cand[order], s[order]

        # partial top-k
        topk_idx = np.argpartition(-s, K)[:K]
        topk_sorted = topk_idx[np.argsort(-s[topk_idx])]
        return cand[topk_sorted], s[topk_sorted]










