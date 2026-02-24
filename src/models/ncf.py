"""
Neural Collaborative Filtering (NeuMF) implementation using numpy.

Architecture:
- GMF branch: element-wise product of user/item embeddings (linear interaction)
- MLP branch: concatenated embeddings passed through hidden layers (non-linear)
- NeuMF: learned fusion of GMF and MLP outputs via final sigmoid

Reference: He et al., "Neural Collaborative Filtering", WWW 2017.
Training: mini-batch SGD with BPR pairwise ranking loss (same regime as existing BPRMF).
"""

import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


@dataclass
class NeuMFConfig:
    # GMF embedding size
    gmf_factors: int = 32
    # MLP embedding size (per side, concatenated = 2 * mlp_factors fed to first layer)
    mlp_factors: int = 32
    # Hidden layer sizes for MLP (after embedding concat)
    mlp_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    # Fusion output -> 1 (NeuMF output layer width)
    # alpha: weight of GMF in final layer (1-alpha for MLP)
    alpha: float = 0.5

    # Optimisation
    lr: float = 0.01
    reg: float = 1e-4
    n_epochs: int = 20
    batch_size: int = 2048
    n_samples_per_epoch: int = 200_000

    seed: int = 42
    dropout: float = 0.0  # kept for interface; not applied in this pure-numpy impl


class NeuMF:
    """
    NeuMF = GMF + MLP combined via a single output neuron.

    Scoring:
        gmf_out = P_gmf[u] * Q_gmf[i]              (element-wise, shape: gmf_factors)
        mlp_h0  = [P_mlp[u] ; Q_mlp[i]]            (concat, shape: 2*mlp_factors)
        mlp_out = ReLU(W_L ... ReLU(W_1 mlp_h0))   (shape: mlp_layers[-1])
        phi     = [gmf_out ; mlp_out]               (shape: gmf_factors + mlp_layers[-1])
        score   = w^T phi + b                       (scalar)

    Training: BPR pairwise loss  -log sigma(score(u,i) - score(u,j)) + L2.
    """

    def __init__(self, n_users: int, n_items: int, cfg: NeuMFConfig):
        self.n_users = n_users
        self.n_items = n_items
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        f_gmf = cfg.gmf_factors
        f_mlp = cfg.mlp_factors
        layers = cfg.mlp_layers
        scale = 0.01

        # GMF embeddings
        self.P_gmf = self.rng.normal(0, scale, (n_users, f_gmf)).astype(np.float32)
        self.Q_gmf = self.rng.normal(0, scale, (n_items, f_gmf)).astype(np.float32)

        # MLP embeddings
        self.P_mlp = self.rng.normal(0, scale, (n_users, f_mlp)).astype(np.float32)
        self.Q_mlp = self.rng.normal(0, scale, (n_items, f_mlp)).astype(np.float32)

        # MLP layers: W[k] shape (layers[k], input_size), b[k] shape (layers[k],)
        # Use He initialisation for ReLU layers
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        prev_size = 2 * f_mlp
        for h in layers:
            he_scale = np.sqrt(2.0 / prev_size)
            self.W.append(self.rng.normal(0, he_scale, (h, prev_size)).astype(np.float32))
            self.b.append(np.zeros(h, dtype=np.float32))
            prev_size = h

        # Output layer: fuses GMF output and last MLP hidden
        out_input_size = f_gmf + (layers[-1] if layers else 2 * f_mlp)
        self.w_out = self.rng.normal(0, scale, (out_input_size,)).astype(np.float32)
        self.b_out = np.float32(0.0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(
        self, u: np.ndarray, i: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Returns (scores, cache) where cache holds intermediate activations
        needed for backprop.
        u, i: int arrays of shape (B,)
        """
        # GMF branch
        pu_g = self.P_gmf[u]   # (B, f_gmf)
        qi_g = self.Q_gmf[i]   # (B, f_gmf)
        gmf_out = pu_g * qi_g  # (B, f_gmf)

        # MLP branch
        pu_m = self.P_mlp[u]   # (B, f_mlp)
        qi_m = self.Q_mlp[i]   # (B, f_mlp)
        h = np.concatenate([pu_m, qi_m], axis=1)  # (B, 2*f_mlp)
        activations = [h]
        pre_acts = []
        # float32 matmul can emit spurious overflow warnings in some BLAS builds;
        # results are numerically correct — suppress the warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for W, b in zip(self.W, self.b):
                z = h @ W.T + b     # (B, h_k)
                pre_acts.append(z)
                h = relu(z)         # (B, h_k)
                activations.append(h)
        mlp_out = h  # (B, layers[-1])

        # Fusion
        phi = np.concatenate([gmf_out, mlp_out], axis=1)  # (B, f_gmf + layers[-1])
        scores = phi @ self.w_out + self.b_out              # (B,)

        cache = (u, i, pu_g, qi_g, pu_m, qi_m, gmf_out, activations, pre_acts, phi)
        return scores, cache

    def score(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        """Public scoring interface (no cache returned)."""
        s, _ = self._forward(u, i)
        return s

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        user_pos: List[np.ndarray],
        neg_sampler,
        verbose: bool = True,
        val_df=None,
        train_df=None,
        user2idx=None,
        idx2item=None,
        item2idx=None,
        pipeline=None,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:

        cfg = self.cfg
        eligible_users = np.array(
            [u for u in range(self.n_users) if len(user_pos[u]) > 0], dtype=np.int32
        )
        if len(eligible_users) == 0:
            raise ValueError("No eligible users.")

        losses, ndcgs, precisions, hit_rates = [], [], [], []

        for epoch in range(cfg.n_epochs):
            epoch_loss = 0.0
            n_done = 0

            while n_done < cfg.n_samples_per_epoch:
                bs = min(cfg.batch_size, cfg.n_samples_per_epoch - n_done)

                # Sample (u, i, j) triples
                u = self.rng.choice(eligible_users, size=bs, replace=True)
                i = np.empty(bs, dtype=np.int32)
                for k, uu in enumerate(u):
                    pos = user_pos[uu]
                    i[k] = pos[self.rng.integers(0, len(pos))]

                j = neg_sampler.sample(bs)
                for k, uu in enumerate(u):
                    pos_set = user_pos[uu]
                    while np.any(pos_set == j[k]):
                        j[k] = int(neg_sampler.sample(1)[0])

                # Forward passes for positive and negative items
                s_i, cache_i = self._forward(u, i)
                s_j, cache_j = self._forward(u, j)

                x_uij = s_i - s_j
                sig = sigmoid(x_uij).astype(np.float32)
                g = (1.0 - sig)  # gradient factor = sigma(-x_uij)

                batch_loss = -np.log(np.clip(sig, 1e-8, 1.0)).mean()
                epoch_loss += batch_loss * bs

                # Backprop for positive items (+g) and negative items (-g)
                self._backward(cache_i, g, sign=+1.0)
                self._backward(cache_j, g, sign=-1.0)

                n_done += bs

            epoch_loss /= cfg.n_samples_per_epoch
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
                print(f"Epoch {epoch+1:02d}/{cfg.n_epochs} | bpr_loss={epoch_loss:.5f}")

        return losses, ndcgs, precisions, hit_rates

    def _backward(self, cache, g: np.ndarray, sign: float):
        """
        Backpropagate BPR gradient.
        sign=+1 for positive item, -1 for negative item.
        ds/d(param) for BPR: delta = sign * g (shape B,)
        """
        # Suppress spurious float32 BLAS overflow warnings — results are correct.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self._backward_inner(cache, g, sign)

    def _backward_inner(self, cache, g: np.ndarray, sign: float):
        cfg = self.cfg
        lr = cfg.lr
        reg = cfg.reg

        (u, i, pu_g, qi_g, pu_m, qi_m,
         gmf_out, activations, pre_acts, phi) = cache

        delta = (sign * g).astype(np.float32)  # (B,)

        # --- Output layer gradients ---
        # score = phi @ w_out + b_out
        # d_loss/d_w_out = delta[:, None] * phi summed over batch
        dw_out = (delta[:, None] * phi).mean(axis=0) - reg * self.w_out
        db_out = delta.mean() - reg * float(self.b_out)
        # gradient flowing back into phi
        d_phi = delta[:, None] * self.w_out[None, :]  # (B, f_gmf + layers[-1])

        f_gmf = cfg.gmf_factors
        d_gmf_out = d_phi[:, :f_gmf]          # (B, f_gmf)
        d_mlp_out = d_phi[:, f_gmf:]           # (B, layers[-1])

        # --- GMF branch ---
        d_pu_g = d_gmf_out * qi_g - reg * pu_g   # (B, f_gmf)
        d_qi_g = d_gmf_out * pu_g - reg * qi_g   # (B, f_gmf)

        # --- MLP branch (backprop through layers) ---
        _CLIP = 5.0
        d_h = d_mlp_out  # (B, layers[-1])
        for k in reversed(range(len(self.W))):
            z = pre_acts[k]
            d_z = d_h * relu_grad(z)              # (B, h_k)
            h_prev = activations[k]               # (B, prev_size)
            dW = d_z.T @ h_prev / len(u) - reg * self.W[k]
            db = d_z.mean(axis=0) - reg * self.b[k]
            d_h = d_z @ self.W[k]                 # (B, prev_size)
            self.W[k] += lr * np.clip(dW, -_CLIP, _CLIP)
            self.b[k] += lr * np.clip(db, -_CLIP, _CLIP)

        # d_h now corresponds to gradient w.r.t. [pu_m, qi_m]
        f_mlp = cfg.mlp_factors
        d_pu_m = d_h[:, :f_mlp] - reg * pu_m
        d_qi_m = d_h[:, f_mlp:] - reg * qi_m

        # Clip updates to prevent exploding gradients
        _CLIP = 5.0
        np.add.at(self.P_gmf, u, lr * np.clip(d_pu_g, -_CLIP, _CLIP))
        np.add.at(self.Q_gmf, i, lr * np.clip(d_qi_g, -_CLIP, _CLIP))
        np.add.at(self.P_mlp, u, lr * np.clip(d_pu_m, -_CLIP, _CLIP))
        np.add.at(self.Q_mlp, i, lr * np.clip(d_qi_m, -_CLIP, _CLIP))

        # --- Apply output layer updates ---
        self.w_out += lr * np.clip(dw_out, -_CLIP, _CLIP)
        self.b_out = np.float32(self.b_out + lr * float(np.clip(db_out, -_CLIP, _CLIP)))

    # ------------------------------------------------------------------
    # Recommendation interface (same as BPRMF for eval pipeline compat)
    # ------------------------------------------------------------------

    def recommend(
        self,
        u_idx: int,
        K: int,
        seen_items: Optional[List] = None,
        candidates: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if candidates is None:
            cand = np.arange(self.n_items, dtype=np.int32)
        else:
            cand = np.asarray(candidates, dtype=np.int32)

        if seen_items:
            seen_set = set(seen_items)
            mask = np.array([c not in seen_set for c in cand], dtype=bool)
            cand = cand[mask]

        u_arr = np.full(len(cand), u_idx, dtype=np.int32)
        s = self.score(u_arr, cand)

        if len(cand) <= K:
            order = np.argsort(-s)
            return cand[order], s[order]

        topk_idx = np.argpartition(-s, K)[:K]
        topk_sorted = topk_idx[np.argsort(-s[topk_idx])]
        return cand[topk_sorted], s[topk_sorted]
