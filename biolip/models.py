"""BioLip model definitions — aligned with arXiv:2604.16808v4.

Paper architecture (§III-E):
    Three-branch network (302K parameters):
    - Temporal branch:  TCN(256→128→128→64) + multi-scale pooling → h_t ∈ R^128
    - Region branch:    MLP(8→32→32) → h_r ∈ R^32
    - Statistical branch: MLP(256→128→64, BN) → h_s ∈ R^64
    - Fusion: concat(h_t, h_r, h_s) ∈ R^224 → Head(224→128→64→1, GELU, Dropout 0.2)

    Training (§III-F):
    - BCEWithLogitsLoss with positive class weighting
    - AdamW lr=3e-4, weight_decay=1e-4
    - Cosine annealing, 60 epochs, eta_min=1e-5
    - Early stopping patience=30 on validation video-level AUC
    - Seeds: 22, 100, 123

The GBM branch (HistGradientBoostingClassifier on tabular window features)
is NOT described in the paper; it is a custom addition that remains for
experimental comparison.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

try:
    from .features import CH_NAMES
except (ImportError, ValueError):
    from features import CH_NAMES


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def clip_mean(keys, z):
    return (
        pd.DataFrame({"key": np.asarray(keys), "z": np.asarray(z)})
        .groupby("key")
        .z.mean()
    )


def clip_auc(meta, z):
    s = clip_mean(meta.key.values, z)
    y = meta.drop_duplicates("key").set_index("key").label.loc[s.index]
    return roc_auc_score(y, s) if y.nunique() > 1 else np.nan


# ---------------------------------------------------------------------------
# GBM branch (custom, not in paper — kept for experimental comparison)
# ---------------------------------------------------------------------------

def train_gbm(Fw, meta, cfg):
    """Train a HistGradientBoostingClassifier with LOGO cross-validation.

    NOTE: This branch is NOT part of the published BioLip method.
    """
    is_tr = (meta.split == "train").values
    tr_gens = sorted(meta[is_tr].generator.unique())
    configs = [
        dict(max_depth=3, learning_rate=0.05, max_iter=200,
             l2_regularization=1.0, min_samples_leaf=30),
        dict(max_depth=4, learning_rate=0.05, max_iter=300,
             l2_regularization=1.0, min_samples_leaf=30),
        dict(max_depth=6, learning_rate=0.03, max_iter=400,
             l2_regularization=3.0, min_samples_leaf=40),
    ]
    best = None
    for c in configs:
        oof = np.full(len(meta), np.nan)
        for g in tr_gens:
            a = is_tr & (meta.generator != g).values
            b = is_tr & (meta.generator == g).values
            oof[b] = (
                HistGradientBoostingClassifier(random_state=cfg.seed, **c)
                .fit(Fw[a], meta.label.values[a])
                .decision_function(Fw[b])
            )
        aucs = [
            clip_auc(
                meta[is_tr & (meta.generator == g).values],
                oof[is_tr & (meta.generator == g).values],
            )
            for g in tr_gens
        ]
        print(c, "-> LOGO clip AUC", np.round(aucs, 3), "mean", np.nanmean(aucs))
        if best is None or np.nanmean(aucs) > best[0]:
            best = (np.nanmean(aucs), c, oof)
    gbm_cfg, oof = best[1], best[2]
    gbm = HistGradientBoostingClassifier(random_state=cfg.seed, **gbm_cfg).fit(
        Fw[is_tr], meta.label.values[is_tr]
    )
    return gbm, gbm.decision_function(Fw), oof, gbm_cfg


# ---------------------------------------------------------------------------
# Paper three-branch architecture (§III-E)
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """1D-Conv block: Conv1d + BatchNorm + GELU."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2)
        self.bn   = nn.BatchNorm1d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.bn(self.conv(x)))


class _TemporalBranch(nn.Module):
    """Temporal branch: Conv1D(256→128→128→64) + multi-scale pooling → R^128.

    Input:  Xt transposed → (B, 256, T)
    Output: h_t ∈ R^128
    """

    def __init__(self, t_len: int = 25):
        super().__init__()
        self.t_len = t_len
        self.convs = nn.Sequential(
            _ConvBlock(256, 128),
            _ConvBlock(128, 128),
            _ConvBlock(128, 64),
        )
        # 384 = 64 (global avg) + 4*64 (local avg, 4 segments) + 64 (global max)
        self.proj = nn.Linear(384, 128)

    def forward(self, Xt: torch.Tensor) -> torch.Tensor:
        # Xt: (B, T, 256)  ->  transpose to (B, 256, T) for Conv1d
        x = Xt.transpose(1, 2)           # (B, 256, T)
        x = self.convs(x)                # (B, 64, T)

        # Multi-scale pooling
        g_avg = x.mean(dim=2)            # (B, 64)
        g_max = x.max(dim=2).values      # (B, 64)

        # Local average over 4 equal temporal segments → (B, 4*64)
        T = x.shape[2]
        seg = T // 4
        segs = [x[:, :, i * seg: (i + 1) * seg].mean(dim=2) for i in range(4)]
        l_avg = torch.cat(segs, dim=1)   # (B, 256)

        pooled = torch.cat([g_avg, l_avg, g_max], dim=1)  # (B, 384)
        return self.proj(pooled)          # (B, 128)


class _RegionBranch(nn.Module):
    """Region branch: MLP(8→32→32) → h_r ∈ R^32."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, 32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU(),
        )

    def forward(self, xr: torch.Tensor) -> torch.Tensor:
        return self.mlp(xr)


class _StatsBranch(nn.Module):
    """Statistical branch: MLP(256→128→64, BN) → h_s ∈ R^64."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.bn1(self.fc1(xs)))
        return F.gelu(self.fc2(x))


class BioLipNet(nn.Module):
    """Full three-branch BioLip network (§III-E, 302K parameters).

    Inputs
    ------
    Xt : (B, T, 256)   temporal sequence
    xr : (B, 8)        region ratio vector
    xs : (B, 256)      statistical vector

    Output
    ------
    logit : (B,)  raw score (apply sigmoid for probability)
    """

    def __init__(self, t_len: int = 25, dropout: float = 0.2):
        super().__init__()
        self.temporal = _TemporalBranch(t_len)  # → 128
        self.region   = _RegionBranch()          # → 32
        self.stats    = _StatsBranch()           # → 64
        # Fusion head: 224 → 128 → 64 → 1
        self.head = nn.Sequential(
            nn.Linear(224, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, Xt: torch.Tensor, xr: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        ht = self.temporal(Xt)               # (B, 128)
        hr = self.region(xr)                 # (B, 32)
        hs = self.stats(xs)                  # (B, 64)
        h  = torch.cat([ht, hr, hs], dim=1)  # (B, 224)
        return self.head(h).squeeze(1)       # (B,)


# ---------------------------------------------------------------------------
# Training  (§III-F)
# ---------------------------------------------------------------------------

def train_biolip(Xt_all, Xr_all, Xs_all, meta, cfg):
    """Train the three-branch BioLipNet following the paper protocol (§III-F).

    Parameters
    ----------
    Xt_all : np.ndarray (N, T, 256)   temporal sequences
    Xr_all : np.ndarray (N, 8)        region ratio vectors
    Xs_all : np.ndarray (N, 256)      statistical vectors
    meta   : pd.DataFrame             window-level metadata with 'split', 'label', etc.
    cfg    : Config

    Returns
    -------
    models : list[BioLipNet]   one per seed
    eps    : list[int]         best epoch per seed
    z      : np.ndarray        ensemble mean window-level logit
    frame  : None              (not applicable for BioLipNet)
    oof    : np.ndarray        OOF logits (LOGO on train generators)
    mu_Xs, sd_Xs : normalisation statistics for xs (computed on train)
    """
    accel = Accelerator()
    dev = accel.device

    is_tr = (meta.split == "train").values
    is_va = (meta.split == "val").values

    # Normalise xs by train statistics (same as paper's train-set standardisation)
    mu_Xs = Xs_all[is_tr].mean(0)
    sd_Xs = Xs_all[is_tr].std(0) + 1e-6
    Xs_n  = ((Xs_all - mu_Xs) / sd_Xs).astype(np.float32)

    # Normalise Xt likewise
    mu_Xt = Xt_all[is_tr].reshape(-1, 256).mean(0)
    sd_Xt = Xt_all[is_tr].reshape(-1, 256).std(0) + 1e-6
    Xt_n  = ((Xt_all - mu_Xt) / sd_Xt).astype(np.float32)

    y_all = torch.tensor(meta.label.values, dtype=torch.float32)

    # Positive class weight = n_neg / n_pos (BCEWithLogitsLoss pos_weight)
    n_pos = int((y_all[is_tr] == 1).sum())
    n_neg = int((y_all[is_tr] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(dev)

    @torch.no_grad()
    def predict(model, idx_mask):
        model.eval()
        logits = []
        idxs = np.flatnonzero(idx_mask)
        for i in range(0, len(idxs), 256):
            b = idxs[i: i + 256]
            xt = torch.from_numpy(Xt_n[b]).to(dev)
            xr = torch.from_numpy(Xr_all[b]).to(dev)
            xs = torch.from_numpy(Xs_n[b]).to(dev)
            logits.append(accel.unwrap_model(model)(xt, xr, xs).cpu())
        return torch.cat(logits).numpy()

    def clip_auc_local(mask, z):
        return clip_auc(meta[mask], z)

    def train_one(mask, seed, epochs, val_mask=None, patience=None):
        if patience is None:
            patience = getattr(cfg, "cnn_patience", 30)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = BioLipNet(t_len=cfg.win)
        n_steps = epochs * int(math.ceil(mask.sum() / 128))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=1e-5
        )
        model, opt, sched = accel.prepare(model, opt, sched)
        idx = np.flatnonzero(mask)
        best_auc, best_ep, bad, best_state = -1.0, epochs, 0, None

        for ep in range(epochs):
            model.train()
            np.random.shuffle(idx)
            for i in range(0, len(idx), 128):
                b = idx[i: i + 128]
                xt = torch.from_numpy(Xt_n[b]).to(dev)
                xr = torch.from_numpy(Xr_all[b]).to(dev)
                xs = torch.from_numpy(Xs_n[b]).to(dev)
                yb = y_all[b].to(dev)
                logit = model(xt, xr, xs)
                loss  = F.binary_cross_entropy_with_logits(
                    logit, yb, pos_weight=pos_weight
                )
                opt.zero_grad()
                accel.backward(loss)
                opt.step()
                sched.step()

            if val_mask is not None:
                z_val = predict(model, val_mask)
                a = clip_auc_local(val_mask, z_val)
                if a > best_auc + 1e-4:
                    best_auc, best_ep, bad = a, ep + 1, 0
                    best_state = {
                        k: v.clone()
                        for k, v in accel.unwrap_model(model).state_dict().items()
                    }
                else:
                    bad += 1
                    if bad >= patience:
                        break

        if best_state is not None:
            accel.unwrap_model(model).load_state_dict(best_state)
        return model, best_ep, best_auc

    models, eps = [], []
    for seed in cfg.cnn_seeds:
        m, ep, a = train_one(is_tr, seed, cfg.cnn_epochs, val_mask=is_va)
        models.append(m)
        eps.append(ep)
        if accel.is_main_process:
            print(f"seed {seed}: best epoch {ep}, val clip AUC {a:.3f}")

    # OOF with median epoch for LOGO
    fixed_ep = int(np.median(eps))
    tr_gens  = sorted(meta[is_tr].generator.unique())
    oof = np.full(len(meta), np.nan)
    for g in tr_gens:
        a_mask = is_tr & (meta.generator != g).values
        b_mask = is_tr & (meta.generator == g).values
        m_oof, _, _ = train_one(a_mask, 0, fixed_ep)
        oof[b_mask] = predict(m_oof, b_mask)
        if accel.is_main_process:
            print("LOGO held-out", g, "clip AUC %.3f" % clip_auc(meta[b_mask], oof[b_mask]))

    # Ensemble: mean window logit across seeds
    all_z = []
    for m in models:
        z = np.full(len(meta), np.nan)
        z_pred = predict(m, np.ones(len(meta), dtype=bool))
        z[:] = z_pred
        all_z.append(z)
    z_ens = np.nanmean(all_z, axis=0)

    return models, eps, z_ens, None, oof, mu_Xs, sd_Xs, mu_Xt, sd_Xt


# ---------------------------------------------------------------------------
# Legacy single-branch TCN (kept for backward compatibility / ablation)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Dilated residual conv block used in the legacy single-branch TCN."""

    def __init__(self, h, d, p):
        super().__init__()
        self.c  = nn.Conv1d(h, h, 3, padding=d, dilation=d)
        self.n  = nn.GroupNorm(8, h)
        self.do = nn.Dropout(p)

    def forward(self, x):
        return x + self.do(F.gelu(self.n(self.c(x))))


class TCN(nn.Module):
    """Legacy single-branch TCN (NOT the paper architecture).

    Kept for ablation / backward compatibility with existing checkpoints.
    """

    def __init__(self, c_in, h=64, p=0.25):
        super().__init__()
        self.inp    = nn.Conv1d(c_in, h, 1)
        self.blocks = nn.Sequential(*[Block(h, d, p) for d in (1, 2, 4, 1)])
        self.head   = nn.Conv1d(h, 1, 1)

    def forward(self, x):
        return self.head(self.blocks(F.gelu(self.inp(x.transpose(1, 2))))).squeeze(1)


def train_cnn_models(Xw, meta, cfg):
    """Train the legacy single-branch TCN (NOT the paper BioLipNet).

    This function is retained for backward compatibility.  For the
    paper-aligned training, use `train_biolip` instead.
    """
    accel = Accelerator()
    dev   = accel.device
    K_TOP = int(math.ceil(cfg.topk * cfg.win))
    is_tr = (meta.split == "train").values
    is_va = (meta.split == "val").values
    mu = Xw[is_tr].reshape(-1, Xw.shape[-1]).mean(0)
    sd = Xw[is_tr].reshape(-1, Xw.shape[-1]).std(0) + 1e-6
    Xn = ((Xw - mu) / sd).astype(np.float32)

    def win_logit(fl):
        return fl.topk(K_TOP, dim=1).values.mean(1)

    @torch.no_grad()
    def predict(model, X):
        model.eval()
        fl = torch.cat(
            [
                accel.unwrap_model(model)(
                    torch.from_numpy(X[i: i + 512]).to(dev)
                ).cpu()
                for i in range(0, len(X), 512)
            ]
        )
        return win_logit(fl).numpy(), fl.numpy()

    def clip_auc_local(mask, z):
        return clip_auc(meta[mask], z)

    def train_one(mask, seed, epochs, val_mask=None, patience=10):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = TCN(Xn.shape[-1])
        opt   = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
        idx   = np.flatnonzero(mask)
        y     = torch.tensor(meta.label.values, dtype=torch.float32)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3e-3,
            total_steps=epochs * int(math.ceil(len(idx) / 128))
        )
        model, opt, sched = accel.prepare(model, opt, sched)
        best, state, best_ep, bad = -1, None, epochs, 0
        for ep in range(epochs):
            model.train()
            np.random.shuffle(idx)
            for i in range(0, len(idx), 128):
                b  = idx[i: i + 128]
                xb = torch.from_numpy(Xn[b]).to(dev)
                yb = y[b].to(dev) * 0.9 + 0.05
                xb = xb + 0.05 * torch.randn_like(xb)
                xb = xb * (torch.rand(xb.shape[0], 1, xb.shape[2], device=dev) > 0.1).float()
                fl  = model(xb)
                loss = F.binary_cross_entropy_with_logits(
                    win_logit(fl), yb
                ) + cfg.aux_w * F.binary_cross_entropy_with_logits(fl.mean(1), yb)
                opt.zero_grad()
                accel.backward(loss)
                opt.step()
                sched.step()
            if val_mask is not None:
                a = clip_auc_local(val_mask, predict(model, Xn[val_mask])[0])
                if a > best + 1e-4:
                    best, best_ep, bad = a, ep + 1, 0
                    state = {k: v.clone() for k, v in accel.unwrap_model(model).state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if state is not None:
            accel.unwrap_model(model).load_state_dict(state)
        return model, best_ep, best

    models, eps = [], []
    for seed in cfg.cnn_seeds:
        m, ep, a = train_one(is_tr, seed, cfg.cnn_epochs, val_mask=is_va)
        models.append(m)
        eps.append(ep)
        if accel.is_main_process:
            print(f"seed {seed}: best epoch {ep}, val clip AUC {a:.3f}")

    fixed_ep = int(np.median(eps))
    tr_gens  = sorted(meta[is_tr].generator.unique())
    oof = np.full(len(meta), np.nan)
    for g in tr_gens:
        a = is_tr & (meta.generator != g).values
        b = is_tr & (meta.generator == g).values
        oof[b] = predict(train_one(a, 0, fixed_ep)[0], Xn[b])[0]
        if accel.is_main_process:
            print("LOGO held-out", g, "clip AUC %.3f" % clip_auc(meta[b], oof[b]))

    outs  = [predict(m, Xn) for m in models]
    z     = np.mean([o[0] for o in outs], 0)
    frame = np.mean([o[1] for o in outs], 0)
    return models, eps, z, frame, oof, mu, sd
