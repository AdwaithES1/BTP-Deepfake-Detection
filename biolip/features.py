"""Feature extraction for BioLip — aligned with arXiv:2604.16808v4.

Key design decisions (§III-C, §III-D):
  - Only the y-coordinate is used after mouth-center / inter-commissure
    scale normalisation (Eq. 1).
  - Statistical features xs ∈ R^{64×4} = R^256 are computed as the
    *standard deviation* over the T-frame window (not log-mean) —
    Eqs. 2–5.
  - Temporal features Xt ∈ R^{T×256} are built from the raw y values
    concatenated with their first three *finite differences* (not
    Savitzky-Golay derivatives) — Eq. 6.
  - The 8-dimensional anatomical ratio vector xr ∈ R^8 (Eq. 7) is
    computed from mean displacement std per region.
  - Window size = 25 frames, stride = 5 (§III-D: "1.0 s at 25 fps").

The Procrustes rigid-body alignment in build_channels() is preserved for
head-pose removal; the normalised y-coordinates are then extracted from
the aligned trajectory, consistent with Eq. 1.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .landmark_defs import PERIORAL, ANCHORS
except (ImportError, ValueError):
    from landmark_defs import PERIORAL, ANCHORS


# ---------------------------------------------------------------------------
# Landmark region slices (indices into the 64 perioral landmark array)
# ---------------------------------------------------------------------------
REG_IDX = {
    "lo_inner": slice(9, 18),   # 9 lower-lip inner contour points
    "up_inner": slice(18, 29),  # portion of upper lip inner
    "lo_outer": slice(0, 9),    # 9 lower-lip outer contour points
    "up_outer": slice(29, 40),  # upper lip outer
    "jaw":      slice(40, 55),  # jaw line
    "mid":      slice(55, 64),  # perioral mid-line region
}

# Four kinematic orders: displacement, velocity, acceleration, jerk
_KIN_ORDERS = ["disp", "vel", "acc", "jerk"]

# 256-dim statistical feature names (64 landmarks × 4 kinematic orders)
CH_NAMES: list = (
    [f"{r}_{k}" for r in REG_IDX for k in _KIN_ORDERS]
    + ["g_inner_h", "g_outer_h", "g_inner_w", "g_outer_w", "g_jaw_open"]
    + ["align_resid"]
)

# Legacy alias kept for backward compatibility
CH = {n: i for i, n in enumerate(CH_NAMES)}


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def fill_gaps(x, valid, max_gap):
    """Linear-interpolate short detection gaps (<= max_gap frames)."""
    T = len(valid)
    good = np.flatnonzero(valid)
    if len(good) < 2:
        return None, np.zeros(T, bool)
    flat = x.reshape(T, -1)
    out = np.empty_like(flat)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(np.arange(T), good, flat[good, j])
    rel = valid.copy()
    i = 0
    while i < T:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < T and not valid[j]:
            j += 1
        if j - i <= max_gap and i > 0 and j < T:
            rel[i:j] = True
        i = j
    return out.reshape(x.shape), rel


def align(pts, anc, template):
    """Procrustes rigid-body alignment of perioral points to a canonical template.

    Removes head-pose and global scale variation, consistent with the
    mouth-centre / inter-commissure normalisation in paper Eq. 1.
    """
    T, K, _ = anc.shape
    S = anc[..., :2]
    mu_s = S.mean(1, keepdims=True)
    Sc = S - mu_s
    mu_d = template.mean(0)
    Dc = template - mu_d
    U, sv, Vt = np.linalg.svd(np.einsum("ki,tkj->tij", Dc, Sc) / K)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    d[d == 0] = 1
    D = np.stack([np.ones_like(d), d], 1)
    R = np.einsum("tij,tj,tjk->tik", U, D, Vt)
    c = (sv * D).sum(1) / np.maximum((Sc ** 2).sum((1, 2)) / K, 1e-9)
    tf = lambda P: c[:, None, None] * np.einsum("tij,tkj->tki", R, P - mu_s) + mu_d
    out = np.empty_like(pts)
    out[..., :2] = tf(pts[..., :2])
    out[..., 2] = (pts[..., 2] - anc[..., 2].mean(1, keepdims=True)) * c[:, None]
    resid = np.sqrt(((tf(S) - template) ** 2).sum(-1).mean(1))
    return out, resid


# ---------------------------------------------------------------------------
# Paper-aligned feature extraction (§III-C, §III-D)
# ---------------------------------------------------------------------------

def compute_xs(y_win):
    """Statistical feature vector xs in R^256 (Eqs. 2-5, §III-D).

    Parameters
    ----------
    y_win : np.ndarray, shape (T, 64) — y-coords for one window.

    Returns
    -------
    xs : np.ndarray, shape (256,) — std of displacement, velocity,
         acceleration, jerk for each of the 64 landmarks.
    """
    orders = [y_win]  # disp (order 0)
    dy = y_win
    for _ in range(3):
        dy = np.diff(dy, axis=0, prepend=dy[:1])
        orders.append(dy)
    # sigma^(k)_i = std_t over the window
    xs = np.concatenate([o.std(axis=0) for o in orders])  # shape (256,)
    return xs.astype(np.float32)


def compute_xt(y_win):
    """Temporal sequence Xt in R^{T x 256} (Eq. 6, §III-D).

    Concatenates per-frame y with three finite differences, each padded
    to length T by repeating the boundary (prepend first value).
    """
    dy1 = np.diff(y_win, axis=0, prepend=y_win[:1])
    dy2 = np.diff(dy1,   axis=0, prepend=dy1[:1])
    dy3 = np.diff(dy2,   axis=0, prepend=dy2[:1])
    Xt = np.concatenate([y_win, dy1, dy2, dy3], axis=1)  # (T, 256)
    return Xt.astype(np.float32)


def compute_xr(y_win):
    """Region ratio vector xr in R^8 (Eq. 7, §III-D).

    rli = mean-displacement-std of inner lower lip
    rlo = outer lower lip
    rup = upper lip (inner + outer averaged)
    rpe = perioral (jaw + mid)
    """
    def _r(slc):
        return float(y_win[:, slc].std(axis=0).mean())

    rli = _r(REG_IDX["lo_inner"])
    rlo = _r(REG_IDX["lo_outer"])
    rup = (_r(REG_IDX["up_inner"]) + _r(REG_IDX["up_outer"])) / 2.0
    rpe = (_r(REG_IDX["jaw"]) + _r(REG_IDX["mid"])) / 2.0

    eps = 1e-9
    xr = np.array([
        rli / (rup + eps),
        rlo / (rup + eps),
        rli / (rlo + eps),
        rpe / (rup + eps),
        rli / (rpe + eps),
        rli,
        rup,
        (rli - rup) / (rli + rup + eps),
    ], dtype=np.float32)
    return xr


# ---------------------------------------------------------------------------
# Build per-clip channel matrix
# ---------------------------------------------------------------------------

def build_channels(lm, anc, valid, cfg):
    """Build kinematics for a single clip.

    Returns
    -------
    ch     : np.ndarray (T, C)  Legacy log-mean channel matrix (GBM branch).
    rel    : np.ndarray (T,)    Validity mask after gap filling.
    y_norm : np.ndarray (T, 64) Aligned y-coords — paper feature input.
    resid  : np.ndarray (T,)    Per-frame alignment residual.
    """
    from scipy.signal import savgol_filter

    P0, rel = fill_gaps(lm, valid, cfg.max_gap)
    A0, _   = fill_gaps(anc, valid, cfg.max_gap)
    if P0 is None or rel.sum() < cfg.win // 2:
        return None

    # Procrustes alignment (head-pose removal, equivalent to Eq. 1)
    tm = A0[rel][:, :, :2].mean(0)
    tm -= tm.mean(0)
    tm /= np.linalg.norm(tm[0] - tm[3]) + 1e-9
    P, resid = align(P0, A0, tm)

    # ---- Legacy GBM channel matrix (log-mean of L2 kinematics) ---------
    K_sg = {0: np.linalg.norm(P - np.median(P, axis=0, keepdims=True), axis=-1)}
    for k in (1, 2, 3):
        K_sg[k] = np.linalg.norm(
            savgol_filter(P, cfg.sg_win, cfg.sg_poly, deriv=k,
                          delta=1 / cfg.fps, axis=0, mode="interp"),
            axis=-1,
        )
    cols = [
        np.log(K_sg[k][:, ix].mean(1) + 1e-3)
        for ix in REG_IDX.values()
        for k in range(4)
    ]
    POS = {p: i for i, p in enumerate(PERIORAL)}
    d = lambda a, b: np.linalg.norm(P[:, POS[a], :2] - P[:, POS[b], :2], axis=-1)
    geo = [d(13, 14), d(0, 17), d(78, 308), d(61, 291), d(152, 2)]
    ch = np.column_stack(cols + geo + [np.log(resid + 1e-3)]).astype(np.float32)

    # ---- Paper-aligned y-only representation ----------------------------
    y_norm = P[:, :, 1].astype(np.float32)  # shape (T, 64) — y-coord only

    return ch, rel, y_norm, resid


# ---------------------------------------------------------------------------
# Window extraction — produces all representations
# ---------------------------------------------------------------------------

def extract_windows(manifest, cfg):
    """Extract sliding-window feature representations from all clips.

    Returns
    -------
    clip_ch : dict[key -> ch array]
        Legacy per-clip channel matrix.
    Xw  : np.ndarray (N, win, C)
        Raw window channel data for GBM feature extraction.
    Xs  : np.ndarray (N, 256)
        Statistical vectors xs (paper Eqs. 2-5).
    Xt  : np.ndarray (N, win, 256)
        Temporal sequences Xt (paper Eq. 6).
    Xr  : np.ndarray (N, 8)
        Region ratio vectors xr (paper Eq. 7).
    meta : pd.DataFrame
        Window metadata (key, start, split, generator, label, clip_id).
    t_feat : float
        Wall-clock seconds spent in feature extraction.
    """
    clip_ch = {}
    Xw_rows, Xs_rows, Xt_rows, Xr_rows, meta_rows = [], [], [], [], []
    t_feat = 0.0

    for r in tqdm(list(manifest.itertuples()), desc="features"):
        d = np.load(r.lm_path)
        t0 = time.perf_counter()
        res = build_channels(d["lm"], d["anc"], d["valid"], cfg)
        t_feat += time.perf_counter() - t0
        if res is None:
            continue
        ch, rel, y_norm, resid = res
        clip_ch[r.key] = ch

        for s in range(0, len(ch) - cfg.win + 1, cfg.stride):
            if rel[s: s + cfg.win].mean() >= cfg.min_valid:
                Xw_rows.append(ch[s: s + cfg.win])
                y_win = y_norm[s: s + cfg.win]     # (25, 64)
                Xs_rows.append(compute_xs(y_win))  # (256,)
                Xt_rows.append(compute_xt(y_win))  # (25, 256)
                Xr_rows.append(compute_xr(y_win))  # (8,)
                meta_rows.append((r.key, s, r.split, r.generator, r.label, r.clip_id))

    Xw = np.stack(Xw_rows)
    Xs = np.stack(Xs_rows)
    Xt = np.stack(Xt_rows)
    Xr = np.stack(Xr_rows)
    meta = pd.DataFrame(meta_rows, columns=["key", "start", "split", "generator", "label", "clip_id"])
    return clip_ch, Xw, Xs, Xt, Xr, meta, t_feat


# ---------------------------------------------------------------------------
# GBM tabular features (legacy branch — not in paper)
# ---------------------------------------------------------------------------

def window_feats(w):
    """Aggregate per-window statistics for the GBM branch (custom, not paper)."""
    stats = [
        w.mean(0), w.std(0), w.min(0), w.max(0),
        np.percentile(w, 10, 0), np.percentile(w, 90, 0),
    ]
    jerky = [
        w[:, CH[f"{r}_jerk"]].mean() - w[:, CH[f"{r}_vel"]].mean()
        for r in REG_IDX
    ]
    return np.concatenate(stats + [np.asarray(jerky)])


def make_gbm_features(Xw):
    Fw = np.nan_to_num(np.stack([window_feats(w) for w in Xw]))
    W_NAMES = [
        f"{c}__{s}"
        for s in ["mean", "std", "min", "max", "p10", "p90"]
        for c in CH_NAMES
    ] + [f"{r}__jerk_over_vel" for r in REG_IDX]
    return Fw, W_NAMES
