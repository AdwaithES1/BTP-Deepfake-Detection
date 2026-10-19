"""BioLip training entry point — arXiv:2604.16808v4.

Run (single GPU / CPU):
    accelerate launch biolip/train.py

Run (multi-GPU, e.g. 4 GPUs):
    accelerate launch --config_file biolip/accelerate_config.yaml biolip/train.py

Dry run (synthetic data, terminates after 2 mini-epochs to verify the loop):
    accelerate launch biolip/train.py --dry-run

All saved artifacts go to:
    $BIOLIP_ROOT/outputs/
    ├── models/
    │   ├── biolipnet_seed{N}_best.pt      # best checkpoint per seed
    │   ├── biolipnet_seed{N}_final.pt     # final state per seed
    │   └── biolipnet_config.json          # architecture + normalisation stats
    ├── scores_clip.csv                    # clip-level scores for all splits
    ├── scores_window.csv                  # window-level scores
    ├── oof_clip.csv                       # OOF scores on train split (LOGO)
    ├── metrics_basic.csv                  # AUC/EER/ACC per split & generator
    └── training_log.jsonl                 # epoch-by-epoch training log
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from sklearn.metrics import roc_auc_score

# ── resolve imports whether run as a module or a script ──────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from config import get_config, Config
from models import BioLipNet, clip_mean, clip_auc


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_video_scores(meta: pd.DataFrame, win_logits: np.ndarray) -> pd.Series:
    """Average window-level sigmoid logits to clip-level probability (§III-F)."""
    sig = torch.sigmoid(torch.from_numpy(win_logits)).numpy()
    return (
        pd.DataFrame({"key": meta.key.values, "p": sig})
        .groupby("key")
        .p.mean()
    )


def _video_auc(meta: pd.DataFrame, win_logits: np.ndarray) -> float:
    p = _make_video_scores(meta, win_logits)
    y = meta.drop_duplicates("key").set_index("key").label.loc[p.index]
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _log(accel: Accelerator, msg: str):
    if accel.is_main_process:
        print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dry-run dataset
# ─────────────────────────────────────────────────────────────────────────────

def _make_dry_run_data(cfg: Config, n_clips_per_gen: int = 6):
    """Generate tiny synthetic Xt/Xr/Xs arrays and a matching meta DataFrame.

    Creates 3 generators × 3 splits × n_clips_per_gen clips (balanced real/fake).
    Window count per clip = ceil((max_frames - win + 1) / stride).
    """
    rng = np.random.default_rng(cfg.seed)
    generators = {"train": ["gen_a", "gen_b", "gen_c"],
                  "val":   ["gen_a", "gen_b", "gen_c"],
                  "test":  ["gen_d", "gen_e", "gen_f"]}   # OOD

    frames_per_clip = cfg.max_frames
    wins_per_clip = max(1, (frames_per_clip - cfg.win) // cfg.stride + 1)
    T = cfg.win  # 25

    rows, Xt_list, Xr_list, Xs_list = [], [], [], []

    for split, gens in generators.items():
        for gen in gens:
            for ci in range(n_clips_per_gen):
                label = ci % 2              # alternating real/fake
                clip_id = f"{gen}_{ci:03d}"
                key = f"{split}/{gen}/{clip_id}"
                for wi in range(wins_per_clip):
                    rows.append({"key": key, "start": wi * cfg.stride,
                                 "split": split, "generator": gen,
                                 "label": label, "clip_id": clip_id})
                    # Fake clips have slightly higher kinematic variance (paper Fig.2)
                    scale = 1.3 if label == 1 else 1.0
                    Xt_list.append(rng.standard_normal((T, 256)).astype(np.float32) * scale)
                    Xr_list.append(rng.standard_normal(8).astype(np.float32) * scale)
                    Xs_list.append(np.abs(rng.standard_normal(256).astype(np.float32)) * scale)

    meta = pd.DataFrame(rows)
    Xt = np.stack(Xt_list)   # (N, T, 256)
    Xr = np.stack(Xr_list)   # (N, 8)
    Xs = np.stack(Xs_list)   # (N, 256)
    return meta, Xt, Xr, Xs


# ─────────────────────────────────────────────────────────────────────────────
# Core training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_seed(
    seed: int,
    Xt_n: np.ndarray,
    Xr: np.ndarray,
    Xs_n: np.ndarray,
    meta: pd.DataFrame,
    cfg: Config,
    out_dir: Path,
    accel: Accelerator,
    dry_run: bool = False,
) -> tuple[BioLipNet, list[dict], np.ndarray]:
    """Train BioLipNet for one random seed following §III-F.

    Returns the best model, the epoch log, and per-window logits.
    """
    is_tr = (meta.split == "train").values
    is_va = (meta.split == "val").values

    y_all = torch.tensor(meta.label.values, dtype=torch.float32)
    n_pos = int(is_tr.sum() * meta.label.values[is_tr].mean())
    n_neg = int(is_tr.sum()) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    # ── model, optimiser, scheduler ─────────────────────────────────────────
    set_seed(seed)
    model = BioLipNet(t_len=cfg.win)

    epochs  = 2 if dry_run else cfg.cnn_epochs
    patience = 2 if dry_run else cfg.cnn_patience
    batch   = 128
    idx_tr  = np.flatnonzero(is_tr)
    n_steps = epochs * int(math.ceil(len(idx_tr) / batch))

    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-5)

    model, opt, sched = accel.prepare(model, opt, sched)
    pos_weight = pos_weight.to(accel.device)

    # ── predict helper (no_grad, batched) ───────────────────────────────────
    @torch.no_grad()
    def predict_all(mask: np.ndarray) -> np.ndarray:
        model.eval()
        idxs   = np.flatnonzero(mask)
        logits = []
        for i in range(0, len(idxs), 512):
            b  = idxs[i: i + 512]
            xt = torch.from_numpy(Xt_n[b]).to(accel.device)
            xr = torch.from_numpy(Xr[b]).to(accel.device)
            xs = torch.from_numpy(Xs_n[b]).to(accel.device)
            logits.append(accel.unwrap_model(model)(xt, xr, xs).cpu())
        return torch.cat(logits).numpy()

    # ── training loop ────────────────────────────────────────────────────────
    best_auc   = -1.0
    best_ep    = 0
    bad        = 0
    best_state = None
    epoch_log  = []

    _log(accel, f"\n{'─'*60}")
    _log(accel, f"  Seed {seed} | {'DRY-RUN ' if dry_run else ''}"
                f"{epochs} epochs | patience {patience} | "
                f"train windows {is_tr.sum()} | val windows {is_va.sum()}")
    _log(accel, f"{'─'*60}")

    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        np.random.shuffle(idx_tr)
        ep_loss = 0.0
        n_batches = 0

        for i in range(0, len(idx_tr), batch):
            b  = idx_tr[i: i + batch]
            xt = torch.from_numpy(Xt_n[b]).to(accel.device)
            xr = torch.from_numpy(Xr[b]).to(accel.device)
            xs = torch.from_numpy(Xs_n[b]).to(accel.device)
            yb = y_all[b].to(accel.device)

            logit = model(xt, xr, xs)
            loss  = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight)
            opt.zero_grad()
            accel.backward(loss)
            torch.nn.utils.clip_grad_norm_(accel.unwrap_model(model).parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss   += loss.item()
            n_batches += 1

        # ── validation ───────────────────────────────────────────────────────
        val_logits = predict_all(is_va)
        val_auc    = _video_auc(meta[is_va], val_logits)
        train_logits = predict_all(is_tr)
        train_auc  = _video_auc(meta[is_tr], train_logits)

        elapsed = time.perf_counter() - t0
        lr_now  = sched.get_last_lr()[0]

        entry = {
            "seed": seed, "epoch": ep + 1,
            "train_loss": round(ep_loss / max(n_batches, 1), 5),
            "train_auc":  round(train_auc, 4),
            "val_auc":    round(val_auc, 4),
            "lr":         round(lr_now, 8),
            "elapsed_s":  round(elapsed, 2),
        }
        epoch_log.append(entry)

        _log(accel,
             f"  ep {ep+1:3d}/{epochs}  loss={entry['train_loss']:.4f}  "
             f"tr_auc={train_auc:.3f}  val_auc={val_auc:.3f}  "
             f"lr={lr_now:.2e}  [{elapsed:.1f}s]")

        # ── early stopping ───────────────────────────────────────────────────
        if val_auc > best_auc + 1e-4:
            best_auc   = val_auc
            best_ep    = ep + 1
            bad        = 0
            best_state = {
                k: v.clone()
                for k, v in accel.unwrap_model(model).state_dict().items()
            }
            if accel.is_main_process:
                ckpt = out_dir / "models" / f"biolipnet_seed{seed}_best.pt"
                torch.save(best_state, ckpt)
                _log(accel, f"    ✓ new best val AUC {best_auc:.4f} → saved {ckpt.name}")
        else:
            bad += 1
            if bad >= patience:
                _log(accel, f"  Early stop at epoch {ep+1} (patience {patience})")
                break

    # Restore best weights
    if best_state is not None:
        accel.unwrap_model(model).load_state_dict(best_state)

    _log(accel, f"  Seed {seed} done  best_ep={best_ep}  best_val_auc={best_auc:.4f}")

    # Final checkpoint
    if accel.is_main_process:
        ckpt = out_dir / "models" / f"biolipnet_seed{seed}_final.pt"
        torch.save(accel.unwrap_model(model).state_dict(), ckpt)
        _log(accel, f"  Saved final checkpoint: {ckpt.name}")

    # Full-data logits for this seed
    all_logits = predict_all(np.ones(len(meta), dtype=bool))
    return accel.unwrap_model(model), epoch_log, all_logits, best_ep


# ─────────────────────────────────────────────────────────────────────────────
# OOF (Leave-One-Generator-Out cross-validation on train split)
# ─────────────────────────────────────────────────────────────────────────────

def compute_oof(
    Xt_n: np.ndarray,
    Xr: np.ndarray,
    Xs_n: np.ndarray,
    meta: pd.DataFrame,
    cfg: Config,
    fixed_ep: int,
    accel: Accelerator,
    dry_run: bool = False,
) -> np.ndarray:
    """LOGO OOF on the train split using the median best-epoch."""
    is_tr  = (meta.split == "train").values
    tr_gens = sorted(meta[is_tr].generator.unique())
    oof    = np.full(len(meta), np.nan)
    epochs = min(fixed_ep, 2) if dry_run else fixed_ep

    _log(accel, f"\n  OOF (LOGO, {len(tr_gens)} folds, fixed_ep={epochs})")

    for g in tr_gens:
        a_mask = is_tr & (meta.generator != g).values
        b_mask = is_tr & (meta.generator == g).values

        set_seed(0)
        m_oof = BioLipNet(t_len=cfg.win)
        idx_a = np.flatnonzero(a_mask)
        n_pos = int(a_mask.sum() * meta.label.values[a_mask].mean())
        n_neg = int(a_mask.sum()) - n_pos
        pw    = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

        opt_oof  = torch.optim.AdamW(m_oof.parameters(), lr=3e-4, weight_decay=1e-4)
        n_steps  = epochs * int(math.ceil(len(idx_a) / 128))
        sch_oof  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_oof, T_max=max(n_steps,1), eta_min=1e-5)
        m_oof, opt_oof, sch_oof = accel.prepare(m_oof, opt_oof, sch_oof)
        pw = pw.to(accel.device)

        for ep in range(epochs):
            m_oof.train()
            np.random.shuffle(idx_a)
            for i in range(0, len(idx_a), 128):
                b  = idx_a[i: i + 128]
                xt = torch.from_numpy(Xt_n[b]).to(accel.device)
                xr = torch.from_numpy(Xr[b]).to(accel.device)
                xs = torch.from_numpy(Xs_n[b]).to(accel.device)
                yb = torch.tensor(meta.label.values[b], dtype=torch.float32).to(accel.device)
                logit = m_oof(xt, xr, xs)
                loss  = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pw)
                opt_oof.zero_grad()
                accel.backward(loss)
                opt_oof.step()
                sch_oof.step()

        @torch.no_grad()
        def _pred_oof(mask):
            m_oof.eval()
            idxs, out = np.flatnonzero(mask), []
            for i in range(0, len(idxs), 512):
                b  = idxs[i: i + 512]
                xt = torch.from_numpy(Xt_n[b]).to(accel.device)
                xr = torch.from_numpy(Xr[b]).to(accel.device)
                xs = torch.from_numpy(Xs_n[b]).to(accel.device)
                out.append(accel.unwrap_model(m_oof)(xt, xr, xs).cpu())
            return torch.cat(out).numpy()

        oof[b_mask] = _pred_oof(b_mask)
        held_auc = clip_auc(meta[b_mask], oof[b_mask])
        _log(accel, f"    LOGO held-out {g}: clip AUC {held_auc:.3f}")

    return oof


# ─────────────────────────────────────────────────────────────────────────────
# Save artifacts
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(
    out_dir: Path,
    meta: pd.DataFrame,
    z_ens: np.ndarray,       # ensemble mean logit (window-level)
    oof_logits: np.ndarray,  # OOF logits (NaN for non-train)
    epoch_logs: list[dict],
    mu_Xt: np.ndarray, sd_Xt: np.ndarray,
    mu_Xs: np.ndarray, sd_Xs: np.ndarray,
    cfg: Config,
    best_eps: list[int],
    accel: Accelerator,
):
    if not accel.is_main_process:
        return

    # ── window-level scores ──────────────────────────────────────────────────
    W = meta.copy()
    W["z_ens"] = z_ens
    W["oof"]   = oof_logits
    W.to_csv(out_dir / "scores_window.csv", index=False)
    _log(accel, f"  Saved scores_window.csv  ({len(W)} rows)")

    # ── clip-level scores ────────────────────────────────────────────────────
    sig = torch.sigmoid(torch.from_numpy(z_ens)).numpy()
    oof_sig = np.where(np.isnan(oof_logits), np.nan,
                       torch.sigmoid(torch.from_numpy(
                           np.nan_to_num(oof_logits))).numpy())

    clip_df = (
        pd.DataFrame({"key": meta.key, "split": meta.split, "generator": meta.generator,
                      "label": meta.label, "clip_id": meta.clip_id,
                      "z_ens": z_ens, "p_ens": sig,
                      "oof_logit": oof_logits, "p_oof": oof_sig})
        .groupby("key")
        .agg({"split": "first", "generator": "first", "label": "first",
              "clip_id": "first", "z_ens": "mean", "p_ens": "mean",
              "oof_logit": "mean", "p_oof": "mean"})
        .reset_index()
    )
    clip_df.to_csv(out_dir / "scores_clip.csv", index=False)
    _log(accel, f"  Saved scores_clip.csv    ({len(clip_df)} clips)")

    # ── OOF clip-level ───────────────────────────────────────────────────────
    oof_clip = clip_df[clip_df.split == "train"][["key","generator","label","oof_logit","p_oof"]]
    oof_clip.to_csv(out_dir / "oof_clip.csv", index=False)
    _log(accel, f"  Saved oof_clip.csv       ({len(oof_clip)} train clips)")

    # ── metrics_basic.csv ────────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score, accuracy_score
    from evaluation import eer as _eer

    rows = []
    for split in ["train", "val", "test"]:
        sub = clip_df[clip_df.split == split]
        y   = sub.label.values
        p   = sub.p_ens.values
        if len(np.unique(y)) < 2:
            continue
        rows.append({
            "split": split, "generator": "ALL", "model": "biolipnet_ens",
            "n": len(sub),
            "AUC":  round(float(roc_auc_score(y, p)), 4),
            "ACC":  round(float(((p >= 0.5) == y).mean()), 4),
            "EER":  round(float(_eer(y, p)), 4),
        })
        for gen, gdf in sub.groupby("generator"):
            yg, pg = gdf.label.values, gdf.p_ens.values
            if len(np.unique(yg)) < 2:
                continue
            rows.append({
                "split": split, "generator": gen, "model": "biolipnet_ens",
                "n": len(gdf),
                "AUC":  round(float(roc_auc_score(yg, pg)), 4),
                "ACC":  round(float(((pg >= 0.5) == yg).mean()), 4),
                "EER":  round(float(_eer(yg, pg)), 4),
            })
    pd.DataFrame(rows).to_csv(out_dir / "metrics_basic.csv", index=False)
    _log(accel, f"  Saved metrics_basic.csv  ({len(rows)} rows)")

    # ── training log ─────────────────────────────────────────────────────────
    log_path = out_dir / "training_log.jsonl"
    with open(log_path, "w") as f:
        for entry in epoch_logs:
            f.write(json.dumps(entry) + "\n")
    _log(accel, f"  Saved training_log.jsonl ({len(epoch_logs)} entries)")

    # ── model config ─────────────────────────────────────────────────────────
    model_cfg = {
        "architecture": "BioLipNet",
        "paper": "arXiv:2604.16808v4",
        "win": cfg.win,
        "stride": cfg.stride,
        "cnn_seeds": list(cfg.cnn_seeds),
        "best_epochs": best_eps,
        "cnn_epochs": cfg.cnn_epochs,
        "cnn_patience": cfg.cnn_patience,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "mu_Xs": mu_Xs.tolist(),
        "sd_Xs": sd_Xs.tolist(),
        "mu_Xt": mu_Xt.tolist(),
        "sd_Xt": sd_Xt.tolist(),
    }
    with open(out_dir / "models" / "biolipnet_config.json", "w") as f:
        json.dump(model_cfg, f, indent=2)
    _log(accel, f"  Saved biolipnet_config.json")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BioLip training — arXiv:2604.16808v4")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 2 epochs on synthetic data to verify the training loop.")
    parser.add_argument("--features-path", type=str, default=None,
                        help="Path to a .npz file with keys Xt, Xr, Xs, and a CSV meta path.")
    parser.add_argument("--meta-path", type=str, default=None,
                        help="CSV path for the window-level metadata DataFrame.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output directory (default: $BIOLIP_ROOT/outputs).")
    args = parser.parse_args()

    cfg = get_config()

    # ── output dir ───────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir) if args.out_dir else cfg.outputs
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    # ── accelerate setup ─────────────────────────────────────────────────────
    proj_cfg = ProjectConfiguration(project_dir=str(out_dir), logging_dir=str(out_dir / "logs"))
    accel    = Accelerator(project_config=proj_cfg, log_with=None)

    _log(accel, f"\n{'='*60}")
    _log(accel, f"  BioLip Training  |  arXiv:2604.16808v4")
    _log(accel, f"  Devices: {accel.num_processes}  |  {'DRY-RUN' if args.dry_run else 'FULL'}")
    _log(accel, f"  Output: {out_dir}")
    _log(accel, f"{'='*60}\n")

    # ── data ─────────────────────────────────────────────────────────────────
    if args.dry_run:
        _log(accel, "  [DRY-RUN] Generating synthetic data...")
        meta, Xt, Xr, Xs = _make_dry_run_data(cfg, n_clips_per_gen=8)
        _log(accel, f"  Windows: {len(meta)}  (train={( meta.split=='train').sum()}, "
                    f"val={(meta.split=='val').sum()}, test={(meta.split=='test').sum()})")
    else:
        if args.features_path is None or args.meta_path is None:
            # Try loading from default pipeline output location
            feat_path = cfg.outputs / "window_features.npz"
            meta_path = cfg.outputs / "scores_window.csv"
            if not feat_path.exists():
                raise FileNotFoundError(
                    f"No features found at {feat_path}. "
                    "Run the data pipeline first, or pass --features-path / --meta-path, "
                    "or use --dry-run to verify the training loop."
                )
        else:
            feat_path = Path(args.features_path)
            meta_path = Path(args.meta_path)

        _log(accel, f"  Loading features from {feat_path}")
        data = np.load(feat_path)
        Xt   = data["Xt"]
        Xr   = data["Xr"]
        Xs   = data["Xs"]
        meta = pd.read_csv(meta_path)
        _log(accel, f"  Windows: {len(meta)}")

    # ── normalise (fit on train) ──────────────────────────────────────────────
    is_tr = (meta.split == "train").values

    mu_Xs = Xs[is_tr].mean(0);  sd_Xs = Xs[is_tr].std(0) + 1e-6
    Xs_n  = ((Xs - mu_Xs) / sd_Xs).astype(np.float32)

    mu_Xt = Xt[is_tr].reshape(-1, 256).mean(0)
    sd_Xt = Xt[is_tr].reshape(-1, 256).std(0) + 1e-6
    Xt_n  = ((Xt - mu_Xt) / sd_Xt).astype(np.float32)

    # ── train each seed ──────────────────────────────────────────────────────
    all_logits  = []
    all_logs    = []
    best_eps    = []

    for seed in cfg.cnn_seeds:
        model, ep_log, logits, best_ep = train_seed(
            seed, Xt_n, Xr, Xs_n, meta, cfg, out_dir, accel, dry_run=args.dry_run
        )
        all_logits.append(logits)
        all_logs.extend(ep_log)
        best_eps.append(best_ep)

    # ── ensemble ─────────────────────────────────────────────────────────────
    z_ens = np.nanmean(all_logits, axis=0)

    # ── OOF (LOGO on train) ───────────────────────────────────────────────────
    fixed_ep = int(np.median(best_eps)) if best_eps else 2
    oof_logits = compute_oof(
        Xt_n, Xr, Xs_n, meta, cfg, fixed_ep, accel, dry_run=args.dry_run
    )

    # ── save all artifacts ────────────────────────────────────────────────────
    _log(accel, f"\n  Saving artifacts to {out_dir} ...")
    save_artifacts(
        out_dir, meta, z_ens, oof_logits, all_logs,
        mu_Xt, sd_Xt, mu_Xs, sd_Xs, cfg, best_eps, accel
    )

    # ── final summary ─────────────────────────────────────────────────────────
    if accel.is_main_process:
        _log(accel, f"\n{'='*60}")
        _log(accel, "  Saved files:")
        for p in sorted(out_dir.rglob("*")):
            if p.is_file() and p.suffix in {".pt", ".json", ".csv", ".jsonl"}:
                _log(accel, f"    {p.relative_to(out_dir)}")
        _log(accel, f"{'='*60}")
        _log(accel, "  Training complete.\n")


if __name__ == "__main__":
    main()
