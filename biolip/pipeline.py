import json, pickle, time
from pathlib import Path
import numpy as np

try:
    from .config import get_config
    from .landmark_defs import PERIORAL, ANCHORS
    from .data import build_manifest, download_and_reencode
    from .landmarks import extract_landmarks, quality_filter
    from .features import extract_windows, make_gbm_features, CH_NAMES
    from .models import train_gbm, train_cnn_models
    from .calibration import calibrate
    from .evaluation import evaluate
    from .visualization import make_curves
except (ImportError, ValueError):
    from config import get_config
    from landmark_defs import PERIORAL, ANCHORS
    from data import build_manifest, download_and_reencode
    from landmarks import extract_landmarks, quality_filter
    from features import extract_windows, make_gbm_features, CH_NAMES
    from models import train_gbm, train_cnn_models
    from calibration import calibrate
    from evaluation import evaluate
    from visualization import make_curves


cfg = get_config()
cfg.make_dirs()
np.random.seed(cfg.seed)
manifest = build_manifest(cfg)
manifest = download_and_reencode(manifest, cfg)
manifest = extract_landmarks(manifest, cfg, PERIORAL, ANCHORS)
# manifest = quality_filter(manifest, cfg)
# manifest.to_csv(cfg.outputs / "manifest.csv", index=False)
# clip_ch, Xw, meta, t_feat = extract_windows(manifest, cfg)
# Fw, W_NAMES = make_gbm_features(Xw)
# gbm, z_gbm, oof_gbm, gbm_cfg = train_gbm(Fw, meta, cfg)
# models, eps, z_cnn, frame_logits, oof_cnn, mu, sd = train_cnn_models(Xw, meta, cfg)
# C, band, cal_models, p_oof, y_oof = calibrate(
#     meta, z_gbm, z_cnn, oof_gbm, oof_cnn, cfg.w_cnn, cfg.target_acc
# )
# R, share = evaluate(C)
# sig = lambda z: 1 / (1 + np.exp(-z))
# curves = make_curves(meta, clip_ch, frame_logits, sig, cfg.win)
# C.reset_index()[
#     [
#         "key",
#         "split",
#         "generator",
#         "label",
#         "clip_id",
#         "z_gbm",
#         "z_cnn",
#         "p_gbm",
#         "p_cnn",
#         "p_ens",
#         "verdict",
#     ]
# ].to_csv(cfg.outputs / "scores_clip.csv", index=False)
# W = meta.copy()
# W["z_gbm"] = z_gbm
# W["z_cnn"] = z_cnn
# W["oof_gbm"] = oof_gbm
# W["oof_cnn"] = oof_cnn
# W.to_csv(cfg.outputs / "scores_window.csv", index=False)
# np.savez_compressed(
#     cfg.outputs / "frame_curves.npz",
#     **{k.replace("/", "|"): v for k, v in curves.items()},
# )
# __import__("gzip")
# import pandas as pd

# pd.DataFrame(Fw, columns=W_NAMES).assign(
#     key=meta.key.values, start=meta.start.values
# ).to_csv(cfg.outputs / "features_window.csv.gz", index=False)
# R.to_csv(cfg.outputs / "metrics_basic.csv", index=False)
# np.save(cfg.outputs / "window_channels.npy", Xw)
# (cfg.outputs / "channel_names.json").write_text(json.dumps(CH_NAMES))
# (cfg.outputs / "models").mkdir(exist_ok=True)
# with open(cfg.outputs / "models/gbm.pkl", "wb") as f:
#     pickle.dump(gbm, f)
# import torch

# for s, m in zip(cfg.cnn_seeds, models):
#     torch.save(m.state_dict(), cfg.outputs / f"models/cnn_seed{s}.pt")
# json.dump(
#     dict(
#         PAIRS=cfg.pairs,
#         WIN=cfg.win,
#         STRIDE=cfg.stride,
#         GBM_CFG=gbm_cfg,
#         CNN_epochs=int(np.median(eps)),
#         W_CNN=cfg.w_cnn,
#         band=list(band),
#         target_acc=cfg.target_acc,
#         channel_mean=mu.tolist(),
#         channel_std=sd.tolist(),
#         landmarks=PERIORAL,
#         anchors=ANCHORS,
#     ),
#     open(cfg.outputs / "config.json", "w"),
#     indent=1,
# )
# print("saved to", cfg.outputs)
# print(
#     R.pivot_table(
#         index=["split", "generator"], columns="model", values=["AUC", "ACC", "EER"]
#     ).round(3)
# )
# return C, R
