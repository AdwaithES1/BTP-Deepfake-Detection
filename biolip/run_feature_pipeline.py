"""End-to-end landmark and feature extraction pipeline for BioLip.

Extracts:
1. MediaPipe facial landmarks (perioral 64 + anchors 8) from re-encoded videos.
2. Paper-defined BioLip features:
   - Xt: (N, 25, 256) temporal kinematic sequences
   - Xr: (N, 8) anatomical region ratios
   - Xs: (N, 256) statistical perioral features
3. Saves:
   - biolip/outputs/window_features.npz
   - biolip/outputs/scores_window.csv
   - biolip/outputs/manifest.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import get_config, Config
from landmark_defs import PERIORAL, ANCHORS
from features import build_channels, extract_windows


# ─────────────────────────────────────────────────────────────────────────────
# Worker for MediaPipe Landmark Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _worker_extract_clip(args_tuple: tuple[str, str, str, int, int]) -> tuple[str, bool, Optional[str]]:
    """Worker function executed in worker processes.

    args_tuple: (src_path, dst_path, task_path, max_frames, fps)
    """
    src, dst, task_file, max_frames, fps = args_tuple
    dst_p = Path(dst)
    if dst_p.exists():
        return src, True, None

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        t0 = time.perf_counter()
        base_opts = mpp.BaseOptions(model_asset_path=task_file)
        opts = vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        fm = vision.FaceLandmarker.create_from_options(opts)

        cap = cv2.VideoCapture(src)
        L, A, V = [], [], []
        while len(V) < max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            ts_ms = int(len(V) * 1000 / fps)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = fm.detect_for_video(mp_img, ts_ms)

            if res.face_landmarks:
                pts = np.array([[q.x, q.y, q.z] for q in res.face_landmarks[0]], dtype=np.float32)
                pts *= np.array([W, H, W], dtype=np.float32)
                L.append(pts[PERIORAL])
                A.append(pts[ANCHORS])
                V.append(True)
            else:
                L.append(np.full((64, 3), np.nan, dtype=np.float32))
                A.append(np.full((len(ANCHORS), 3), np.nan, dtype=np.float32))
                V.append(False)

        cap.release()
        fm.close()

        dst_p.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst_p.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            lm=np.array(L, dtype=np.float32),
            anc=np.array(A, dtype=np.float32),
            valid=np.array(V, dtype=bool),
            t=time.perf_counter() - t0,
        )
        os.replace(tmp, dst_p)
        return src, True, None

    except Exception as e:
        return src, False, repr(e)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    reencoded_dir: Path,
    out_dir: Path,
    landmarks_dir: Path,
    task_path: Path,
    workers: int = 8,
    limit: Optional[int] = None,
):
    print("=" * 60)
    print("  BioLip Landmark & Feature Extraction Pipeline")
    print("=" * 60)
    print(f"Reencoded Dir : {reencoded_dir}")
    print(f"Landmarks Dir : {landmarks_dir}")
    print(f"Output Dir    : {out_dir}")
    print(f"Workers       : {workers}")

    cfg = get_config()
    out_dir.mkdir(parents=True, exist_ok=True)
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover videos
    mp4_files = sorted([f for f in reencoded_dir.rglob("*.mp4") if not f.name.endswith(".tmp.mp4")])
    if limit is not None:
        mp4_files = mp4_files[:limit]
    print(f"Found {len(mp4_files)} videos.")

    rows = []
    for f in mp4_files:
        parts = f.relative_to(reencoded_dir).parts
        if len(parts) == 4:
            split, label_name, generator, filename = parts
            clip_id = filename[:-4]
            label = 0 if label_name == "real" else 1
            key = f"{split}/{label_name}/{generator}/{clip_id}"
            pair = f"{split}/{generator}/{clip_id}"
            lm_path = landmarks_dir / f"{key}.npz"
            rows.append({
                "split": split,
                "generator": generator,
                "clip_id": clip_id,
                "label": label,
                "label_name": label_name,
                "key": key,
                "pair": pair,
                "reenc_path": str(f.resolve()),
                "lm_path": str(lm_path.resolve()),
            })

    manifest = pd.DataFrame(rows)

    # 2. Extract Landmarks with ProcessPoolExecutor
    print(f"\n[1/3] Extracting facial landmarks (MediaPipe) using {workers} processes...")
    tasks = [
        (r.reenc_path, r.lm_path, str(task_path.resolve()), cfg.max_frames, cfg.fps)
        for r in manifest.itertuples()
    ]

    already_done = sum(Path(t[1]).exists() for t in tasks)
    print(f"  Existing landmarks: {already_done}/{len(tasks)} (resuming remaining)")

    failed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker_extract_clip, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="Landmarks") as pbar:
            for fut in as_completed(futures):
                src, ok, err = fut.result()
                if not ok:
                    failed += 1
                    print(f"\nFailed on {src}: {err}")
                pbar.update(1)

    print(f"Landmark extraction complete. Succeeded: {len(tasks) - failed}, Failed: {failed}")

    # 3. Quality Filtering
    print("\n[2/3] Checking landmark quality and filtering...")
    qual = []
    for r in manifest.itertuples():
        p = Path(r.lm_path)
        if not p.exists():
            qual.append((0, 0.0, 0.0))
            continue
        try:
            d = np.load(p)
            v = d["valid"]
            qual.append((len(v), float(v.mean()), float(d["t"])))
        except Exception:
            qual.append((0, 0.0, 0.0))

    manifest["n_frames"] = [q[0] for q in qual]
    manifest["det_rate"] = [q[1] for q in qual]
    manifest["t_extract"] = [q[2] for q in qual]

    # Filter
    valid_mask = (manifest.n_frames >= cfg.win) & (manifest.det_rate >= 0.5)
    filtered_manifest = manifest[valid_mask].copy().reset_index(drop=True)
    print(f"  Clips passing quality filter: {len(filtered_manifest)}/{len(manifest)}")

    # Save manifest
    manifest_csv = out_dir / "manifest.csv"
    filtered_manifest.to_csv(manifest_csv, index=False)
    print(f"  Saved manifest to {manifest_csv}")

    # 4. Extract sliding-window features (Xt, Xr, Xs)
    print("\n[3/3] Extracting BioLip sliding window features (Xt, Xr, Xs)...")
    clip_ch, Xw, Xs, Xt, Xr, meta, t_feat = extract_windows(filtered_manifest, cfg)
    print(f"Extracted {len(meta)} windows across {len(filtered_manifest)} clips.")
    print(f"  Xt shape: {Xt.shape} (temporal sequences)")
    print(f"  Xr shape: {Xr.shape} (region ratios)")
    print(f"  Xs shape: {Xs.shape} (statistical std features)")

    # Save window_features.npz
    npz_path = out_dir / "window_features.npz"
    np.savez_compressed(npz_path, Xt=Xt, Xr=Xr, Xs=Xs)
    print(f"  ✓ Saved {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")

    # Save scores_window.csv
    meta_path = out_dir / "scores_window.csv"
    meta.to_csv(meta_path, index=False)
    print(f"  ✓ Saved {meta_path} ({len(meta)} rows)")

    # Copy / Mirror to biolip_data/outputs if different
    alt_out = cfg.outputs
    if alt_out.resolve() != out_dir.resolve():
        alt_out.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(npz_path, alt_out / "window_features.npz")
        shutil.copy(meta_path, alt_out / "scores_window.csv")
        shutil.copy(manifest_csv, alt_out / "manifest.csv")
        print(f"  ✓ Mirrored files to {alt_out}")

    print("\n" + "=" * 60)
    print("Pipeline finished successfully! You can now run training on MPS:")
    print("  biolip/.venv/bin/accelerate launch --num_processes 1 --mixed_precision no biolip/train.py")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="BioLip Feature Pipeline")
    parser.add_argument("--reencoded-dir", type=str, default="biolip/biolip_data/reencoded")
    parser.add_argument("--landmarks-dir", type=str, default="biolip/biolip_data/landmarks")
    parser.add_argument("--out-dir", type=str, default="biolip/outputs")
    parser.add_argument("--task-path", type=str, default="biolip/face_landmarker.task")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos (for testing)")
    args = parser.parse_args()

    run_pipeline(
        reencoded_dir=Path(args.reencoded_dir),
        out_dir=Path(args.out_dir),
        landmarks_dir=Path(args.landmarks_dir),
        task_path=Path(args.task_path),
        workers=args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
