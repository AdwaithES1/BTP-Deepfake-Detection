"""Data preparation module for BioLip pipeline.

Delegates to dataset_pipeline for robust, resumable, and atomic 4-stage dataset preparation.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Add repo root to sys.path so dataset_pipeline can be imported
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dataset_pipeline import (
    PipelineConfig,
    build_manifest,
    download_dataset,
    preprocess_dataset,
    validate_dataset,
    save_manifest,
    load_manifest,
)


def _ensure_pipeline_cfg(cfg) -> PipelineConfig:
    """Adapt biolip Config or PipelineConfig to PipelineConfig instance."""
    if isinstance(cfg, PipelineConfig):
        return cfg

    pairs = getattr(cfg, "pairs", {"train": 200, "val": 60, "test": 100})
    return PipelineConfig(
        repo=getattr(cfg, "repo", "zaqxsw0526/MMDF"),
        root=getattr(cfg, "root", Path("./biolip_data")),
        dataset_dir=getattr(cfg, "dataset_dir", None),
        reencoded_dir=getattr(cfg, "reencoded", None) or getattr(cfg, "reencoded_dir", None),
        wav_dir=getattr(cfg, "wavs", None) or getattr(cfg, "wav_dir", None),
        manifest_path=getattr(cfg, "manifest_path", None),
        failures_path=getattr(cfg, "failures_path", None),
        seed=getattr(cfg, "seed", 42),
        pairs=pairs,
        download_mode=getattr(cfg, "download_mode", "selected"),
        workers=getattr(cfg, "workers", None) or 4,
        ffmpeg_threads=getattr(cfg, "ffmpeg_threads", 2),
        fps=getattr(cfg, "fps", 25),
        video_size=getattr(cfg, "video_size", "512x512"),
        duration=getattr(cfg, "duration", 5.0),
        audio_sample_rate=getattr(cfg, "audio_sample_rate", 16000),
    )


def download_and_reencode(manifest: pd.DataFrame, cfg) -> pd.DataFrame:
    """Download, re-encode, and validate clips using the refactored 4-stage pipeline.

    Maintains backwards compatibility with existing biolip pipelines while ensuring:
    - Independent bulk downloading
    - Combined single-pass atomic FFmpeg encoding
    - Idempotent resumability
    - Media validation with ffprobe
    """
    pipe_cfg = _ensure_pipeline_cfg(cfg)

    # 1. Bulk download
    manifest, dl_fails = download_dataset(manifest, pipe_cfg)

    # 2. Local preprocessing
    manifest, prep_fails = preprocess_dataset(manifest, pipe_cfg)

    # 3. Media validation
    manifest, val_fails = validate_dataset(manifest, pipe_cfg)

    # Retain successful clips
    complete_manifest = manifest[manifest["status"] == "complete"].copy().reset_index(drop=True)
    if len(complete_manifest) == 0:
        all_errors = [f["error"] for f in (dl_fails + prep_fails + val_fails) if "error" in f]
        last_err = all_errors[-1] if all_errors else "unknown error"
        raise RuntimeError(f"No clips survived preprocessing and validation: {last_err}")

    # Set reenc_path and src_path for backwards compatibility with downstream modules
    complete_manifest["reenc_path"] = complete_manifest["video_path"]
    complete_manifest["src_path"] = complete_manifest["source_path"]

    return complete_manifest


__all__ = [
    "build_manifest",
    "download_dataset",
    "preprocess_dataset",
    "validate_dataset",
    "download_and_reencode",
    "save_manifest",
    "load_manifest",
]
