"""Unit and integration tests for the dataset preparation pipeline."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dataset_pipeline.config import PipelineConfig, PairsConfig
from dataset_pipeline.manifest import build_manifest, save_manifest, load_manifest
from dataset_pipeline.download import download_dataset
from dataset_pipeline.preprocess import preprocess_single_clip, preprocess_dataset
from dataset_pipeline.validate import validate_sample, validate_dataset, probe_media_file


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def synthetic_video(temp_workspace):
    """Generate a valid test MP4 video with audio using FFmpeg."""
    src_file = temp_workspace / "source_clip.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=3",
        "-c:v", "libx264", "-c:a", "aac",
        str(src_file)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return src_file


def test_pipeline_config(temp_workspace):
    """Test PipelineConfig initialization, paths, and properties."""
    cfg = PipelineConfig(root=temp_workspace, video_size="512x512", fps=25)
    assert cfg.width == 512
    assert cfg.height == 512
    assert cfg.pairs.train == 200
    assert cfg.pairs["val"] == 60

    cfg.make_dirs()
    assert cfg.dataset_dir.exists()
    assert cfg.reencoded_dir.exists()
    assert cfg.wav_dir.exists()


def test_manifest_construction_logic(temp_workspace):
    """Test pairing semantics and deterministic sampling in manifest generation."""
    cfg = PipelineConfig(
        root=temp_workspace,
        pairs=PairsConfig(train=2, val=1, test=1),
        seed=123,
    )

    # Mock list_repo_tree to return synthetic paths
    # train: gen1 has 3 pairs (clip1, clip2, clip3) and 1 unmatched real (clip4)
    # val: gen2 has 1 pair (clip1)
    mock_paths = [
        # train gen1 - real
        "train/real/gen1/c1.mp4",
        "train/real/gen1/c2.mp4",
        "train/real/gen1/c3.mp4",
        "train/real/gen1/c4_unmatched.mp4",
        # train gen1 - fake
        "train/fake/gen1/c1.mp4",
        "train/fake/gen1/c2.mp4",
        "train/fake/gen1/c3.mp4",
        "train/fake/gen1/c5_unmatched.mp4",
        # val gen2
        "val/real/gen2/v1.mp4",
        "val/fake/gen2/v1.mp4",
        # test gen3
        "test/real/gen3/t1.mp4",
        "test/fake/gen3/t1.mp4",
    ]

    mock_items = [MagicMock(path=p) for p in mock_paths]

    with patch("dataset_pipeline.manifest.HfApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api.list_repo_tree.return_value = mock_items
        mock_api_cls.return_value = mock_api

        df = build_manifest(cfg)

        # Expected:
        # train: min(2, 3) = 2 pairs = 4 rows (c4 and c5 not included)
        # val: 1 pair = 2 rows
        # test: 1 pair = 2 rows
        # Total: 8 rows
        assert len(df) == 8

        # Verify columns
        required_cols = {
            "split", "generator", "clip_id", "label", "label_name",
            "hf_path", "pair", "key", "status"
        }
        assert required_cols.issubset(df.columns)

        # Verify pairing: each pair should have exactly 1 real and 1 fake
        for pair_id, group in df.groupby("pair"):
            assert len(group) == 2
            assert set(group["label_name"]) == {"real", "fake"}

        # Verify deterministic sampling with same seed
        df2 = build_manifest(cfg)
        assert df.equals(df2)

        # Test save and load
        save_manifest(df, cfg.manifest_path)
        loaded = load_manifest(cfg.manifest_path)
        assert len(loaded) == len(df)


def test_local_preprocessing_and_resumability(synthetic_video, temp_workspace):
    """Test FFmpeg preprocessing, atomic creation, media validation, and resumability."""
    cfg = PipelineConfig(
        root=temp_workspace,
        workers=2,
        ffmpeg_threads=1,
        fps=25,
        video_size="512x512",
        duration=5.0,
        audio_sample_rate=16000,
    )
    cfg.make_dirs()

    # Place synthetic video into dataset_dir
    rel_path = "train/real/gen1/sample1.mp4"
    dest_src = cfg.dataset_dir / rel_path
    dest_src.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(synthetic_video, dest_src)

    row = pd.Series({
        "split": "train",
        "generator": "gen1",
        "clip_id": "sample1",
        "label": 0,
        "label_name": "real",
        "hf_path": rel_path,
        "source_path": str(dest_src),
        "key": "train/real/gen1/sample1",
        "status": "pending",
    })

    # First run: should preprocess and validate successfully
    res1 = preprocess_single_clip(row, cfg)
    assert res1["status"] == "complete"
    assert res1["error"] is None

    final_video = Path(res1["video_path"])
    final_audio = Path(res1["audio_path"])

    assert final_video.exists()
    assert final_audio.exists()
    # Check that temp files were cleanly replaced
    assert not final_video.with_suffix(".tmp.mp4").exists()
    assert not final_audio.with_suffix(".tmp.wav").exists()

    # Verify video properties using ffprobe
    v_probe = probe_media_file(final_video)
    v_stream = next(s for s in v_probe["streams"] if s["codec_type"] == "video")
    assert v_stream["width"] == 512
    assert v_stream["height"] == 512

    # Verify audio properties using ffprobe
    a_probe = probe_media_file(final_audio)
    a_stream = next(s for s in a_probe["streams"] if s["codec_type"] == "audio")
    assert int(a_stream["sample_rate"]) == 16000
    assert int(a_stream["channels"]) == 1

    # Record modification times
    mtime_v1 = final_video.stat().st_mtime
    mtime_a1 = final_audio.stat().st_mtime

    # Second run (Idempotency / Resumability):
    # Running again should skip re-encoding
    res2 = preprocess_single_clip(row, cfg)
    assert res2["status"] == "complete"
    assert final_video.stat().st_mtime == mtime_v1
    assert final_audio.stat().st_mtime == mtime_a1

    # Third run: Corrupt the video file, run again, verify it regenerates!
    final_video.write_bytes(b"corrupted video content")
    res3 = preprocess_single_clip(row, cfg)
    assert res3["status"] == "complete"
    # Should have regenerated a valid video
    is_valid, _ = validate_sample(final_video, final_audio, cfg)
    assert is_valid


def test_media_validation_failures(temp_workspace):
    """Test validation catches missing files, empty files, and incorrect specifications."""
    cfg = PipelineConfig(root=temp_workspace)

    # 1. Non-existent files
    is_valid, err = validate_sample(temp_workspace / "none.mp4", temp_workspace / "none.wav", cfg)
    assert not is_valid
    assert "does not exist" in err

    # 2. Empty video file
    empty_v = temp_workspace / "empty.mp4"
    empty_v.write_bytes(b"")
    empty_a = temp_workspace / "empty.wav"
    empty_a.write_bytes(b"RIFF" + b"\x00" * 40)
    is_valid, err = validate_sample(empty_v, empty_a, cfg)
    assert not is_valid
    assert "0 bytes" in err


def test_download_dataset_resilient_error_handling(temp_workspace):
    """Test download_dataset gracefully records failures without crashing."""
    cfg = PipelineConfig(root=temp_workspace, download_mode="selected")
    cfg.make_dirs()

    manifest = pd.DataFrame([{
        "split": "train",
        "generator": "gen1",
        "clip_id": "nonexistent_clip",
        "label": 0,
        "label_name": "real",
        "hf_path": "train/real/gen1/nonexistent_clip.mp4",
        "status": "pending",
        "error": "",
    }])

    with patch("dataset_pipeline.download.hf_hub_download") as mock_dl:
        mock_dl.side_effect = Exception("404 Client Error: Not Found")

        updated_manifest, failures = download_dataset(manifest, cfg)
        assert len(failures) == 1
        assert failures[0]["stage"] == "download"
        assert updated_manifest.loc[0, "status"] == "failed"
