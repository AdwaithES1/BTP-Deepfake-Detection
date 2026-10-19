# BioLip-style video branch

This project is a notebook-to-Python refactor of `biolip_implementation.ipynb`. The computation is split into data acquisition, landmark extraction, kinematic feature construction, model training, calibration/evaluation, and visualization.

## Pipeline

1. Select paired real/fake MMDF clips.
2. Download and re-encode every clip with the same ffmpeg recipe.
3. Extract 64 perioral landmarks + 8 rigid anchors with MediaPipe.
4. Fill short detection gaps and remove head pose with similarity alignment.
5. Build lip displacement/velocity/acceleration/jerk channels and geometry features.
6. Create 2-second windows.
7. Train a HistGradientBoosting model with leave-one-generator-out validation.
8. Train an ensemble of temporal CNNs and produce per-frame anomaly logits.
9. Calibrate both models on held-out-generator OOF predictions and form Real/Fake/Ambiguous bands.
10. Save clip scores, window scores, frame curves, features, metrics, and model checkpoints.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`ffmpeg` must also be installed and available on `PATH`.

Authenticate with Hugging Face before running, for example with `huggingface-cli login`, or use the normal Hugging Face environment/token mechanism.

Set the cache/output root if desired:

```bash
export BIOLIP_ROOT=/path/to/btp_biolip
```

Then run:

```bash
python scripts/run.py
```

The default settings intentionally mirror the notebook. Edit `biolip/config.py` for experiment-level settings.

## Output layout

`outputs/manifest.csv`, `scores_clip.csv`, `scores_window.csv`, `frame_curves.npz`, `features_window.csv.gz`, `metrics_basic.csv`, `window_channels.npy`, `channel_names.json`, `config.json`, and model checkpoints are generated under `BIOLIP_ROOT/outputs`.

## Notes

The original notebook was written for Google Colab and mounted Google Drive. This version removes the Colab-specific `drive.mount()` and stores all caches relative to `BIOLIP_ROOT`. The MediaPipe worker remains a separate Python process because the notebook explicitly used process isolation for MediaPipe stability.
