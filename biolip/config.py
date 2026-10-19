from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class Config:
    seed: int = 42
    repo: str = "zaqxsw0526/MMDF"
    pairs: dict = field(default_factory=lambda: {"train": 200, "val": 60, "test": 100})
    fps: int = 25
    max_frames: int = 125
    win: int = 25
    stride: int = 5
    min_valid: float = 0.9
    max_gap: int = 5
    sg_win: int = 7
    sg_poly: int = 3
    cnn_seeds: tuple = (22, 100, 123)  # paper seeds (§III-F PolyGlotFake evaluation)
    cnn_epochs: int = 60
    cnn_patience: int = 30              # early-stopping patience on val video-AUC (§III-F)
    topk: float = 0.4
    aux_w: float = 0.3
    w_cnn: float = 0.5
    target_acc: float = 0.95
    root: Path = Path(os.environ.get("BIOLIP_ROOT", "./biolip_data"))
    download_mode: str = "selected"
    workers: int = min(4, os.cpu_count() or 1)
    ffmpeg_threads: int = 2
    video_size: str = "512x512"
    duration: float = 5.0
    audio_sample_rate: int = 16000

    @property
    def dataset_dir(self):
        return self.root / "raw"

    @property
    def reencoded(self):
        return self.root / "reencoded"

    @property
    def reencoded_dir(self):
        return self.reencoded

    @property
    def wavs(self):
        return self.root / "wav16k"

    @property
    def wav_dir(self):
        return self.wavs

    @property
    def manifest_path(self):
        return self.root / "manifest.parquet"

    @property
    def failures_path(self):
        return self.root / "failures.parquet"


    @property
    def landmarks(self):
        return self.root / "landmarks"

    @property
    def outputs(self):
        return self.root / "outputs"

    @property
    def task(self):
        return self.root / "face_landmarker.task"

    def make_dirs(self):
        for d in (self.root, self.reencoded, self.wavs, self.landmarks, self.outputs):
            d.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    return Config()
