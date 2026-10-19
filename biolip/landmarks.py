import json, os, subprocess, sys, time, urllib.request
from pathlib import Path
import numpy as np

WORKER = r"""
import sys, os, json, time
os.environ["GLOG_minloglevel"] = "2"
import numpy as np, cv2, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
cfg = json.load(open(sys.argv[1])); tasks = json.load(open(sys.argv[2]))
for src, dst in tasks:
    if os.path.exists(dst): continue
    try:
        t0 = time.perf_counter()
        fm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=cfg["task"]), running_mode=vision.RunningMode.VIDEO, num_faces=1))
        cap = cv2.VideoCapture(src); L, A, V = [], [], []
        while len(V) < cfg["max_frames"]:
            ok, fr = cap.read()
            if not ok: break
            rgb = np.ascontiguousarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); H, W = rgb.shape[:2]
            res = fm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(len(V) * 1000 / cfg["fps"]))
            if res.face_landmarks:
                p = np.array([[q.x, q.y, q.z] for q in res.face_landmarks[0]], np.float32) * np.array([W, H, W], np.float32)
                L.append(p[cfg["peri"]]); A.append(p[cfg["anc"]]); V.append(True)
            else:
                L.append(np.full((64, 3), np.nan, np.float32)); A.append(np.full((len(cfg["anc"]), 3), np.nan, np.float32)); V.append(False)
        cap.release(); fm.close()
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, lm=np.array(L), anc=np.array(A), valid=np.array(V, bool), t=time.perf_counter()-t0)
        os.replace(tmp, dst)
    except Exception as e:
        print("ERROR", src, repr(e), flush=True)
"""


def prepare_worker(cfg, perioral, anchors):
    if not cfg.task.exists():
        cfg.task.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            str(cfg.task),
        )
    worker = cfg.root / "worker.py"
    worker.write_text(WORKER)
    (cfg.root / "cfg.json").write_text(
        json.dumps(
            dict(
                task=str(cfg.task),
                fps=cfg.fps,
                max_frames=cfg.max_frames,
                peri=perioral,
                anc=anchors,
            )
        )
    )
    return worker


def extract_landmarks(manifest, cfg, perioral, anchors):
    prepare_worker(cfg, perioral, anchors)
    manifest["lm_path"] = [str(cfg.landmarks / f"{k}.npz") for k in manifest.key]
    jobs = list(zip(manifest.reenc_path, manifest.lm_path))
    nproc = min(os.cpu_count() or 1, 4)
    for attempt in range(3):
        todo = [j for j in jobs if not os.path.exists(j[1])]
        if not todo:
            break
        procs = []
        for i in range(nproc):
            chunk = todo[i::nproc]
            if not chunk:
                continue
            jf = cfg.root / f"jobs_{i}.json"
            jf.write_text(json.dumps(chunk))
            log = open(cfg.root / f"worker_{i}.log", "w")
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(cfg.root / "worker.py"),
                        str(cfg.root / "cfg.json"),
                        str(jf),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        while any(p.poll() is None for p in procs):
            time.sleep(3)
    return manifest


def quality_filter(manifest, cfg):
    info = []
    for r in manifest.itertuples():
        if not os.path.exists(r.lm_path):
            info.append((np.nan, np.nan, np.nan))
            continue
        d = np.load(r.lm_path)
        info.append((len(d["valid"]), d["valid"].mean(), float(d["t"])))
    manifest[["n_frames", "det_rate", "t_extract"]] = np.asarray(info)
    return manifest[
        (manifest.n_frames >= cfg.win) & (manifest.det_rate >= 0.5)
    ].reset_index(drop=True)
