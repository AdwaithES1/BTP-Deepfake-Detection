# Audio-Visual Cross-Alignment for Multimodal Deepfake Detection

B.Tech Project (BTP) at the Indian Institute of Information Technology Kottayam.

Deepfakes can manipulate voice, facial motion, or both. Audio and video may each look realistic on their own while the relationship between them stays inconsistent. This project detects deepfakes by checking that relationship: **audio-visual cross-alignment**.

## Team

| Name | Roll No. |
|---|---|
| Abhinav Marlingaplar | 2023BCD0013 |
| Karthik P | 2023BCD0051 |
| Adwaith E.S | 2023BCD0058 |
| Arjun Manjunath | 2023BCD0067 |

**Guide:** Dr. Jayakrushna Sahoo

## Approach

An agreement-aware, three-stage cascade: cheap evidence first, expensive analysis only when needed.

1. **Stage 1: Lightweight anomaly detection.** Two independent unimodal detectors each output an anomaly trajectory over time.
   - Video: **BioLip** (biomechanical lip kinematics: displacement, velocity, acceleration, jerk)
   - Audio: **FlowFake** (liquid time-constant ODE networks over speech dynamics)
2. **Stage 2: Audio-visual cross-alignment.** If the Stage 1 verdicts conflict or are ambiguous, check whether audio and visual anomalies occur at related times and are coupled. This aligns anomaly-score trajectories, not raw embeddings.
3. **Stage 3: Selective generator-aware probing.** Clips still unresolved are escalated to an X-AVDT style probe (diffusion inversion, roughly 1 min/clip), so the expensive step runs rarely.

Full motivation, literature review and architecture diagram are in the [Review 1 slides](Submissions/Review_1/BTP-Review_1.pdf).

## Repository structure

```
Papers/        Reference papers (FlowFake, BioLip, ...)
Notebooks/     Experiment notebooks, one per component
Data/          Local datasets and features (git-ignored, see below)
Submissions/   Review deliverables
  Review_1/    Slides (done)
  Review_2/    Notebooks + slides (in progress)
```

## Progress

| Milestone | Scope | Status |
|---|---|---|
| Review 1 | Literature review, problem statement, methodology, architecture | Done |
| Review 2 | Data collection and prep, Stage 1 models (BioLip video, FlowFake audio), two anomaly trajectories | In progress |
| Review 3 | Stage 2 cross-alignment, Stage 3 selective X-AVDT probe, cascade, end-to-end test | Planned |
| Final | Fine-tuning, cross-dataset evaluation, ablations, documentation and report | Planned |

### Review 2 deliverables

| Deliverable | Location | Status |
|---|---|---|
| BioLip notebook, Part 1 | `Notebooks/` | Done |
| FlowFake notebook | `Notebooks/` | In progress (teammates) |
| Review 2 PPT | `Submissions/Review_2/` | To be made at the end |

## Datasets

Candidates (check each upstream page for access requirements, terms of use and download instructions):

- [MMDF](https://huggingface.co/datasets/zaqxsw0526/MMDF/tree/main)
- [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb/tree/main/dataset)
- [FaceForensics](https://github.com/ondyari/FaceForensics/tree/master)

Download them into `Data/`. Its contents are git-ignored, so never commit datasets, media files or model weights.

## Key references

Full list is in the Review 1 slides.

- FlowFake: Liquid Networks for Audio Deepfake Detection (2026)
- BioLip: Language-Generalizable Lip-Sync Deepfake Detection via Biomechanical Constraint Violation Modeling (2026)
- AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection (CVPR 2024)
- X-AVDT: Audio-Visual Cross-Attention for Robust Deepfake Detection (CVPR 2026). The PDF is git-ignored; keep your own local copy in `Papers/`.

## Contributing

- Work on a branch (`git checkout -b yourname/short-topic`) and open a pull request rather than pushing to `main`.
- Name notebooks `<Component>_<Part>.ipynb`, for example `BioLip_Part1.ipynb`, and put a short purpose and inputs/outputs note in the first cell.
- Clear heavy outputs before committing notebooks, and keep file paths relative to the repo root (for example `Data/...`).
- Add new papers to `Papers/` and finished review material to the matching `Submissions/Review_N/` folder.