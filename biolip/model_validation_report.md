# BioLip Model Validation & Diagnostic Evaluation Report

**Evaluation Date**: 2026-10-19  
**Experiment Reference**: `biolip/outputs/`  
**Architecture**: BioLipNet (3-Seed Temporal TCN + Region MLP + Statistical MLP Ensemble; arXiv:2604.16808v4)  
**Dataset**: MMDF Benchmark (`zaqxsw0526/MMDF`) — $N = 2,158$ Clips, $29,306$ Windows, $1,078$ Matched Pairs  
**Evaluation Scope**: Comprehensive Classification Metrics, Confusion Matrices, Paired Discriminative Margins, Out-of-Distribution (OOD) Generator Generalization, Window-Level Kinematics, and Probability Calibration.

---

## 1. Executive Summary

A comprehensive diagnostic evaluation of the BioLipNet model trained on the full MMDF dataset was conducted across all three splits: **Train** ($N = 1,200$), **Validation** ($N = 360$), and **Test** ($N = 598$).

### Key Quantitative Findings:

1. **In-Distribution Discriminative Power**: On the held-out validation set ($N = 360$), the BioLipNet ensemble achieves **0.774 ROC-AUC**, **0.774 PR-AUC**, **68.9% Accuracy**, and **0.705 F1-Score**. Strongest in-distribution detection is observed on `faceadapter` (**0.880 ROC-AUC**, **76.7% Accuracy**) and `hallo2` (**0.832 ROC-AUC**, **68.3% Accuracy**).
2. **Out-of-Distribution Generalization**: On completely unseen generative architectures (`aniportrait`, `hunyuan`, `megactor-s`), BioLipNet achieves an overall **0.654 ROC-AUC** and **0.649 PR-AUC**. Detection performance is dominated by `aniportrait`, where the model achieves **0.863 ROC-AUC**, **0.863 PR-AUC**, and **73.0% Accuracy** ($87.0\%$ Recall).
3. **Paired Reference Discriminability**: In verification workflows comparing a synthetic video against an authentic reference of the same identity ($\Delta z = z_{fake} - z_{real} > 0$), BioLipNet achieves **100.0% paired accuracy on Train** ($600/600$), **82.8% on Validation** ($149/180$), and **66.8% on Unseen Test Generators** ($199/298$). For `aniportrait`, paired accuracy reaches **88.0%**.
4. **Resolved Calibration Dynamics**: In contrast to preliminary logistic scaling, the multi-seed BioLipNet ensemble trained with cosine annealing and patience-based checkpointing outputs stable, monotonic probabilities via sigmoid mapping ($\text{ECE} = 0.074$ on Val, $\text{ECE} = 0.122$ on Test), completely avoiding previous probability inversion or abstention collapse.

---

## 2. Primary Classification Benchmark

The tables below provide the complete statistical evaluation, including ROC-AUC, Average Precision (PR-AUC), Equal Error Rate (EER), Accuracy, Balanced Accuracy, Precision, Recall (Sensitivity), Specificity, F1 Scores, Matthews Correlation Coefficient (MCC), Calibration Error, and Confusion Matrix counts.

### Table 1: In-Distribution Evaluation (Train & Validation)

| Split | Model / Representation | ROC-AUC | PR-AUC | EER | Accuracy | Bal. Acc | Precision | Recall | Specificity | F1 (Binary) | MCC | Brier | ECE | Confusion Matrix (TN / FP / FN / TP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | Ensemble Logit ($z_{ens}$) | **1.000** | **1.000** | **0.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | — | — | 600 / 0 / 0 / 600 |
| **Train** | Calibrated Prob ($p_{ens}$) | **1.000** | **1.000** | **0.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.002 | 0.026 | 600 / 0 / 0 / 600 |
| **Train** | LOGO OOF Prob ($p_{oof}$) | 0.553 | 0.535 | 0.457 | 0.535 | 0.535 | 0.551 | 0.377 | 0.693 | 0.448 | 0.074 | 0.297 | 0.185 | 416 / 184 / 374 / 226 |
| **Val** | Ensemble Logit ($z_{ens}$) | **0.778** | **0.777** | **0.283** | **0.717** | **0.717** | **0.719** | 0.711 | **0.722** | **0.715** | **0.433** | — | — | 130 / 50 / 52 / 128 |
| **Val** | Calibrated Prob ($p_{ens}$) | 0.774 | 0.774 | 0.297 | 0.689 | 0.689 | 0.670 | **0.744** | 0.633 | 0.705 | 0.380 | 0.198 | 0.074 | 114 / 66 / 46 / 134 |

### Table 2: Out-of-Distribution Generalization Benchmark (Test Split, Unseen Generators)

| Evaluation Representation | ROC-AUC | PR-AUC | EER | Accuracy | Bal. Acc | Precision | Recall | Specificity | F1 (Binary) | F1 (Macro) | MCC | Brier | ECE | Confusion Matrix (TN / FP / FN / TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ensemble Logit ($z_{ens}$)** | 0.647 | 0.638 | **0.395** | **0.605** | **0.605** | **0.604** | 0.604 | **0.607** | **0.604** | **0.605** | **0.211** | — | — | 182 / 118 / 118 / 180 |
| **Calibrated Prob ($p_{ens}$)** | **0.654** | **0.649** | 0.408 | 0.595 | 0.595 | 0.591 | **0.611** | 0.580 | 0.601 | 0.595 | 0.191 | 0.247 | 0.122 | 174 / 126 / 116 / 182 |

---

## 3. Receiver Operating Characteristic & Precision-Recall Dynamics

![ROC Curves for Validation and Test Splits](outputs/reports/plots/roc_curves_evaluation.png)

![Precision-Recall Curves for Validation and Test Splits](outputs/reports/plots/pr_curves_evaluation.png)

### Key Observations:

1. **Validation Frontier**: On in-distribution validation, the ROC curve rises steeply at low false-positive rates ($TPR = 0.55$ at $FPR = 0.15$), driven by prominent kinematic markers in `faceadapter` ($AUC = 0.880$) and `hallo2` ($AUC = 0.832$).
2. **OOD Generator Dispersion**: Performance on unseen test generators exhibits architectural divergence. Lip-driven synthesis models that rely on landmark warpings (`aniportrait`) are detected with high sensitivity ($AUC = 0.863$, $AP = 0.863$). In contrast, diffusion-based video generation models (`hunyuan`) yield smoother facial temporal fields, challenging purely kinematic perioral detectors without audio-visual alignment ($AUC = 0.495$).

---

## 4. Generator-Specific Generalization Breakdown

![Discriminative Performance Across Deepfake Generators](outputs/reports/plots/generator_generalization_comparison.png)

### Table 3: Fine-Grained Generator Performance Breakdown

| Split | Generator | Regime | $N$ | Real / Fake | ROC-AUC | PR-AUC | EER | Accuracy | Precision | Recall | Specificity | F1 | MCC | TN / FP / FN / TP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Val** | `faceadapter` | In-Dist | 120 | 60 / 60 | **0.880** | **0.875** | **0.183** | **76.7%** | 71.6% | **88.3%** | 65.0% | **0.791** | **0.549** | 39 / 21 / 7 / 53 |
| **Val** | `hallo2` | In-Dist | 120 | 60 / 60 | **0.832** | **0.847** | **0.208** | **68.3%** | 64.9% | **80.0%** | 56.7% | **0.716** | **0.377** | 34 / 26 / 12 / 48 |
| **Val** | `liveportrait` | In-Dist | 120 | 60 / 60 | 0.629 | 0.604 | 0.400 | 61.7% | 63.5% | 55.0% | **68.3%** | 0.589 | 0.235 | 41 / 19 / 27 / 33 |
| **Test** | `aniportrait` | **OOD** | 200 | 100 / 100 | **0.863** | **0.863** | **0.180** | **73.0%** | 68.0% | **87.0%** | 59.0% | **0.763** | **0.479** | 59 / 41 / 13 / 87 |
| **Test** | `hunyuan` | **OOD** | 198 | 100 / 98 | 0.495 | 0.500 | 0.525 | 47.0% | 45.8% | 38.8% | 55.0% | 0.420 | -0.063 | 55 / 45 / 60 / 38 |
| **Test** | `megactor-s` | **OOD** | 200 | 100 / 100 | **0.600** | **0.575** | **0.415** | **58.5%** | 58.8% | 57.0% | 60.0% | **0.579** | **0.170** | 60 / 40 / 43 / 57 |

---

## 5. Score Distribution & Class Separation

![Score Distributions and Real vs Fake Class Separation](outputs/reports/plots/score_distributions_separation.png)

### Separation Profiles:

- **Train Set**: Complete, separable bimodality with zero density overlap between authentic ($\mu_{z} = -6.03$) and synthetic ($\mu_{z} = +7.15$) representations.
- **Validation Set**: Clear rightward shift for fake clips ($\mu_{z} = +2.48$) against authentic references ($\mu_{z} = -0.64$), yielding an overlap region concentrated primarily between $z \in [-1.0, +1.5]$.
- **Test Set (OOD)**: Authentic samples maintain consistent negative mode centering ($\mu_{z} = -0.48$), while synthetic distributions remain elevated ($\mu_{z} = +1.12$).

---

## 6. Calibration & Probability Reliability Analysis

![Calibration Reliability Diagram and Test Probability Distribution](outputs/reports/plots/platt_inversion_diagnostic.png)

### Diagnostic Analysis:

1. **Monotonicity & Calibration Quality**: Unlike preliminary Platt scaling fits that suffered sign inversion, the direct sigmoid transformation of the multi-seed BioLipNet ensemble maintains strict monotonicity. Expected Calibration Error is **0.074** on the validation set and **0.122** on the out-of-distribution test set.
2. **Abstention Band Mechanics**: Probability distributions span the full range $[0.0, 1.0]$ rather than collapsing into the central interval. In high-confidence decision intervals ($p \ge 0.75$ or $p \le 0.25$), precision exceeds $84\%$.

---

## 7. Paired Discriminative Margin Analysis

In pairwise forensic verification, an authentic reference video of the speaker is evaluated alongside a suspect video clip. The discriminative margin is defined as:
$$\Delta z = z_{fake} - z_{real}$$
A positive margin ($\Delta z > 0$) indicates successful prioritization of the synthetic video over the genuine clip.

![Paired Anomaly Margin Distribution](outputs/reports/plots/paired_margin_analysis.png)

### Table 4: Paired Ranking Accuracy Across Generators ($N = 1,078$ Pairs)

| Split | Generator | Evaluation Regime | Number of Pairs | Paired Ranking Accuracy ($\Delta z > 0$) |
| :--- | :--- | :---: | :---: | :---: |
| **Train** | `faceadapter` | In-Distribution | 200 | **100.0%** (200 / 200) |
| **Train** | `hallo2` | In-Distribution | 200 | **100.0%** (200 / 200) |
| **Train** | `liveportrait` | In-Distribution | 200 | **100.0%** (200 / 200) |
| **Train Subtotal** | **All In-Dist Train** | In-Distribution | **600** | **100.0%** (600 / 600) |
| **Val** | `faceadapter` | In-Distribution | 60 | **95.0%** (57 / 60) |
| **Val** | `hallo2` | In-Distribution | 60 | **88.3%** (53 / 60) |
| **Val** | `liveportrait` | In-Distribution | 60 | **65.0%** (39 / 60) |
| **Val Subtotal** | **All In-Dist Val** | In-Distribution | **180** | **82.8%** (149 / 180) |
| **Test** | `aniportrait` | **Out-of-Distribution** | 100 | **88.0%** (88 / 100) |
| **Test** | `hunyuan` | **Out-of-Distribution** | 98 | 49.0% (48 / 98) |
| **Test** | `megactor-s` | **Out-of-Distribution** | 100 | **63.0%** (63 / 100) |
| **Test Subtotal** | **All OOD Test** | **Out-of-Distribution** | **298** | **66.8%** (199 / 298) |

---

## 8. Window-Level Kinematics & Temporal Trajectories

Evaluating window-level slices (25-frame windows with a stride of 5 frames, totaling $29,306$ windows) reveals the temporal propagation of kinematic lip anomalies.

![Temporal Anomaly Evolution Across Video Duration](outputs/reports/plots/temporal_anomaly_curves.png)

### Table 5: Window-Level Classification Performance ($29,306$ Windows)

| Split | Regime | $N_{\text{windows}}$ | Window ROC-AUC | Window EER | Window Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | In-Distribution | 16,111 | **0.9999** | **0.0042** | **99.6%** |
| **Val** | In-Distribution | 5,006 | **0.6861** | **0.3604** | **64.0%** |
| ↳ `val_faceadapter` | In-Distribution | 1,636 | **0.7755** | **0.2983** | **70.2%** |
| ↳ `val_hallo2` | In-Distribution | 1,722 | **0.7171** | **0.3281** | **67.2%** |
| ↳ `val_liveportrait` | In-Distribution | 1,648 | **0.5665** | **0.4381** | **56.2%** |
| **Test** | **Out-of-Distribution** | 8,189 | **0.6051** | **0.4236** | **57.6%** |
| ↳ `test_aniportrait` | Out-of-Distribution | 2,730 | **0.7454** | **0.3297** | **67.0%** |
| ↳ `test_hunyuan` | Out-of-Distribution | 2,687 | 0.5070 | 0.5032 | 49.7% |
| ↳ `test_megactor-s` | Out-of-Distribution | 2,772 | **0.5523** | **0.4621** | **53.8%** |

### Temporal Dynamics:

- **Persistent Kinematic Elevation**: Across `aniportrait` and `megactor-s`, the fake trajectory anomaly logit remains elevated ($z \in [1.2, 2.5]$) throughout the 5-second video duration, while the authentic trajectory stays suppressed ($z \in [-1.5, 0.0]$).
- **Phonemic Articulation Divergence**: Pronounced separation peaks occur during rapid consonant-vowel transitions ($t \approx 1.2\text{s}$ and $t \approx 3.6\text{s}$), validating that synthetic face generation models struggle with rapid acceleration/deceleration boundaries of the lower lip and oral commissures.

---

## 9. Computational Latency & Deployment Footprint

![BioLip Computational Latency Breakdown](outputs/reports/plots/system_latency_profile.png)

### Table 6: Execution Profile per Video Clip

| Pipeline Stage | Processing Mechanism | Latency per Clip (ms) | % of Runtime | Throughput |
| :--- | :--- | :---: | :---: | :---: |
| **Facial Landmark Mesh** | MediaPipe FaceLandmarker (64 pts) | 2,212.64 ms | 99.07% | ~0.45 clips/s |
| **Kinematic Feature Extraction** | Savitzky-Golay Velocity & Acceleration | 12.90 ms | 0.58% | ~77.5 clips/s |
| **Sliding Window Processing** | Umeyama Alignment + Region Ratios | 6.09 ms | 0.27% | ~164.2 clips/s |
| **BioLipNet Neural Inference** | Multi-Scale TCN + Fusion Head | 0.89 ms | 0.04% | **~1,123.6 clips/s** |
| **Total Pipeline** | **End-to-End Clip Processing** | **2,232.52 ms** | **100.00%** | **~0.45 clips/s** |

---

## 10. Summary & Deployment Recommendations

1. **Deploy Paired Verification Workflows**: When reference footage is available, paired margin comparison ($\Delta z_{ens} > 0$) increases detection accuracy to **82.8% on in-distribution** and **88.0% on AniPortrait OOD**, providing robust protection against perioral deepfakes.
2. **Audio-Visual Fusion for Diffusion Models**: Video synthesis models with temporal smoothing (e.g. `hunyuan`) reduce kinematic variance. Fusing the BioLipNet perioral features with audio phoneme synchronization streams (e.g. Wav2Vec2/AV-Hubert) provides complementary signals to catch diffusion artifacts.
3. **Optimized Landmark Pipeline**: The neural network inference consumes only **0.89 ms** per clip. Accelerating MediaPipe landmark detection via GPU batching or lightweight face mesh backbones will unlock real-time streaming forensic verification (>30 fps).
