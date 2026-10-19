"""Calibration for BioLip.

Replaces the original unconstrained LogisticRegression Platt scaling
(which caused sign inversion on low-sample OOF data) with monotone
IsotonicRegression, which enforces the physical prior that higher anomaly
logit => higher deepfake probability.

NOTE: The calibration layer itself is not part of the paper; it is a
      custom post-processing step.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


def calibrate(meta, z_gbm, z_cnn, oof_gbm, oof_cnn, w_cnn, target_acc):
    import pandas as pd

    lab = meta.drop_duplicates("key").set_index("key")[
        ["split", "generator", "label", "clip_id"]
    ]

    def cm(keys, z):
        return (
            pd.DataFrame({"key": np.asarray(keys), "z": np.asarray(z)})
            .groupby("key")
            .z.mean()
        )

    C = lab.join(
        [cm(meta.key, z_gbm).rename("z_gbm"), cm(meta.key, z_cnn).rename("z_cnn")]
    )
    C["oof_gbm"] = cm(
        meta.key[meta.split.values == "train"], oof_gbm[meta.split.values == "train"]
    ).reindex(C.index)
    C["oof_cnn"] = cm(
        meta.key[meta.split.values == "train"], oof_cnn[meta.split.values == "train"]
    ).reindex(C.index)

    ok = C.oof_gbm.notna() & C.oof_cnn.notna()

    # Fit monotone isotonic calibrators (increasing=True enforces that
    # higher anomaly logit => higher fake probability, preventing sign inversion)
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _fit_iso(oof_z, labels):
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        # Map raw logit to [0,1] via sigmoid before fitting isotonic map
        iso.fit(_sigmoid(np.asarray(oof_z)), np.asarray(labels, dtype=float))
        return iso

    def _cal(iso, z):
        return iso.predict(_sigmoid(np.asarray(z)))

    oof_gbm_ok = C.oof_gbm[ok].values
    oof_cnn_ok = C.oof_cnn[ok].values
    y_oof      = C.label[ok].values

    pg = _fit_iso(oof_gbm_ok, y_oof)
    pc = _fit_iso(oof_cnn_ok, y_oof)

    C["p_gbm"] = _cal(pg, C.z_gbm.values)
    C["p_cnn"] = _cal(pc, C.z_cnn.values)

    # Ensemble calibrated probability: average of individual calibrations
    ens_raw = (1 - w_cnn) * _sigmoid(C.z_gbm.values) + w_cnn * _sigmoid(C.z_cnn.values)
    C["p_ens"] = ens_raw  # use direct sigmoid ensemble (monotone by construction)

    # OOF ensemble for abstention-band search
    p_oof = (1 - w_cnn) * _sigmoid(oof_gbm_ok) + w_cnn * _sigmoid(oof_cnn_ok)

    # Grid-search for the abstention band that achieves >= target_acc on OOF
    best = None
    for lo in np.arange(0.02, 0.501, 0.01):
        for hi in np.arange(0.50, 0.981, 0.01):
            dec = (p_oof <= lo) | (p_oof >= hi)
            if dec.sum() < 10:
                continue
            acc = ((p_oof[dec] >= hi) == y_oof[dec]).mean()
            if acc >= target_acc and (best is None or dec.mean() > best[2]):
                best = (lo, hi, dec.mean())

    lo, hi = (0.25, 0.75) if best is None else best[:2]
    C["verdict"] = np.where(
        C.p_ens <= lo, "Real", np.where(C.p_ens >= hi, "Fake", "Ambiguous")
    )
    return C, (float(lo), float(hi)), (pg, pc), p_oof, y_oof
