import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def evaluate(C):
    def metrics(d, col):
        y = d.label.values
        if len(np.unique(y)) < 2:
            return dict(n=len(d), AUC=np.nan, ACC=np.nan, EER=np.nan)
        return dict(
            n=len(d),
            AUC=roc_auc_score(y, d[col]),
            ACC=((d[col] >= 0.5) == y).mean(),
            EER=eer(y, d[col]),
        )

    R = pd.DataFrame(
        [
            dict(split=sp, generator=gn, model=col, **metrics(d, col))
            for sp in ("val", "test")
            for gn, d in [("ALL", C[C.split == sp])]
            + list(C[C.split == sp].groupby("generator"))
            for col in ("p_gbm", "p_cnn", "p_ens")
        ]
    )
    share = (
        C[C.split.isin(["val", "test"])]
        .groupby(["split", "verdict"])
        .size()
        .unstack(fill_value=0)
    )
    return R, share
