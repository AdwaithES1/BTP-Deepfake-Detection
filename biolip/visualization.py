import numpy as np
import matplotlib.pyplot as plt


def make_curves(meta, clip_ch, frame_logits, sig, win):
    curves = {}
    for k, g in meta.groupby("key"):
        T = len(clip_ch[k])
        s = np.zeros(T)
        n = np.zeros(T)
        for i, st in zip(g.index, g.start):
            s[st : st + win] += sig(frame_logits[i])
            n[st : st + win] += 1
        curves[k] = np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)
    return curves


def plot_pair(pair, manifest, clip_ch, curves, ch):
    sp, gn, cid = pair.split("/")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    for nm in ("real", "fake"):
        k = f"{sp}/{nm}/{gn}/{cid}"
        t = np.arange(len(clip_ch[k])) / 25
        ax[0].plot(t, clip_ch[k][:, ch["lo_inner_vel"]], label=nm)
        ax[1].plot(t, curves[k], label=nm)
    ax[0].set_title(f"lower-inner-lip log-velocity ({gn})")
    ax[1].set_title("CNN per-frame anomaly score")
    ax[1].set_ylim(0, 1)
    for a in ax:
        a.set_xlabel("time (s)")
        a.legend()
    fig.tight_layout()
    return fig
