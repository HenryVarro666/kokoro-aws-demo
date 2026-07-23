#!/usr/bin/env python3
"""Render the 3-panel Graviton vocoder A/B figure used in the README (docs/vocoder_ab.png).

Usage:
    python make_spectral_ab.py golden.wav corrupted.wav fixed.wav [out.png]

- golden.wav    : Apple-Silicon (aarch64) local run — the golden reference.
- corrupted.wav : Graviton with torch's oneDNN(ACL) CPU backend enabled — audibly
                  degraded (~8 dB log-mel spectral distance vs golden, -6 dB level),
                  INDEPENDENT of thread count on torch 2.12.0.
- fixed.wav     : Graviton with torch.backends.mkldnn.enabled=False — restores
                  golden-matching audio (~0.8 dB = run-to-run noise floor).

The A/B axis is the oneDNN backend on/off, NOT single- vs multi-thread. Prints each
panel's log-spectral distance to the golden reference, computed live.
Deps (in training/.venv): librosa, matplotlib, numpy.
"""
import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

SR = 24000  # Kokoro output rate


def logmel_db(path):
    y, _ = librosa.load(path, sr=SR)
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=256, n_mels=128)
    return librosa.power_to_db(S, ref=np.max)


def dist_db(ref_db, other_db):
    n = min(ref_db.shape[1], other_db.shape[1])
    return float(np.sqrt(np.mean((ref_db[:, :n] - other_db[:, :n]) ** 2)))


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    golden, corrupted, fixed = sys.argv[1:4]
    out = sys.argv[4] if len(sys.argv) > 4 else "docs/vocoder_ab.png"

    g, c, f = logmel_db(golden), logmel_db(corrupted), logmel_db(fixed)
    dc, df = dist_db(g, c), dist_db(g, f)
    print(f"corrupted vs golden: {dc:.2f} dB   fixed vs golden: {df:.2f} dB")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, S_db, title in (
        (axes[0], g, "① Golden — Apple M4 (local)"),
        (axes[1], c, f"② Graviton + oneDNN — corrupted (Δ {dc:.1f} dB)"),
        (axes[2], f, f"③ Graviton, mkldnn disabled — fixed (Δ {df:.1f} dB)"),
    ):
        img = librosa.display.specshow(S_db, sr=SR, hop_length=256,
                                       x_axis="time", y_axis="mel", ax=ax, cmap="magma")
        ax.set_title(title, fontsize=11)
    fig.colorbar(img, ax=axes, format="%+2.0f dB", location="right", shrink=0.85)
    fig.suptitle("Kokoro vocoder on Graviton: the oneDNN/ACL corruption and its fix "
                 "(torch.backends.mkldnn.enabled=False)", fontweight="bold", fontsize=12)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
