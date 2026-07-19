"""
Predictive-uncertainty figure via MC-dropout ensembling (+ small input jitter).
Runs N stochastic forward passes on a training sequence, then plots the ensemble
mean forecast and the per-pixel standard deviation (uncertainty), which
concentrates along evolving cloud edges.

Output: results/fig_uncertainty.png  and  results/fig_uncertainty_calibration.png
"""
import os, glob, warnings
import numpy as np
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")
from data_split import chrono_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pidm_model import PIConditionalForecaster

CACHE, BEST, OUT = "cache_128", "pidm_best.pth", "results"
BANDS = ["TIR1", "TIR2", "WV", "VIS", "MIR", "SWIR"]
VIS, T, N = 3, 3, 20


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(BEST, map_location=device, weights_only=False)
    a = ck.get("args", {})
    model = PIConditionalForecaster(base=a.get("base", 64),
                                    refine_steps=a.get("refine", 2)).to(device)
    model.load_state_dict(ck["model"]); model.eval(); enable_mc_dropout(model)

    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
    tr, _, _ = chrono_split(len(files))
    idx = tr[len(tr) // 2]
    d = torch.load(files[idx], map_location="cpu", weights_only=False)
    inp, tgt = d["input"].float(), d["target"].float()

    preds = []
    with torch.no_grad():
        for _ in range(N):
            x = inp.unsqueeze(0).to(device)
            x = x + 0.01 * torch.randn_like(x)          # small input jitter
            preds.append(model(x).cpu().numpy()[0])
    preds = np.stack(preds)                              # [N,T,C,H,W]
    mean, std = preds.mean(0), preds.std(0)
    gt = tgt.numpy()

    fig, ax = plt.subplots(3, T, figsize=(4 * T, 11))
    for t in range(T):
        ax[0, t].imshow(mean[t, VIS], cmap="gray", vmin=0, vmax=1)
        ax[0, t].set_title(f"Ensemble mean T+{t+1}")
        im = ax[1, t].imshow(std[t, VIS], cmap="magma")
        ax[1, t].set_title(f"Uncertainty (σ) T+{t+1}")
        plt.colorbar(im, ax=ax[1, t], fraction=0.046)
        ax[2, t].imshow(gt[t, VIS], cmap="gray", vmin=0, vmax=1)
        ax[2, t].set_title(f"Ground truth T+{t+1}")
        for r in range(3): ax[r, t].axis("off")
    fig.suptitle(f"Predictive uncertainty via MC-dropout (N={N}, band {BANDS[VIS]}, "
                 f"training example #{idx})", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_uncertainty.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # calibration: does higher σ co-locate with higher error?
    err = np.abs(mean - gt)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    s = std[:, VIS].ravel(); e = err[:, VIS].ravel()
    sel = np.random.RandomState(0).choice(len(s), size=min(4000, len(s)), replace=False)
    ax2.scatter(s[sel], e[sel], s=4, alpha=0.25, color="#7a4fa3")
    if s.std() > 1e-6:
        r = np.corrcoef(s, e)[0, 1]
        ax2.set_title(f"Uncertainty calibration (Pearson r={r:.2f})", fontweight="bold")
    ax2.set_xlabel("predicted σ (uncertainty)"); ax2.set_ylabel("|error|")
    fig2.tight_layout(); fig2.savefig(f"{OUT}/fig_uncertainty_calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"[uncertainty] wrote fig_uncertainty.png and calibration "
          f"(mean sigma={std[:,VIS].mean():.4f})")


if __name__ == "__main__":
    main()
