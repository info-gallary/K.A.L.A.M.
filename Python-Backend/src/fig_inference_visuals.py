"""
Inference / qualitative visualisations (uses TRAINING-set images, as requested —
illustrative of the model mechanism; headline metrics remain on the held-out
test set in evaluate_and_plot.py).

Produces, for a few training sequences:
  results/fig_diffusion_journey_*.png  input stack -> persistence prior ->
                                        coarse+residual -> refine steps -> outcome
  results/fig_qualitative_train_*.png   persistence / GT / prediction / error
  results/fig_rgb_composite_train_*.png false-colour composite
  results/fig_temporal_strip_train_*.png full observed+forecast strip
"""
import os, glob, warnings
import numpy as np
import torch
from torch.utils.data import Dataset
warnings.filterwarnings("ignore")
from data_split import chrono_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pidm_model import PIConditionalForecaster
try:
    from skimage.metrics import structural_similarity as sk_ssim
    HAVE_SK = True
except Exception:
    HAVE_SK = False

CACHE, BEST, OUT = "cache_128", "pidm_best.pth", "results"
BANDS = ["TIR1", "TIR2", "WV", "VIS", "MIR", "SWIR"]
VIS = 3
T = 3


def ssim_one(p, g):
    p = np.clip(p, 0, 1).astype(np.float64); g = np.clip(g, 0, 1).astype(np.float64)
    if HAVE_SK:
        return float(sk_ssim(g, p, data_range=1.0, gaussian_weights=True,
                             win_size=11, use_sample_covariance=False))
    return float(1 - np.mean((p - g) ** 2))


def load_model(device):
    ck = torch.load(BEST, map_location=device, weights_only=False)
    a = ck.get("args", {})
    m = PIConditionalForecaster(base=a.get("base", 64),
                                refine_steps=a.get("refine", 2)).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck.get("epoch", "?")


def pick_train_samples(k=3):
    # held-out TEST samples (camera-ready: qualitative figures on the test split)
    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
    _, _, te = chrono_split(len(files))
    idxs = [te[int(p * (len(te) - 1))] for p in (0.15, 0.5, 0.82)][:k]
    out = []
    for i in idxs:
        d = torch.load(files[i], map_location="cpu", weights_only=False)
        out.append((i, d["input"].float(), d["target"].float()))
    return out


def diffusion_journey(model, idx, inp, tgt, device, k):
    """input stack (top) + per-lead-time journey grid (stages + GT)."""
    with torch.no_grad():
        pred, journey = model(inp.unsqueeze(0).to(device), return_journey=True)
    stages = [(lbl, s[0].cpu().numpy()) for lbl, s in journey]   # each [T,C,H,W]
    gt = tgt.numpy()
    inp_np = inp.numpy()

    n_stage = len(stages)
    ncol = n_stage + 1                      # stages + GT
    fig = plt.figure(figsize=(3.0 * ncol, 3.0 * (T + 1) + 1.2))
    gs = gridspec.GridSpec(T + 1, ncol, figure=fig, hspace=0.35, wspace=0.08)

    # --- top strip: input stack (4 observed frames) ---
    for j in range(4):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(inp_np[j, VIS], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Input t{j-3}" if j < 3 else "Input t0 (last)", fontsize=11)
        ax.axis("off")
    ax = fig.add_subplot(gs[0, ncol - 1])
    ax.text(0.5, 0.5, "INPUT\nSTACK\n(4×6 bands)\n→ forecast", ha="center",
            va="center", fontsize=12, fontweight="bold")
    ax.axis("off")

    # --- journey grid: rows = lead times, cols = stages then GT ---
    for t in range(T):
        for s, (lbl, arr) in enumerate(stages):
            ax = fig.add_subplot(gs[t + 1, s])
            ax.imshow(arr[t, VIS], cmap="gray", vmin=0, vmax=1)
            sv = ssim_one(arr[t, VIS], gt[t, VIS])
            ax.set_title(f"{lbl}\nT+{t+1}  SSIM {sv:.3f}", fontsize=10)
            ax.axis("off")
        ax = fig.add_subplot(gs[t + 1, ncol - 1])
        ax.imshow(gt[t, VIS], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Ground truth\nT+{t+1}", fontsize=10, color="#1a7a1a")
        ax.axis("off")

    fig.suptitle(f"Conditional-Diffusion Forecast Journey — band {BANDS[VIS]}  "
                 f"(test example #{idx}, illustrative)",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.savefig(f"{OUT}/fig_diffusion_journey_{k}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote fig_diffusion_journey_{k}.png")


def qualitative(model, idx, inp, tgt, device, k):
    with torch.no_grad():
        pred = model(inp.unsqueeze(0).to(device)).cpu().numpy()[0]
    gt = tgt.numpy(); last = inp.numpy()[-1]
    fig, ax = plt.subplots(4, T, figsize=(4 * T, 15))
    for t in range(T):
        ax[0, t].imshow(last[VIS], cmap="gray", vmin=0, vmax=1); ax[0, t].set_title(f"Persistence→T+{t+1}")
        ax[1, t].imshow(gt[t, VIS], cmap="gray", vmin=0, vmax=1); ax[1, t].set_title(f"Ground truth T+{t+1}")
        sv = ssim_one(pred[t, VIS], gt[t, VIS])
        ax[2, t].imshow(pred[t, VIS], cmap="gray", vmin=0, vmax=1); ax[2, t].set_title(f"Ours T+{t+1} (SSIM {sv:.3f})")
        ax[3, t].imshow(np.abs(pred[t, VIS]-gt[t, VIS]), cmap="inferno", vmin=0, vmax=0.5); ax[3, t].set_title(f"|error| T+{t+1}")
        for r in range(4): ax[r, t].axis("off")
    fig.suptitle(f"Qualitative forecast (test example #{idx}, band {BANDS[VIS]})",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_qualitative_train_{k}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote fig_qualitative_train_{k}.png")


def rgb_composite(model, idx, inp, tgt, device, k):
    with torch.no_grad():
        pred = model(inp.unsqueeze(0).to(device)).cpu().numpy()[0]
    gt = tgt.numpy(); last = inp.numpy()[-1]; ci = [3, 5, 1]
    comp = lambda a: np.clip(np.stack([a[i] for i in ci], -1), 0, 1)
    fig, ax = plt.subplots(3, T, figsize=(4 * T, 11))
    for t in range(T):
        ax[0, t].imshow(comp(last)); ax[0, t].set_title(f"Persistence→T+{t+1}")
        ax[1, t].imshow(comp(gt[t])); ax[1, t].set_title(f"Ground truth T+{t+1}")
        ax[2, t].imshow(comp(pred[t])); ax[2, t].set_title(f"Ours T+{t+1}")
        for r in range(3): ax[r, t].axis("off")
    fig.suptitle(f"False-colour (VIS/SWIR/TIR2) — test example #{idx}", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_rgb_composite_train_{k}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote fig_rgb_composite_train_{k}.png")


def temporal_strip(model, idx, inp, tgt, device, k):
    with torch.no_grad():
        pred = model(inp.unsqueeze(0).to(device)).cpu().numpy()[0]
    gt = tgt.numpy(); inp_np = inp.numpy()
    fig, ax = plt.subplots(2, 7, figsize=(22, 6.5))
    for i in range(4):
        ax[0, i].imshow(inp_np[i, VIS], cmap="gray", vmin=0, vmax=1); ax[0, i].set_title(f"Input t{i-3}")
    for t in range(T):
        ax[0, 4+t].imshow(gt[t, VIS], cmap="gray", vmin=0, vmax=1); ax[0, 4+t].set_title(f"GT T+{t+1}")
        ax[1, 4+t].imshow(pred[t, VIS], cmap="gray", vmin=0, vmax=1); ax[1, 4+t].set_title(f"Pred T+{t+1}")
    for i in range(7): ax[0, i].axis("off")
    for i in range(7): ax[1, i].axis("off")
    ax[1, 1].text(0.5, 0.5, "observed →  |  ← forecast", ha="center", va="center", fontsize=13)
    fig.suptitle(f"Temporal sequence (band {BANDS[VIS]}) — test example #{idx}", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_temporal_strip_train_{k}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote fig_temporal_strip_train_{k}.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cpu"      # CPU to avoid contending with GPU training jobs
    model, ep = load_model(device)
    print(f"[visuals] loaded {BEST} (epoch {ep}); device={device}")
    for k, (idx, inp, tgt) in enumerate(pick_train_samples(3)):
        print(f"[visuals] training sample #{idx}")
        diffusion_journey(model, idx, inp, tgt, device, k)
        qualitative(model, idx, inp, tgt, device, k)
        rgb_composite(model, idx, inp, tgt, device, k)
        temporal_strip(model, idx, inp, tgt, device, k)
    print(f"[visuals] done -> {OUT}/")


if __name__ == "__main__":
    main()
