"""
Evaluate + visualise the conditional residual diffusion refiner.

- Loads frozen deterministic model (pidm_best.pth) + trained diffusion (diffusion.pth).
- On training samples: shows deterministic forecast, individual DDIM samples,
  ensemble mean, and per-pixel uncertainty (std) vs ground truth.
- Reports ensemble-mean SSIM on a test subset (deterministic stays the headline;
  diffusion adds generative sampling + uncertainty).

Outputs: results/fig_diffusion_samples_*.png, results/fig_diffusion_uncertainty.png
"""
import os, glob, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")
from data_split import chrono_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pidm_model import PIConditionalForecaster
from pidm_diffusion import ConditionalResidualDiffusion
from skimage.metrics import structural_similarity as sk_ssim

CACHE, DET, DIFF, OUT = "cache_128", "pidm_best.pth", "diffusion.pth", "results"
BANDS = ["TIR1", "TIR2", "WV", "VIS", "MIR", "SWIR"]
VIS, T = 3, 3


def ssim_one(p, g):
    return float(sk_ssim(np.clip(g, 0, 1).astype(np.float64), np.clip(p, 0, 1).astype(np.float64),
                         data_range=1.0, gaussian_weights=True, win_size=11, use_sample_covariance=False))


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(DET, map_location=device, weights_only=False)
    a = ck.get("args", {})
    det = PIConditionalForecaster(base=a.get("base", 64), refine_steps=a.get("refine", 2)).to(device)
    det.load_state_dict(ck["model"]); det.eval()
    dck = torch.load(DIFF, map_location=device, weights_only=False)
    diff = ConditionalResidualDiffusion(base=dck.get("base", 64)).to(device)
    diff.load_state_dict(dck["model"]); diff.eval()
    print(f"[diff-eval] det + diffusion (epoch {dck.get('epoch')}) loaded on {device}")

    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
    tr, _, te = chrono_split(len(files))

    # qualitative samples (training images)
    for k, idx in enumerate([tr[len(tr) // 4], tr[len(tr) // 2], tr[3 * len(tr) // 4]]):
        d = torch.load(files[idx], map_location="cpu", weights_only=False)
        inp = d["input"].float().unsqueeze(0).to(device); gt = d["target"].float().numpy()
        with torch.no_grad():
            ydet = det(inp).float()
            samples = [diff.sample(inp, ydet, steps=25).cpu().numpy()[0] for _ in range(3)]
            mean, std = diff.sample_ensemble(inp, ydet, n=8, steps=25)
        ydet_np = ydet.cpu().numpy()[0]; mean = mean.cpu().numpy()[0]; std = std.cpu().numpy()[0]
        cols = 3 + 1 + 1 + 1 + 1   # det, 2 samples, mean, std, gt
        fig, ax = plt.subplots(T, 7, figsize=(22, 9))
        for t in range(T):
            panels = [("Determ.", ydet_np[t, VIS]), ("Sample 1", samples[0][t, VIS]),
                      ("Sample 2", samples[1][t, VIS]), ("Sample 3", samples[2][t, VIS]),
                      ("Ens. mean", mean[t, VIS]), ("Uncertainty", std[t, VIS]),
                      ("Ground truth", gt[t, VIS])]
            for j, (title, img) in enumerate(panels):
                cmap = "magma" if title == "Uncertainty" else "gray"
                vargs = {} if title == "Uncertainty" else {"vmin": 0, "vmax": 1}
                ax[t, j].imshow(img, cmap=cmap, **vargs)
                if t == 0:
                    ax[t, j].set_title(title)
                if j == 0:
                    ax[t, j].set_ylabel(f"T+{t+1}")
                ax[t, j].set_xticks([]); ax[t, j].set_yticks([])
        fig.suptitle(f"Conditional diffusion refiner — training sample #{idx} (band {BANDS[VIS]})",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_diffusion_samples_{k}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote fig_diffusion_samples_{k}.png")

    # quantitative: ensemble-mean SSIM on a test subset
    sub = te[::6]
    sm, sd = 0.0, 0
    with torch.no_grad():
        for idx in sub:
            d = torch.load(files[idx], map_location="cpu", weights_only=False)
            inp = d["input"].float().unsqueeze(0).to(device); gt = d["target"].float().numpy()
            ydet = det(inp).float()
            mean, _ = diff.sample_ensemble(inp, ydet, n=8, steps=20)
            mean = mean.cpu().numpy()[0]
            for t in range(T):
                for c in range(6):
                    sm += ssim_one(mean[t, c], gt[t, c]); sd += 1
    print(f"[diff-eval] diffusion ENSEMBLE-MEAN test SSIM (n={len(sub)} seqs): {sm/sd:.4f}")
    print(f"[diff-eval] (deterministic headline stays 0.727; diffusion adds uncertainty)")


if __name__ == "__main__":
    main()
