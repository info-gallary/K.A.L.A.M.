"""
Ablation study for the journal: train compact (matched-budget) variants of the
forecaster, evaluate each on the held-out chronological TEST split, and write an
ablation table + figure.

Variants (each trained for --epochs at matched settings):
  full        : all components
  no_residual : predict absolute frames (drop persistence prior)
  no_meta     : drop meta-learner FiLM modulation
  no_refine   : drop diffusion-style iterative refinement
  no_physics  : SSIM+L1 only (drop physics regularisers)

Run AFTER the headline model finishes (single GPU). Example:
    python run_ablations.py --epochs 45
"""
import os, json, glob, subprocess, sys, argparse, warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
warnings.filterwarnings("ignore")
from data_split import chrono_split
from pidm_model import PIConditionalForecaster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from skimage.metrics import structural_similarity as sk_ssim
    HAVE_SK = True
except Exception:
    HAVE_SK = False

CACHE = "cache_128"; BANDS = 6; T = 3
VARIANTS = [
    ("full", []),
    ("no_residual", ["--no_residual"]),
    ("no_meta", ["--no_meta"]),
    ("no_refine", ["--no_refine"]),
    ("no_physics", ["--no_physics"]),
]


class CachedSeq(Dataset):
    def __init__(self, files): self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = torch.load(self.files[i], map_location="cpu", weights_only=False)
        return d["input"].float(), d["target"].float()


def ssim_one(p, g):
    p = np.clip(p, 0, 1).astype(np.float64); g = np.clip(g, 0, 1).astype(np.float64)
    if HAVE_SK:
        return float(sk_ssim(g, p, data_range=1.0, gaussian_weights=True,
                             win_size=11, use_sample_covariance=False))
    mse = np.mean((p - g) ** 2)
    return 1.0 - min(1.0, mse * 10)


def psnr_one(p, g):
    mse = np.mean((np.clip(p, 0, 1) - np.clip(g, 0, 1)) ** 2)
    return 50.0 if mse < 1e-10 else float(min(10*np.log10(1.0/mse), 50.0))


def quick_eval(tag, base, refine):
    ckp = f"pidm_best_{tag}.pth"
    if not os.path.exists(ckp):
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckp, map_location=dev, weights_only=False)
    a = ck.get("args", {})
    m = PIConditionalForecaster(
        base=a.get("base", base), refine_steps=a.get("refine", refine),
        use_residual=not a.get("no_residual", False),
        use_meta=not a.get("no_meta", False),
        use_refine=not a.get("no_refine", False)).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
    _, _, te = chrono_split(len(files))
    dl = DataLoader(Subset(CachedSeq(files), te), batch_size=4, num_workers=0)
    ss, ps, n = 0.0, 0.0, 0
    with torch.no_grad():
        for inp, tgt in dl:
            pr = m(inp.to(dev)).float().cpu().numpy(); tg = tgt.numpy()
            for b in range(pr.shape[0]):
                for t in range(T):
                    for c in range(BANDS):
                        ss += ssim_one(pr[b, t, c], tg[b, t, c])
                        ps += psnr_one(pr[b, t, c], tg[b, t, c])
                n += 1
    d = n * T * BANDS
    return {"ssim": ss / d, "psnr": ps / d, "epoch": ck.get("epoch")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--hours_each", type=float, default=1.2)
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    results = {}
    for tag, flags in VARIANTS:
        if not args.skip_train:
            cmd = [sys.executable, "train_pidm.py", "--epochs", str(args.epochs),
                   "--batch", str(args.batch), "--base", str(args.base),
                   "--refine", str(args.refine), "--workers", "4",
                   "--hours", str(args.hours_each), "--tag", tag,
                   # same recipe as the final model so only the ablated component differs
                   "--w_msssim", "2.0", "--lead_weights", "0.6,1.0,1.4",
                   "--augment", "--ema", "0.999"] + flags
            print(f"\n=== TRAIN ablation: {tag}  {flags} ===", flush=True)
            subprocess.run(cmd, check=False)
        r = quick_eval(tag, args.base, args.refine)
        results[tag] = r
        print(f"=== {tag}: {r} ===", flush=True)
        json.dump(results, open("results/ablation_results.json", "w"), indent=2)

    # figure
    os.makedirs("results", exist_ok=True)
    tags = [t for t, _ in VARIANTS if results.get(t)]
    ssim_v = [results[t]["ssim"] for t in tags]
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#2ca02c"] + ["#7fb3d5"] * (len(tags) - 1)
    ax.bar(tags, ssim_v, color=colors)
    ax.axhline(0.80, color="red", ls="--", lw=2, label="target 0.80")
    for i, v in enumerate(ssim_v):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_ylabel("Test SSIM"); ax.set_title("Ablation Study (held-out test)",
                                             fontsize=16, fontweight="bold")
    ax.legend(); plt.xticks(rotation=20)
    fig.tight_layout(); fig.savefig("results/fig_ablation.png", dpi=200, bbox_inches="tight")
    print("wrote results/ablation_results.json and results/fig_ablation.png")


if __name__ == "__main__":
    main()
