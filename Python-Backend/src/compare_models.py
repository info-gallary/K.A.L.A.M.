"""Evaluate several checkpoints on the held-out TEST set (skimage SSIM + TTA) and
report the winner, so we keep whichever model genuinely generalizes best.

    python compare_models.py pidm_best.pth pidm_best_v2.pth pidm_last_v2.pth
"""
import sys, glob, os, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")
from data_split import chrono_split
from pidm_model import PIConditionalForecaster
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

CACHE, T, C = "cache_128", 3, 6


def ssim_one(p, g):
    p = np.clip(p, 0, 1).astype(np.float64); g = np.clip(g, 0, 1).astype(np.float64)
    return float(sk_ssim(g, p, data_range=1.0, gaussian_weights=True, win_size=11,
                         use_sample_covariance=False))


def psnr_one(p, g):
    mse = np.mean((np.clip(p, 0, 1) - np.clip(g, 0, 1)) ** 2)
    return 50.0 if mse < 1e-10 else float(min(10 * np.log10(1.0 / mse), 50.0))


def tta(model, inp, device):
    inp = inp.to(device); outs = [model(inp)]
    for d in ([-1], [-2], [-1, -2]):
        outs.append(torch.flip(model(torch.flip(inp, d)), d))
    return torch.stack(outs).mean(0)


def evaluate(path, device, files, te):
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck.get("args", {})
    m = PIConditionalForecaster(base=a.get("base", 64),
                                refine_steps=a.get("refine", 2)).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    S = np.zeros((T, C)); P = 0.0; n = 0
    with torch.no_grad():
        for i in te:
            d = torch.load(files[i], map_location="cpu", weights_only=False)
            inp = d["input"].float().unsqueeze(0); g = d["target"].float().numpy()
            pr = tta(m, inp, device).float().cpu().numpy()[0]
            for t in range(T):
                for c in range(C):
                    S[t, c] += ssim_one(pr[t, c], g[t, c]); P += psnr_one(pr[t, c], g[t, c])
            n += 1
    S /= n; P /= (n * T * C)
    return {"ssim": float(S.mean()), "psnr": P,
            "per_lead": [float(S[t].mean()) for t in range(T)],
            "epoch": ck.get("epoch")}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
    _, _, te = chrono_split(len(files))
    best = (None, -1)
    for path in sys.argv[1:]:
        r = evaluate(path, device, files, te)
        if r is None:
            print(f"{path}: (missing)"); continue
        print(f"{path:24s} epoch={r['epoch']}  TEST SSIM={r['ssim']:.4f}  "
              f"PSNR={r['psnr']:.2f}  per-lead={[round(x,3) for x in r['per_lead']]}")
        if r["ssim"] > best[1]:
            best = (path, r["ssim"])
    print(f"\nWINNER: {best[0]}  (test SSIM {best[1]:.4f})")


if __name__ == "__main__":
    main()
