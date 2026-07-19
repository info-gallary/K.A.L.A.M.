"""Complete metrics (SSIM, PSNR, MAE, RMSE, LPIPS) for EVERY method on the held-out
test set, in one table -> results/complete_metrics.json + fig_complete_metrics.png.
Addresses 'complete metrics for every method'."""
import os, glob, json, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")
from data_split import chrono_split
from pidm_model import PIConditionalForecaster
from baseline_simvp import SimVP
import cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as sk_ssim, peak_signal_noise_ratio as sk_psnr
try:
    import lpips as L; LP = L.LPIPS(net="alex", verbose=False); HAVE_LP = True
except Exception:
    HAVE_LP = False

CACHE, T, C, HW = "cache_128", 3, 6, 128


def of_forecast(inp):
    out = np.zeros((T, C, HW, HW), np.float32); gx, gy = np.meshgrid(np.arange(HW), np.arange(HW))
    for c in range(C):
        p = (np.clip(inp[-2, c], 0, 1) * 255).astype(np.uint8); q = (np.clip(inp[-1, c], 0, 1) * 255).astype(np.uint8)
        fl = cv2.calcOpticalFlowFarneback(p, q, None, 0.5, 3, 15, 3, 5, 1.2, 0); base = np.clip(inp[-1, c], 0, 1).astype(np.float32)
        for t in range(T):
            out[t, c] = cv2.remap(base, (gx + (t + 1) * fl[..., 0]).astype(np.float32),
                                  (gy + (t + 1) * fl[..., 1]).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.clip(out, 0, 1)


def metrics(pred, gt):
    s = ps = mae = mse = 0.0
    for t in range(T):
        for c in range(C):
            a = np.clip(pred[t, c], 0, 1).astype(np.float64); b = np.clip(gt[t, c], 0, 1).astype(np.float64)
            s += sk_ssim(b, a, data_range=1.0, gaussian_weights=True, win_size=11, use_sample_covariance=False)
            ps += (50.0 if np.allclose(a, b) else min(sk_psnr(b, a, data_range=1.0), 50.0))
            mae += np.abs(a - b).mean(); mse += ((a - b) ** 2).mean()
    n = T * C; return s / n, ps / n, mae / n, mse / n


def lpips_of(pred, gt, dev):
    if not HAVE_LP: return None
    ci = [3, 5, 1]
    pc = torch.from_numpy(np.stack([pred[:, i] for i in ci], 1)).float()
    gc = torch.from_numpy(np.stack([gt[:, i] for i in ci], 1)).float()
    with torch.no_grad(): return float(LP(pc * 2 - 1, gc * 2 - 1).mean())


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(os.path.join(CACHE, "*.pt"))); _, _, te = chrono_split(len(files))
    ck = torch.load("pidm_best.pth", map_location=dev, weights_only=False); a = ck.get("args", {})
    ours = PIConditionalForecaster(base=a.get("base", 64), refine_steps=a.get("refine", 2)).to(dev); ours.load_state_dict(ck["model"]); ours.eval()
    sv = None
    if os.path.exists("simvp.pth"):
        sv = SimVP().to(dev); sv.load_state_dict(torch.load("simvp.pth", map_location=dev, weights_only=False)["model"]); sv.eval()

    def tta(m, x):
        x = x.to(dev); o = [m(x)]
        for d in ([-1], [-2], [-1, -2]): o.append(torch.flip(m(torch.flip(x, d)), d))
        return torch.stack(o).mean(0).float().cpu().numpy()[0]

    # accumulate [SSIM, PSNR, MAE, MSE, LPIPS]; RMSE = sqrt(global mean MSE) at the
    # end (same convention as evaluate_and_plot / Tables IV-VI -> KALAM RMSE 0.0928,
    # all four methods consistent), NOT mean-of-per-sample-sqrt (Jensen-biased low).
    acc = {k: np.zeros(5) for k in ("Persistence", "Optical-flow", "SimVP", "Ours")}; n = 0
    with torch.no_grad():
        for idx in te:
            d = torch.load(files[idx], map_location="cpu", weights_only=False); inp = d["input"].float(); gt = d["target"].float().numpy()
            x = inp.unsqueeze(0); last = inp.numpy()[-1]
            preds = {"Persistence": np.repeat(last[None], T, 0), "Optical-flow": of_forecast(inp.numpy()),
                     "Ours": tta(ours, x)}
            if sv is not None: preds["SimVP"] = tta(sv, x)
            for k, pr in preds.items():
                s, ps, mae, mse = metrics(pr, gt); lp = lpips_of(pr, gt, dev) or 0.0
                acc[k] += np.array([s, ps, mae, mse, lp])
            n += 1
    rows = {k: {"SSIM": v[0]/n, "PSNR": v[1]/n, "MAE": v[2]/n, "RMSE": float(np.sqrt(v[3]/n)), "LPIPS": (v[4]/n if HAVE_LP else None)}
            for k, v in acc.items() if not (k == "SimVP" and sv is None)}
    json.dump(rows, open("results/complete_metrics.json", "w"), indent=2)
    print(f"{'Method':14s} {'SSIM':>7s} {'PSNR':>7s} {'MAE':>7s} {'RMSE':>7s} {'LPIPS':>7s}")
    for k, v in rows.items():
        print(f"{k:14s} {v['SSIM']:7.4f} {v['PSNR']:7.2f} {v['MAE']:7.4f} {v['RMSE']:7.4f} {(v['LPIPS'] if v['LPIPS'] else 0):7.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ks = list(rows); col = ["#bbbbbb", "#7fb3d5", "#9b7", "#2ca02c"][:len(ks)]
    ax[0].bar(ks, [rows[k]["SSIM"] for k in ks], color=col); ax[0].set_title("SSIM (higher better)"); ax[0].set_ylabel("SSIM")
    ax[1].bar(ks, [rows[k]["RMSE"] for k in ks], color=col); ax[1].set_title("RMSE (lower better)")
    for a_ in ax: a_.tick_params(axis="x", rotation=15)
    fig.suptitle("Complete metrics on held-out test (all methods)", fontweight="bold", fontsize=14)
    fig.tight_layout(); fig.savefig("results/fig_complete_metrics.png", dpi=200, bbox_inches="tight")
    print("wrote results/complete_metrics.json + fig_complete_metrics.png")


if __name__ == "__main__":
    main()
