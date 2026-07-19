"""Proper ablation: evaluate every trained variant checkpoint on the held-out TEST
set under the SAME protocol (4x-flip TTA, skimage SSIM/PSNR/MAE/RMSE, global RMSE).
-> results/ablation_full.json. Variants share architecture flags stored in the
checkpoint's 'args' (no_residual / no_meta / no_refine; no_physics is a loss-only
change, same architecture)."""
import os, glob, json, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
from data_split import chrono_split
from pidm_model import PIConditionalForecaster
from skimage.metrics import structural_similarity as sk_ssim, peak_signal_noise_ratio as sk_psnr

CACHE, T, C = "cache_128", 3, 6
VARIANTS = {
    "Full (residual+meta+refine)": ("pidm_best_full.pth", 45),
    "minus meta-learner":          ("pidm_best_no_meta.pth", 45),
    "minus refine":                ("pidm_best_no_refine.pth", 28),
    "minus physics (loss)":        ("pidm_best_no_physics.pth", 28),
    "minus residual-from-persist": ("pidm_best_no_residual.pth", 45),
}


def metrics(pred, gt):
    s = ps = mae = mse = 0.0
    for t in range(T):
        for c in range(C):
            a = np.clip(pred[t, c], 0, 1).astype(np.float64); b = np.clip(gt[t, c], 0, 1).astype(np.float64)
            s += sk_ssim(b, a, data_range=1.0, gaussian_weights=True, win_size=11, use_sample_covariance=False)
            ps += (50.0 if np.allclose(a, b) else min(sk_psnr(b, a, data_range=1.0), 50.0))
            mae += np.abs(a - b).mean(); mse += ((a - b) ** 2).mean()
    n = T * C; return s / n, ps / n, mae / n, mse / n


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(os.path.join(CACHE, "*.pt"))); _, _, te = chrono_split(len(files))

    def tta(m, x):
        x = x.to(dev); o = [m(x)]
        for d in ([-1], [-2], [-1, -2]): o.append(torch.flip(m(torch.flip(x, d)), d))
        return torch.stack(o).mean(0).float().cpu().numpy()[0]

    rows = {}
    for name, (ckpt, ep) in VARIANTS.items():
        if not os.path.exists(ckpt):
            print("skip (missing):", ckpt); continue
        ck = torch.load(ckpt, map_location=dev, weights_only=False); a = ck.get("args", {})
        m = PIConditionalForecaster(base=a.get("base", 64), refine_steps=a.get("refine", 2),
                                    use_residual=not a.get("no_residual", False),
                                    use_meta=not a.get("no_meta", False),
                                    use_refine=not a.get("no_refine", False)).to(dev)
        m.load_state_dict(ck["model"]); m.eval()
        acc = np.zeros(4); n = 0
        with torch.no_grad():
            for idx in te:
                d = torch.load(files[idx], map_location="cpu", weights_only=False)
                inp = d["input"].float(); gt = d["target"].float().numpy()
                s, ps, mae, mse = metrics(tta(m, inp.unsqueeze(0)), gt)
                acc += np.array([s, ps, mae, mse]); n += 1
        rows[name] = {"epochs": ep, "SSIM": acc[0]/n, "PSNR": acc[1]/n,
                      "MAE": acc[2]/n, "RMSE": float(np.sqrt(acc[3]/n))}
        print(f"{name:30s} ep{ep:<3d} SSIM {rows[name]['SSIM']:.4f}  PSNR {rows[name]['PSNR']:.2f}  "
              f"MAE {rows[name]['MAE']:.4f}  RMSE {rows[name]['RMSE']:.4f}")
        del m; torch.cuda.empty_cache()

    # deltas vs Full
    if "Full (residual+meta+refine)" in rows:
        full = rows["Full (residual+meta+refine)"]["SSIM"]  # noqa
        for k in rows: rows[k]["dSSIM_vs_full"] = rows[k]["SSIM"] - full
    json.dump(rows, open("results/ablation_full.json", "w"), indent=2)
    print("\nwrote results/ablation_full.json")
    print("\nSSIM deltas vs Full:")
    for k, v in rows.items(): print(f"  {k:30s} {v['dSSIM_vs_full']:+.4f}  (ep{v['epochs']})")


if __name__ == "__main__":
    main()
