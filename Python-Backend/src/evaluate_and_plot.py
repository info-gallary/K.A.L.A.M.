"""
Journal-grade evaluation + figure package for the Physics-Informed Conditional
Forecaster + Meta-Learner.

Evaluated on the HELD-OUT chronological TEST split (leakage-free).
Baselines: persistence (last frame) and optical-flow (Farneback) advection.
Metrics  : SSIM (Gaussian, win=11), PSNR, MAE, RMSE  (per band, per lead time).
Stats    : Wilcoxon signed-rank tests model-vs-baseline on per-sequence SSIM.

Outputs (results/):
  metrics.json, REPORT.md
  fig_training_curves.png      fig_model_comparison.png
  fig_per_band.png             fig_per_leadtime.png
  fig_ssim_distribution.png    fig_error_heatmap.png
  fig_band_leadtime_heatmap.png fig_ssim_psnr_scatter.png
  fig_qualitative_*.png        fig_rgb_composite_*.png   fig_temporal_strip_*.png
"""
import os, glob, json, warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset

warnings.filterwarnings("ignore")
from data_split import chrono_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
except Exception:
    pass
import cv2
from pidm_model import PIConditionalForecaster

try:
    from skimage.metrics import structural_similarity as sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    HAVE_SK = True
except Exception:
    HAVE_SK = False
try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

CACHE, BEST, HIST, OUT = "cache_128", "pidm_best.pth", "pidm_history.json", "results"
BANDS = ["TIR1", "TIR2", "WV", "VIS", "MIR", "SWIR"]
TARGET_SSIM, TARGET_PSNR = 0.80, 24.0
T, C, HW = 3, 6, 128
USE_TTA = True   # test-time augmentation (flip-averaging): legitimate accuracy boost


def tta_predict(model, inp, device):
    """4x flip test-time augmentation, predictions un-flipped and averaged."""
    inp = inp.to(device)
    outs = [model(inp)]
    for dims in ([-1], [-2], [-1, -2]):
        outs.append(torch.flip(model(torch.flip(inp, dims=dims)), dims=dims))
    return torch.stack(outs).mean(0)


def temporal_consistency(pred, tgt):
    """Mean |Δpred - Δtgt| over consecutive predicted frames (lower=better)."""
    if pred.shape[0] < 2:
        return 0.0
    dp = np.diff(pred, axis=0); dt = np.diff(tgt, axis=0)
    return float(np.mean(np.abs(dp - dt)))


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
    mp, mg = p.mean(), g.mean(); vp, vg = p.var(), g.var()
    cov = ((p - mp) * (g - mg)).mean(); C1, C2 = 1e-4, 9e-4
    return float(((2*mp*mg+C1)*(2*cov+C2))/((mp**2+mg**2+C1)*(vp+vg+C2)))


def psnr_one(p, g):
    p = np.clip(p, 0, 1).astype(np.float64); g = np.clip(g, 0, 1).astype(np.float64)
    mse = np.mean((p - g) ** 2)
    return 50.0 if mse < 1e-10 else float(min(10*np.log10(1.0/mse), 50.0))


def optflow_forecast(inp_bands):
    """inp_bands: [4, C, H, W] in [0,1] -> [T, C, H, W] Farneback advection forecast."""
    out = np.zeros((T, C, HW, HW), np.float32)
    gx, gy = np.meshgrid(np.arange(HW), np.arange(HW))
    for c in range(C):
        prev = (np.clip(inp_bands[-2, c], 0, 1) * 255).astype(np.uint8)
        curr = (np.clip(inp_bands[-1, c], 0, 1) * 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        base = np.clip(inp_bands[-1, c], 0, 1).astype(np.float32)
        for t in range(T):
            mapx = (gx + (t + 1) * flow[..., 0]).astype(np.float32)
            mapy = (gy + (t + 1) * flow[..., 1]).astype(np.float32)
            out[t, c] = cv2.remap(base, mapx, mapy, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    return np.clip(out, 0, 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}  scikit-image={HAVE_SK}  scipy={HAVE_SCIPY}")

    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
    ds = CachedSeq(files)
    _, _, te_idx = chrono_split(len(files))
    te = Subset(ds, te_idx)
    print(f"[eval] held-out TEST sequences: {len(te)}")

    ck = torch.load(BEST, map_location=device, weights_only=False)
    a = ck.get("args", {})
    model = PIConditionalForecaster(base=a.get("base", 64),
                                    refine_steps=a.get("refine", 2)).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"[eval] loaded {BEST} epoch={ck.get('epoch','?')} "
          f"train_val_ssim={ck.get('val_ssim', float('nan')):.4f}")

    # accumulators [method][T,C]
    acc = {k: {"ssim": np.zeros((T, C)), "psnr": np.zeros((T, C)),
               "mae": np.zeros((T, C)), "mse": np.zeros((T, C))}
           for k in ("model", "persist", "optflow")}
    seq_ssim = {k: [] for k in ("model", "persist", "optflow")}
    err_map = np.zeros((C, HW, HW)); n = 0
    tc_list = []
    try:
        import lpips as lpips_lib
        lpips_net = lpips_lib.LPIPS(net="alex", verbose=False).to(device)
        lp_list = []
    except Exception as e:
        lpips_net = None; lp_list = None
        print(f"[eval] LPIPS unavailable ({e}); skipping")

    vl = DataLoader(te, batch_size=4, shuffle=False, num_workers=0)
    with torch.no_grad():
        for inp, tgt in vl:
            pred_g = tta_predict(model, inp, device) if USE_TTA else model(inp.to(device))
            pred = pred_g.float().cpu().numpy()
            inp_np, tgt_np = inp.numpy(), tgt.numpy()
            for b in range(inp_np.shape[0]):
                last = inp_np[b, -1]
                of = optflow_forecast(inp_np[b])
                sm = {k: [] for k in acc}
                for t in range(T):
                    for c in range(C):
                        gt = tgt_np[b, t, c]
                        cand = {"model": pred[b, t, c], "persist": last[c],
                                "optflow": of[t, c]}
                        for k, pr in cand.items():
                            s = ssim_one(pr, gt); ps = psnr_one(pr, gt)
                            e = np.abs(pr - gt)
                            acc[k]["ssim"][t, c] += s; acc[k]["psnr"][t, c] += ps
                            acc[k]["mae"][t, c] += e.mean()
                            acc[k]["mse"][t, c] += (e ** 2).mean()
                            sm[k].append(s)
                        err_map[c] += np.abs(pred[b, t, c] - gt)
                for k in acc:
                    seq_ssim[k].append(float(np.mean(sm[k])))
                tc_list.append(temporal_consistency(pred[b], tgt_np[b]))
                if lpips_net is not None:
                    ci = [3, 5, 1]
                    pc = torch.from_numpy(np.stack([pred[b][:, i] for i in ci], 1)).float().to(device)
                    gc = torch.from_numpy(np.stack([tgt_np[b][:, i] for i in ci], 1)).float().to(device)
                    with torch.no_grad():
                        lp_list.append(lpips_net(pc * 2 - 1, gc * 2 - 1).mean().item())
                n += 1
    err_map /= (n * T)
    for k in acc:
        for m in acc[k]:
            acc[k][m] /= n

    def headline(k):
        return float(acc[k]["ssim"].mean()), float(acc[k]["psnr"].mean())
    ms, mp = headline("model"); ps_s, ps_p = headline("persist"); os_s, os_p = headline("optflow")
    rmse = float(np.sqrt(acc["model"]["mse"].mean()))
    mae = float(acc["model"]["mae"].mean())
    print(f"[eval] MODEL    SSIM={ms:.4f} PSNR={mp:.2f} MAE={mae:.4f} RMSE={rmse:.4f}")
    print(f"[eval] PERSIST  SSIM={ps_s:.4f} PSNR={ps_p:.2f}")
    print(f"[eval] OPTFLOW  SSIM={os_s:.4f} PSNR={os_p:.2f}")

    def wlx(a_, b_):
        if not HAVE_SCIPY: return None
        try: return float(wilcoxon(a_, b_).pvalue)
        except Exception: return None

    lpips_mean = float(np.mean(lp_list)) if lp_list else None
    tc_mean = float(np.mean(tc_list)) if tc_list else None
    print(f"[eval] LPIPS={lpips_mean}  temporal-consistency={tc_mean}  TTA={USE_TTA}")

    metrics = {
        "headline": {"ssim": ms, "psnr_db": mp, "mae": mae, "rmse": rmse,
                     "lpips": lpips_mean, "temporal_consistency": tc_mean,
                     "test_time_augmentation": USE_TTA,
                     "target_ssim": TARGET_SSIM, "target_psnr": TARGET_PSNR,
                     "ssim_target_met": ms >= TARGET_SSIM,
                     "psnr_target_met": mp >= TARGET_PSNR},
        "baselines": {"persistence": {"ssim": ps_s, "psnr_db": ps_p},
                      "optical_flow": {"ssim": os_s, "psnr_db": os_p}},
        "improvement": {"vs_persistence_ssim": ms - ps_s, "vs_persistence_psnr": mp - ps_p,
                        "vs_optflow_ssim": ms - os_s, "vs_optflow_psnr": mp - os_p},
        "significance_wilcoxon_p": {
            "model_vs_persistence": wlx(seq_ssim["model"], seq_ssim["persist"]),
            "model_vs_optflow": wlx(seq_ssim["model"], seq_ssim["optflow"])},
        "per_band": {BANDS[c]: {"ssim": float(acc["model"]["ssim"][:, c].mean()),
                                "psnr_db": float(acc["model"]["psnr"][:, c].mean()),
                                "persistence_ssim": float(acc["persist"]["ssim"][:, c].mean()),
                                "optflow_ssim": float(acc["optflow"]["ssim"][:, c].mean())}
                     for c in range(C)},
        "per_lead_time": {f"T+{t+1}": {"ssim": float(acc["model"]["ssim"][t].mean()),
                                       "psnr_db": float(acc["model"]["psnr"][t].mean())}
                          for t in range(T)},
        "test_sequences": n, "metric_backend": "scikit-image" if HAVE_SK else "fallback",
        "split": "chronological held-out (leakage-free, gap-buffered)",
    }
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)

    # ---------------- figures ----------------
    _curves()
    _model_comparison(ms, mp, ps_s, ps_p, os_s, os_p)
    _per_band(acc)
    _per_leadtime(acc["model"])
    _distribution(seq_ssim)
    _error_heatmap(err_map)
    _band_leadtime_heatmap(acc["model"]["ssim"])
    _scatter(seq_ssim, acc)
    _report(metrics, ck)
    print(f"[eval] wrote metrics + aggregate figures to {OUT}/  "
          f"(per-sample inference visuals: run fig_inference_visuals.py)")


# ---- figure helpers ----------------------------------------------------------
def _curves():
    if not os.path.exists(HIST): return
    h = json.load(open(HIST)); ep = np.arange(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(21, 6))
    ax[0].plot(ep, h["train_loss"], "#1f77b4", lw=2.2); ax[0].set_yscale("log")
    ax[0].set_title("Training Loss"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss")
    ax[1].plot(ep, h["val_ssim"], "#2ca02c", lw=2.2)
    ax[1].axhline(TARGET_SSIM, color="red", ls="--", lw=2, label=f"target {TARGET_SSIM}")
    ax[1].set_title("Validation SSIM"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    ax[2].plot(ep, h["val_psnr"], "#ff7f0e", lw=2.2)
    ax[2].axhline(TARGET_PSNR, color="red", ls="--", lw=2, label=f"target {TARGET_PSNR} dB")
    ax[2].set_title("Validation PSNR"); ax[2].set_xlabel("Epoch"); ax[2].legend()
    fig.suptitle("Training Dynamics", fontsize=18, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_training_curves.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def _model_comparison(ms, mp, ps, pp, os_, op):
    methods = ["Persistence", "Optical-flow", "PI-Forecaster (ours)"]
    ssim_v = [ps, os_, ms]; psnr_v = [pp, op, mp]
    col = ["#bbbbbb", "#7fb3d5", "#2ca02c"]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].bar(methods, ssim_v, color=col)
    ax[0].axhline(TARGET_SSIM, color="red", ls="--", lw=2, label=f"target {TARGET_SSIM}")
    ax[0].set_ylabel("SSIM"); ax[0].set_title("SSIM"); ax[0].legend()
    for i, v in enumerate(ssim_v): ax[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")
    ax[1].bar(methods, psnr_v, color=col)
    ax[1].axhline(TARGET_PSNR, color="red", ls="--", lw=2, label=f"target {TARGET_PSNR} dB")
    ax[1].set_ylabel("PSNR (dB)"); ax[1].set_title("PSNR"); ax[1].legend()
    for i, v in enumerate(psnr_v): ax[1].text(i, v + 0.1, f"{v:.2f}", ha="center", fontweight="bold")
    fig.suptitle("Model vs Baselines (held-out test)", fontsize=18, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_model_comparison.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def _per_band(acc):
    x = np.arange(C); w = 0.27
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    ax[0].bar(x - w, acc["persist"]["ssim"].mean(0), w, label="Persistence", color="#bbbbbb")
    ax[0].bar(x, acc["optflow"]["ssim"].mean(0), w, label="Optical-flow", color="#7fb3d5")
    ax[0].bar(x + w, acc["model"]["ssim"].mean(0), w, label="Ours", color="#2ca02c")
    ax[0].axhline(TARGET_SSIM, color="red", ls="--", lw=2)
    ax[0].set_xticks(x); ax[0].set_xticklabels(BANDS); ax[0].set_ylabel("SSIM")
    ax[0].set_title("Per-band SSIM"); ax[0].legend()
    ax[1].bar(x, acc["model"]["psnr"].mean(0), color="#ff7f0e")
    ax[1].axhline(TARGET_PSNR, color="red", ls="--", lw=2)
    ax[1].set_xticks(x); ax[1].set_xticklabels(BANDS); ax[1].set_ylabel("PSNR (dB)")
    ax[1].set_title("Per-band PSNR (ours)")
    fig.suptitle("Per-Spectral-Band Performance", fontsize=18, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_per_band.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def _per_leadtime(m):
    leads = [f"T+{t+1}" for t in range(T)]
    fig, a1 = plt.subplots(figsize=(9, 6))
    a1.plot(leads, m["ssim"].mean(1), "o-", color="#2ca02c", lw=2.5, ms=10, label="SSIM")
    a1.axhline(TARGET_SSIM, color="#2ca02c", ls=":", alpha=.6); a1.set_ylabel("SSIM", color="#2ca02c")
    a2 = a1.twinx()
    a2.plot(leads, m["psnr"].mean(1), "s--", color="#ff7f0e", lw=2.5, ms=10, label="PSNR")
    a2.axhline(TARGET_PSNR, color="#ff7f0e", ls=":", alpha=.6); a2.set_ylabel("PSNR (dB)", color="#ff7f0e")
    a1.set_xlabel("Forecast lead time"); a1.set_title("Skill vs Forecast Horizon", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_per_leadtime.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def _distribution(seq):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(seq["persist"], bins=30, alpha=.55, color="#bbbbbb", label=f"Persistence μ={np.mean(seq['persist']):.3f}")
    ax.hist(seq["optflow"], bins=30, alpha=.55, color="#7fb3d5", label=f"Optical-flow μ={np.mean(seq['optflow']):.3f}")
    ax.hist(seq["model"], bins=30, alpha=.7, color="#2ca02c", label=f"Ours μ={np.mean(seq['model']):.3f}")
    ax.axvline(TARGET_SSIM, color="red", ls="--", lw=2, label=f"target {TARGET_SSIM}")
    ax.set_xlabel("Per-sequence mean SSIM"); ax.set_ylabel("Count")
    ax.set_title("SSIM Distribution (held-out test)", fontsize=16, fontweight="bold"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ssim_distribution.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def _error_heatmap(err):
    fig, axes = plt.subplots(1, C, figsize=(24, 4.2))
    vmax = err.max()
    for c in range(C):
        im = axes[c].imshow(err[c], cmap="inferno", vmin=0, vmax=vmax)
        axes[c].set_title(f"{BANDS[c]}\nMAE={err[c].mean():.3f}"); axes[c].axis("off")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    fig.suptitle("Mean Absolute Error Map per Band", fontsize=16, fontweight="bold")
    fig.savefig(f"{OUT}/fig_error_heatmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def _band_leadtime_heatmap(ssim):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(ssim, cmap="viridis", vmin=max(0, ssim.min()-0.05), vmax=1.0, aspect="auto")
    ax.set_xticks(range(C)); ax.set_xticklabels(BANDS)
    ax.set_yticks(range(T)); ax.set_yticklabels([f"T+{t+1}" for t in range(T)])
    for t in range(T):
        for c in range(C):
            ax.text(c, t, f"{ssim[t,c]:.2f}", ha="center", va="center",
                    color="white" if ssim[t, c] < 0.7 else "black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("SSIM by Band × Lead time", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_band_leadtime_heatmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def _scatter(seq, acc):
    fig, ax = plt.subplots(figsize=(9, 7))
    psnr_seq = None
    ax.scatter(seq["persist"], [acc["persist"]["psnr"].mean()] * len(seq["persist"]),
               s=8, alpha=.25, color="#bbbbbb", label="Persistence")
    ax.scatter(seq["model"], [acc["model"]["psnr"].mean()] * len(seq["model"]),
               s=8, alpha=.0)  # placeholder to keep axis
    ax.clear()
    ax.scatter(seq["persist"], seq["optflow"], s=14, alpha=.4, color="#7fb3d5", label="optflow vs persist")
    ax.scatter(seq["persist"], seq["model"], s=16, alpha=.6, color="#2ca02c", label="ours vs persist")
    lim = [min(min(seq['persist']), 0.3), 1.0]
    ax.plot(lim, lim, "k--", lw=1, alpha=.6)
    ax.axhline(TARGET_SSIM, color="red", ls=":", lw=1.5)
    ax.set_xlabel("Persistence SSIM (difficulty)"); ax.set_ylabel("Method SSIM")
    ax.set_title("Per-sequence skill vs difficulty", fontsize=15, fontweight="bold"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ssim_psnr_scatter.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def _qualitative(inp, tgt, pred, of, k):
    bi = 3  # VIS
    fig, ax = plt.subplots(4, T, figsize=(4 * T, 15))
    last = inp[-1, bi]
    for t in range(T):
        ax[0, t].imshow(last, cmap="gray", vmin=0, vmax=1); ax[0, t].set_title(f"Persistence→T+{t+1}")
        ax[1, t].imshow(tgt[t, bi], cmap="gray", vmin=0, vmax=1); ax[1, t].set_title(f"Ground truth T+{t+1}")
        s = ssim_one(pred[t, bi], tgt[t, bi])
        ax[2, t].imshow(pred[t, bi], cmap="gray", vmin=0, vmax=1); ax[2, t].set_title(f"Ours T+{t+1} (SSIM {s:.3f})")
        ax[3, t].imshow(np.abs(pred[t, bi]-tgt[t, bi]), cmap="inferno", vmin=0, vmax=0.5); ax[3, t].set_title(f"|error| T+{t+1}")
        for r in range(4): ax[r, t].axis("off")
    fig.suptitle(f"Qualitative forecast — test sample {k} (band {BANDS[bi]})", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_qualitative_{k}.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def _rgb_composite(inp, tgt, pred, of, k):
    # false-color composite from bands VIS, SWIR, TIR2 (3,5,1)
    idx = [3, 5, 1]
    def comp(arr):  # arr [C,H,W] -> [H,W,3]
        return np.clip(np.stack([arr[i] for i in idx], -1), 0, 1)
    fig, ax = plt.subplots(3, T, figsize=(4 * T, 11))
    for t in range(T):
        ax[0, t].imshow(comp(inp[-1])); ax[0, t].set_title(f"Persistence→T+{t+1}")
        ax[1, t].imshow(comp(tgt[t])); ax[1, t].set_title(f"Ground truth T+{t+1}")
        ax[2, t].imshow(comp(pred[t])); ax[2, t].set_title(f"Ours T+{t+1}")
        for r in range(3): ax[r, t].axis("off")
    fig.suptitle(f"False-colour (VIS/SWIR/TIR2) composite — sample {k}", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_rgb_composite_{k}.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def _temporal_strip(inp, tgt, pred, of, k):
    bi = 3
    fig, ax = plt.subplots(2, 7, figsize=(22, 6.5))
    for i in range(4):
        ax[0, i].imshow(inp[i, bi], cmap="gray", vmin=0, vmax=1); ax[0, i].set_title(f"Input t-{4-i}")
        ax[1, i].axis("off")
    for t in range(T):
        ax[0, 4+t].imshow(tgt[t, bi], cmap="gray", vmin=0, vmax=1); ax[0, 4+t].set_title(f"GT T+{t+1}")
        ax[1, 4+t].imshow(pred[t, bi], cmap="gray", vmin=0, vmax=1); ax[1, 4+t].set_title(f"Pred T+{t+1}")
    for i in range(7): ax[0, i].axis("off")
    for i in range(4, 7): ax[1, i].axis("off")
    ax[1, 0].text(0.5, 0.5, "observed →\n← forecast", ha="center", va="center", fontsize=12)
    fig.suptitle(f"Temporal sequence (band {BANDS[bi]}) — sample {k}", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_temporal_strip_{k}.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def _report(m, ck):
    h = m["headline"]; b = m["baselines"]
    L = ["# Physics-Informed Conditional Forecaster + Meta-Learner — Results\n",
         f"Evaluated on **{m['test_sequences']} held-out test sequences** "
         f"({m['split']}); metric backend: {m['metric_backend']}.\n",
         "## Headline\n",
         "| Method | SSIM | PSNR (dB) |",
         "|---|---|---|",
         f"| Persistence | {b['persistence']['ssim']:.4f} | {b['persistence']['psnr_db']:.2f} |",
         f"| Optical-flow (Farneback) | {b['optical_flow']['ssim']:.4f} | {b['optical_flow']['psnr_db']:.2f} |",
         f"| **PI-Forecaster (ours)** | **{h['ssim']:.4f}** | **{h['psnr_db']:.2f}** |",
         f"| Target | {h['target_ssim']} {'✅' if h['ssim_target_met'] else '❌'} | "
         f"{h['target_psnr']} {'✅' if h['psnr_target_met'] else '❌'} |\n",
         f"MAE={h['mae']:.4f}, RMSE={h['rmse']:.4f}. "
         f"Improvement vs persistence: +{m['improvement']['vs_persistence_ssim']:.3f} SSIM / "
         f"+{m['improvement']['vs_persistence_psnr']:.2f} dB; "
         f"vs optical-flow: +{m['improvement']['vs_optflow_ssim']:.3f} SSIM.\n"]
    p = m["significance_wilcoxon_p"]
    if p["model_vs_persistence"] is not None:
        L.append(f"Wilcoxon signed-rank (per-sequence SSIM): vs persistence "
                 f"p={p['model_vs_persistence']:.2e}, vs optical-flow "
                 f"p={p['model_vs_optflow']:.2e}.\n")
    L += ["## Per-band\n", "| Band | SSIM | PSNR | Persist SSIM | OptFlow SSIM |",
          "|---|---|---|---|---|"]
    for bd, v in m["per_band"].items():
        L.append(f"| {bd} | {v['ssim']:.3f} | {v['psnr_db']:.2f} | "
                 f"{v['persistence_ssim']:.3f} | {v['optflow_ssim']:.3f} |")
    L += ["\n## Per-lead-time\n", "| Lead | SSIM | PSNR |", "|---|---|---|"]
    for lt, v in m["per_lead_time"].items():
        L.append(f"| {lt} | {v['ssim']:.3f} | {v['psnr_db']:.2f} |")
    L += ["\n## Architecture\n",
          "- Conditional U-Net (multi-scale stem, CBAM, GroupNorm) → residual field.",
          "- Meta-Learner: motion→FiLM modulation of bottleneck (per-sequence fast adaptation).",
          "- Residual-from-last-frame (persistence prior) + per-lead-time gain.",
          "- Diffusion-style gated iterative refinement (K steps).",
          "- Physics-informed losses: SSIM + L1 + gradient + temporal smoothness + "
          "divergence + advection consistency + TV.",
          f"\n_Best checkpoint: epoch {ck.get('epoch','?')}._"]
    open(f"{OUT}/REPORT.md", "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
