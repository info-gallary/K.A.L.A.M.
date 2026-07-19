"""
Quantified, calibrated uncertainty for the conditional diffusion ensemble — the
TGRS-level deliverable. Computes proper probabilistic scores and calibration on
the held-out TEST split:

  - CRPS (Continuous Ranked Probability Score), fair ensemble estimator, vs a
    deterministic-as-degenerate-ensemble reference.
  - Spread–skill correlation: per-sequence ensemble spread vs ensemble-mean RMSE
    (a calibrated ensemble has spread tracking error, correlation > 0).
  - Reliability: binning pixels by predicted std and measuring realised RMSE per
    bin (a calibrated forecast lies on the diagonal).

Outputs: results/uncertainty_calibration.json, results/fig_calibration.png
Run after the GPU is free (e.g. after the ablation).
"""
import os, glob, json, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")
from data_split import chrono_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pidm_model import PIConditionalForecaster
from pidm_diffusion import ConditionalResidualDiffusion

CACHE, DET, DIFF, OUT = "cache_128", "pidm_best.pth", "diffusion.pth", "results"
M, STEPS = 12, 20          # ensemble members, DDIM steps


def crps_ensemble(members, y):
    """members: [M, ...]  y: [...]  -> mean CRPS (fair estimator)."""
    m = members.shape[0]
    term1 = np.mean(np.abs(members - y[None]), axis=0)
    diff = np.abs(members[:, None] - members[None, :]).mean(axis=(0, 1))
    return float((term1 - 0.5 * diff).mean())


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

    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
    _, _, te = chrono_split(len(files))
    sub = te[::5]
    print(f"[calib] {len(sub)} test sequences, M={M} members")

    crps_diff, crps_det, spreads, skills = [], [], [], []
    std_all, err_all = [], []
    with torch.no_grad():
        for idx in sub:
            d = torch.load(files[idx], map_location="cpu", weights_only=False)
            inp = d["input"].float().unsqueeze(0).to(device); gt = d["target"].float().numpy()[0] if False else d["target"].float().numpy()
            ydet = det(inp).float()
            members = np.stack([diff.sample(inp, ydet, steps=STEPS).cpu().numpy()[0] for _ in range(M)])  # [M,T,C,H,W]
            mean = members.mean(0); std = members.std(0)
            crps_diff.append(crps_ensemble(members, gt))
            crps_det.append(float(np.abs(ydet.cpu().numpy()[0] - gt).mean()))  # degenerate ensemble -> MAE
            spreads.append(float(std.mean()))
            skills.append(float(np.sqrt(((mean - gt) ** 2).mean())))
            std_all.append(std.ravel()); err_all.append(np.abs(mean - gt).ravel())
    std_all = np.concatenate(std_all); err_all = np.concatenate(err_all)

    # spread-skill correlation
    ss_corr = float(np.corrcoef(spreads, skills)[0, 1]) if len(spreads) > 2 else None
    # reliability: bin by std decile -> realised RMSE
    qs = np.quantile(std_all, np.linspace(0, 1, 11))
    bin_std, bin_rmse = [], []
    for i in range(10):
        m = (std_all >= qs[i]) & (std_all < qs[i + 1] + (1e-9 if i == 9 else 0))
        if m.sum() > 50:
            bin_std.append(std_all[m].mean()); bin_rmse.append(np.sqrt((err_all[m] ** 2).mean()))
    rel_corr = float(np.corrcoef(bin_std, bin_rmse)[0, 1]) if len(bin_std) > 2 else None

    res = {"crps_diffusion": float(np.mean(crps_diff)), "crps_deterministic_MAE": float(np.mean(crps_det)),
           "spread_skill_corr": ss_corr, "reliability_corr": rel_corr,
           "mean_spread": float(np.mean(spreads)), "mean_skill_rmse": float(np.mean(skills)),
           "n_test": len(sub), "ensemble_M": M}
    json.dump(res, open(f"{OUT}/uncertainty_calibration.json", "w"), indent=2)
    print(f"[calib] CRPS(diff)={res['crps_diffusion']:.4f}  CRPS-ref(MAE)={res['crps_deterministic_MAE']:.4f}")
    print(f"[calib] spread-skill corr={ss_corr}  reliability corr={rel_corr}")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].scatter(spreads, skills, s=28, color="#7a4fa3", alpha=0.7)
    lim = [0, max(max(spreads), max(skills)) * 1.1]
    ax[0].plot(lim, lim, "k--", alpha=.6, label="perfect calibration")
    ax[0].set_xlabel("ensemble spread (σ)"); ax[0].set_ylabel("ensemble-mean RMSE")
    ax[0].set_title(f"Spread–skill (corr={ss_corr:.2f})" if ss_corr else "Spread–skill")
    ax[0].legend()
    if bin_std:
        ax[1].plot(bin_std, bin_rmse, "o-", color="#2ca02c", lw=2)
        lim2 = [0, max(max(bin_std), max(bin_rmse)) * 1.1]
        ax[1].plot(lim2, lim2, "k--", alpha=.6, label="ideal")
        ax[1].set_xlabel("predicted σ (binned)"); ax[1].set_ylabel("realised RMSE")
        ax[1].set_title(f"Reliability (corr={rel_corr:.2f})" if rel_corr else "Reliability"); ax[1].legend()
    fig.suptitle("Calibrated predictive uncertainty (diffusion ensemble, held-out test)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_calibration.png", dpi=180, bbox_inches="tight")
    print(f"[calib] wrote results/uncertainty_calibration.json and fig_calibration.png")


if __name__ == "__main__":
    main()
