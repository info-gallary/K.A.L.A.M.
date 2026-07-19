"""Regenerate IEEE-clean ablation + efficiency figures from the benchmark JSONs."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 10, "figure.dpi": 200})

# ---------- Ablation ----------
ab = json.load(open("results/ablation_full.json"))
order = ["Full (residual+meta+refine)", "minus meta-learner", "minus refine",
         "minus physics (loss)", "minus residual-from-persist"]
labels = ["Full\n(res+meta+refine)", "− meta-\nlearner", "− refine", "− physics\nloss",
          "− residual-from-\npersistence"]
ssim = [ab[k]["SSIM"] for k in order]; dl = [ab[k]["dSSIM_vs_full"] for k in order]
ep = [ab[k]["epochs"] for k in order]
cols = ["#2ca02c", "#d62728", "#7fb3d5", "#9bbb9b", "#7a0010"]

fig, ax = plt.subplots(figsize=(8.4, 4.6))
bars = ax.bar(labels, ssim, color=cols, edgecolor="black", linewidth=0.6)
ax.axhline(0.630, ls="--", c="gray", lw=1.2); ax.text(4.35, 0.638, "persistence 0.630", color="gray", fontsize=9, ha="right")
ax.set_ylabel("Test SSIM"); ax.set_ylim(0, 0.80)
ax.set_title("Component ablation (held-out test, matched 4×-flip TTA protocol)", fontweight="bold")
for b, s, d, e in zip(bars, ssim, dl, ep):
    lab = f"{s:.3f}" + ("" if d == 0 else f"\n({d:+.3f})")
    ax.text(b.get_x() + b.get_width()/2, s + 0.008, lab, ha="center", va="bottom", fontsize=8.6)
    ax.text(b.get_x() + b.get_width()/2, 0.02, f"{e} ep", ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
ax.text(0.5, -0.20, "Matched 45-ep: residual −0.487 (dominant), meta-learner −0.077 (substantive); "
        "refine/physics at 28 ep are minor.", transform=ax.transAxes, ha="center", fontsize=8.4, color="#333")
fig.tight_layout(); fig.savefig("results/fig_ablation.png", bbox_inches="tight"); plt.close(fig)
print("wrote results/fig_ablation.png")

# ---------- Efficiency ----------
ef = json.load(open("results/efficiency_benchmark.json"))
lat = ef["latency_ms"]
keys = [("deterministic", "Deterministic"), ("deterministic_TTA4x", "Det + 4× TTA"),
        ("diffusion_DDIM25_1sample", "Diffusion\n(DDIM-25)"), ("det+diffusion_DDIM25", "Det + Diffusion")]
vals = [lat[k]/1000.0 for k, _ in keys]; names = [n for _, n in keys]
fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
b = ax[0].bar(names, vals, color=["#2ca02c", "#7fb3d5", "#ff7f0e", "#9467bd"], edgecolor="black", linewidth=0.6)
ax[0].set_ylabel("Latency per forecast (s)"); ax[0].set_title("Inference latency — RTX 3050 6 GB (batch 1, 128×128×6)", fontweight="bold", fontsize=11)
for bb, v in zip(b, vals): ax[0].text(bb.get_x()+bb.get_width()/2, v+0.05, f"{v:.2f}s", ha="center", fontsize=9.5)
txt =(f"Parameters: {ef['params_M']['total']} M\n"
       f"Deploy weights: {ef['deployable_weights_MB']['deterministic_fp32']} MB fp32 /\n"
       f"   {ef['deployable_weights_MB']['deterministic_fp16']} MB fp16\n"
       f"Throughput (det): {ef['throughput_fps']['deterministic']} forecasts/s\n"
       f"Peak GPU mem: {ef['peak_mem_MB']['deterministic']/1000:.2f} GB (det)\n"
       f"   {ef['peak_mem_MB']['diffusion_DDIM25']/1000:.2f} GB (diffusion)\n\n"
       f"→ +15/30/45-min forecast in 0.68 s\n   (~1300× the 15-min refresh)")
ax[1].axis("off"); ax[1].text(0.0, 0.95, "Deployment footprint", fontweight="bold", fontsize=11, va="top")
ax[1].text(0.0, 0.80, txt, va="top", fontsize=9.8, family="monospace")
fig.tight_layout(); fig.savefig("results/fig_efficiency.png", bbox_inches="tight"); plt.close(fig)
print("wrote results/fig_efficiency.png")
