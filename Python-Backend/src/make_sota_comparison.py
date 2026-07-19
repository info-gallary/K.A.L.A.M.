"""
Aggregate the head-to-head: our physics-informed forecaster vs the learned
baselines (SimVP, ConvLSTM, ...) trained under the SAME leakage-free protocol,
plus persistence / optical-flow. Writes a comparison table + figure and declares
whether we beat the learned SOTA baselines on our data.

Run after the baselines finish: produces results/sota_comparison.{json,png}.
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results"


def load(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


def main():
    m = load(os.path.join(OUT, "metrics.json"))
    rows = []
    # baselines from the main eval
    if m:
        b = m["baselines"]
        rows.append(("Persistence", b["persistence"]["ssim"], b["persistence"]["psnr_db"], None, [None]*3))
        rows.append(("Optical-flow", b["optical_flow"]["ssim"], b["optical_flow"]["psnr_db"], None, [None]*3))
    # learned baselines under our protocol
    for name, fn in [("ConvLSTM", "baseline_convlstm.json"), ("SimVP", "baseline_simvp.json"),
                     ("PredRNN", "baseline_predrnn.json")]:
        r = load(os.path.join(OUT, fn))
        if r:
            rows.append((name + " (our protocol)", r["test_ssim"], None, r.get("params"), r.get("per_lead", [None]*3)))
    # ours
    if m:
        h = m["headline"]; pl = [m["per_lead_time"][f"T+{t+1}"]["ssim"] for t in range(3)]
        rows.append(("PI-Forecaster (ours)", h["ssim"], h["psnr_db"], None, pl))

    # winner among learned methods (incl ours)
    learned = [r for r in rows if "protocol" in r[0] or "ours" in r[0]]
    winner = max(learned, key=lambda r: r[1]) if learned else None
    ours = next((r for r in rows if "ours" in r[0]), None)
    beat = (winner is not None and ours is not None and winner[0] == ours[0])

    summary = {"rows": [{"method": r[0], "ssim": r[1], "psnr": r[2], "params": r[3],
                         "per_lead": r[4]} for r in rows],
               "winner_among_learned": winner[0] if winner else None,
               "ours_beats_learned_baselines": bool(beat)}
    json.dump(summary, open(os.path.join(OUT, "sota_comparison.json"), "w"), indent=2)

    # figure
    names = [r[0].replace(" (our protocol)", "*").replace(" (ours)", "\n(ours)") for r in rows]
    ssims = [r[1] for r in rows]
    colors = ["#bbbbbb" if ("Persist" in r[0] or "flow" in r[0])
              else ("#2ca02c" if "ours" in r[0] else "#7fb3d5") for r in rows]
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(rows)), 6))
    bars = ax.bar(names, ssims, color=colors)
    for b, s in zip(bars, ssims):
        ax.text(b.get_x() + b.get_width()/2, s + 0.004, f"{s:.3f}", ha="center", fontweight="bold")
    ax.axhline(0.80, color="red", ls="--", lw=1.5, alpha=.6, label="0.80")
    ax.set_ylabel("Held-out test SSIM")
    ax.set_title("Head-to-head on our data (same leakage-free protocol)  *=learned baseline",
                 fontsize=14, fontweight="bold")
    ax.legend(); plt.xticks(rotation=15, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_sota_comparison.png"), dpi=200, bbox_inches="tight")

    print("=== SOTA head-to-head (held-out test SSIM) ===")
    for r in rows:
        print(f"  {r[0]:28s} SSIM={r[1]:.4f}" + (f"  params={r[3]:,}" if r[3] else ""))
    print(f"WINNER among learned: {summary['winner_among_learned']}")
    print(f"Ours beats learned baselines: {summary['ours_beats_learned_baselines']}")


if __name__ == "__main__":
    main()
