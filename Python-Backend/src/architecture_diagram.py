"""Self-contained schematic of the Physics-Informed Conditional Forecaster +
Meta-Learner. Pure matplotlib (no torch) -> runnable any time.
Outputs: results/fig_architecture.png and results/fig_pipeline.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "results"
os.makedirs(OUT, exist_ok=True)


def _box(ax, x, y, w, h, text, fc, ec="#222", fs=11, tc="#111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.6, edgecolor=ec, facecolor=fc, alpha=0.95))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, wrap=True, fontweight="medium")


def _arrow(ax, p1, p2, color="#333", style="-|>", lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                                 lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def architecture():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Physics-Informed Conditional Forecaster + Meta-Learner",
                 fontsize=20, fontweight="bold", pad=16)

    blue, green, orange, purple, gray, red = (
        "#cfe3f7", "#cce8cc", "#ffe0b3", "#e6d4f2", "#e8e8e8", "#f6c6c6")

    _box(ax, 0.3, 4.1, 2.4, 1.8,
         "Input\n4 frames × 6 bands\n[4,6,128,128]", blue, fs=11)

    _box(ax, 3.3, 6.6, 2.6, 1.4, "Multi-scale stem\n(1/3/5/7 conv)", green, fs=10)
    _box(ax, 3.3, 4.6, 2.6, 1.6, "Encoder\n64→128→256→512\nResBlock + CBAM", green, fs=10)
    _box(ax, 3.3, 2.4, 2.6, 1.6, "Bottleneck\n512  (8×8)", green, fs=10)

    _box(ax, 0.3, 1.0, 2.4, 1.9,
         "Meta-Learner\nmotion diffs → MLP\n→ FiLM (γ, β)", purple, fs=10)

    _box(ax, 6.6, 4.4, 2.7, 1.8,
         "Decoder\n512→256→128→64\nskip + ResBlock", green, fs=10)
    _box(ax, 9.9, 4.5, 2.4, 1.6, "Δ head\n(residual field)\n[3,6,128,128]", orange, fs=10)

    _box(ax, 9.9, 7.0, 2.4, 1.3, "Last frame\n(persistence prior)", gray, fs=10)
    _box(ax, 12.9, 5.6, 2.3, 1.5, "ŷ = last + γ·Δ\n(residual add)", red, fs=11)

    _box(ax, 12.9, 3.2, 2.3, 1.7,
         "Diffusion-style\niterative refine\n(K gated steps)", orange, fs=10)
    _box(ax, 15.6, 4.4, 2.1, 1.6, "Forecast\n3 frames × 6\n[3,6,128,128]", blue, fs=10)

    # arrows
    _arrow(ax, (2.7, 5.2), (3.3, 7.3))             # input->stem
    _arrow(ax, (4.6, 6.6), (4.6, 6.2))             # stem->encoder
    _arrow(ax, (4.6, 4.6), (4.6, 4.0))             # encoder->bottleneck
    _arrow(ax, (2.7, 2.0), (3.3, 2.9), color="#7a4fa3", rad=-0.2)  # meta->bottleneck
    _arrow(ax, (5.9, 3.2), (6.6, 4.8), rad=-0.2)   # bottleneck->decoder
    _arrow(ax, (9.3, 5.3), (9.9, 5.3))             # decoder->delta
    _arrow(ax, (12.3, 5.3), (12.9, 6.0))           # delta->add
    _arrow(ax, (11.1, 7.0), (13.4, 7.1), rad=-0.1) # lastframe->add
    _arrow(ax, (14.0, 5.6), (14.0, 4.9))           # add->refine
    _arrow(ax, (15.2, 4.0), (15.7, 4.6), rad=-0.2) # refine->forecast
    _arrow(ax, (2.7, 4.8), (10.5, 7.0), color="#888", lw=1.2, rad=0.25)  # input->lastframe

    ax.text(9, 0.4,
            "Physics-informed objective:  L1  +  (1−SSIM)  +  gradient  +  "
            "temporal-smoothness  +  divergence  +  advection-consistency  +  TV",
            ha="center", fontsize=12, style="italic", color="#444")
    fig.savefig(os.path.join(OUT, "fig_architecture.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, "fig_architecture.png"))


def pipeline():
    fig, ax = plt.subplots(figsize=(18, 4.5))
    ax.set_xlim(0, 18); ax.set_ylim(0, 4.5); ax.axis("off")
    ax.set_title("End-to-End Pipeline", fontsize=18, fontweight="bold", pad=10)
    steps = [
        ("INSAT multispectral\nsequences (.pt)", "#cfe3f7"),
        ("128px tensor cache\n(prep_cache)", "#e8e8e8"),
        ("PI-Conditional\nForecaster + Meta", "#cce8cc"),
        ("Physics-informed\ntraining (AMP, GPU)", "#ffe0b3"),
        ("Evaluation\nSSIM / PSNR", "#e6d4f2"),
        ("Top-tier figures\n+ REPORT.md", "#f6c6c6"),
    ]
    w, gap, y, h = 2.5, 0.55, 1.6, 1.4
    x = 0.4
    for i, (txt, fc) in enumerate(steps):
        _box(ax, x, y, w, h, txt, fc, fs=10)
        if i < len(steps) - 1:
            _arrow(ax, (x + w, y + h / 2), (x + w + gap, y + h / 2))
        x += w + gap
    fig.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, "fig_pipeline.png"))


if __name__ == "__main__":
    architecture()
    pipeline()
