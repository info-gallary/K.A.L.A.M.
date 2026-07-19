"""
Explainable-AI (XAI) for the forecaster. Post-hoc, no retraining. Runs on CPU by
default so it can run alongside GPU training.

Produces (results/):
  fig_xai_saliency.png         input-gradient saliency per observed frame (what
                               input regions drive the forecast)
  fig_xai_attention.png        CBAM spatial-attention maps across encoder depth
                               (where the model attends)
  fig_xai_attribution.png      per-input-frame and per-band gradient attribution
"""
import os, glob, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")
from data_split import chrono_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pidm_model import PIConditionalForecaster, SpatialAttention

CACHE, BEST, OUT = "cache_128", "pidm_best.pth", "results"
BANDS = ["TIR1", "TIR2", "WV", "VIS", "MIR", "SWIR"]
VIS = 3


def load(device):
    ck = torch.load(BEST, map_location=device, weights_only=False)
    a = ck.get("args", {})
    m = PIConditionalForecaster(base=a.get("base", 64), refine_steps=a.get("refine", 2)).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m


def train_sample():
    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
    tr, _, _ = chrono_split(len(files))
    i = tr[len(tr) // 2]
    d = torch.load(files[i], map_location="cpu", weights_only=False)
    return i, d["input"].float(), d["target"].float()


def saliency(model, inp, device):
    x = inp.unsqueeze(0).to(device).requires_grad_(True)
    out = model(x)
    score = out.mean()
    model.zero_grad()
    score.backward()
    g = x.grad.detach()[0].abs()                      # [4,6,H,W]
    return g.cpu().numpy(), out.detach()


def get_attention_maps(model, inp, device):
    maps = []
    handles = []

    def hook(mod, inp_t, out_t):
        x = inp_t[0]
        avg = x.mean(1, keepdim=True); mx, _ = x.max(1, keepdim=True)
        att = mod.sigmoid(mod.conv(torch.cat([avg, mx], 1)))
        maps.append(att.detach()[0, 0].cpu().numpy())

    for mod in model.modules():
        if isinstance(mod, SpatialAttention):
            handles.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        model(inp.unsqueeze(0).to(device))
    for h in handles:
        h.remove()
    return maps


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cpu"           # safe alongside GPU training
    model = load(device)
    idx, inp, tgt = train_sample()
    print(f"[xai] explaining training sample #{idx} on {device}")

    sal, _ = saliency(model, inp, device)             # [4,6,H,W]
    inp_np = inp.numpy()

    # --- saliency per observed frame (summed over bands) ---
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for t in range(4):
        ax[0, t].imshow(inp_np[t, VIS], cmap="gray", vmin=0, vmax=1)
        ax[0, t].set_title(f"Input t{t-3} ({BANDS[VIS]})"); ax[0, t].axis("off")
        s = sal[t].sum(0); s = s / (s.max() + 1e-8)
        im = ax[1, t].imshow(s, cmap="inferno"); ax[1, t].set_title(f"Saliency t{t-3}")
        ax[1, t].axis("off")
    fig.colorbar(im, ax=ax[1, :].tolist(), fraction=0.02)
    fig.suptitle("XAI: input-gradient saliency — which observed regions drive the forecast",
                 fontsize=15, fontweight="bold")
    fig.savefig(f"{OUT}/fig_xai_saliency.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    # --- attention maps across depth ---
    maps = get_attention_maps(model, inp, device)
    k = min(6, len(maps))
    sel = [maps[i] for i in np.linspace(0, len(maps) - 1, k).astype(int)] if maps else []
    if sel:
        fig, ax = plt.subplots(1, k, figsize=(4 * k, 4))
        if k == 1:
            ax = [ax]
        for j, mp in enumerate(sel):
            ax[j].imshow(mp, cmap="viridis"); ax[j].set_title(f"CBAM attn #{j+1}\n{mp.shape[0]}x{mp.shape[1]}")
            ax[j].axis("off")
        fig.suptitle("XAI: CBAM spatial-attention maps across encoder/decoder depth",
                     fontsize=15, fontweight="bold")
        fig.savefig(f"{OUT}/fig_xai_attention.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    # --- per-frame and per-band attribution ---
    frame_attr = sal.sum((1, 2, 3)); frame_attr /= frame_attr.sum() + 1e-8
    band_attr = sal.sum((0, 2, 3)); band_attr /= band_attr.sum() + 1e-8
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].bar([f"t{t-3}" for t in range(4)], frame_attr, color="#2ca02c")
    ax[0].set_title("Attribution by input frame"); ax[0].set_ylabel("relative importance")
    ax[1].bar(BANDS, band_attr, color="#ff7f0e")
    ax[1].set_title("Attribution by spectral band"); ax[1].set_ylabel("relative importance")
    fig.suptitle("XAI: gradient attribution — input frame & band importance",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_xai_attribution.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] frame importance {dict(zip(['t-3','t-2','t-1','t0'], np.round(frame_attr,3)))}")
    print(f"[xai] band importance  {dict(zip(BANDS, np.round(band_attr,3)))}")
    print(f"[xai] wrote 3 XAI figures to {OUT}/")


if __name__ == "__main__":
    main()
