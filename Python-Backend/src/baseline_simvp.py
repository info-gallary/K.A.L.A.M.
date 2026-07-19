"""
SimVP-style learned baseline, trained under OUR protocol (same leakage-free
chronological split, cache, augmentation, loss, epochs) for a fair head-to-head
against the physics-informed forecaster. SimVP = Conv encoder -> Inception-style
conv translator -> Conv decoder (Gao et al., CVPR 2022), the standard strong,
simple video-prediction baseline.

    python baseline_simvp.py --epochs 80          # train + eval on held-out test
"""
import os, glob, time, json, argparse, warnings, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from contextlib import nullcontext
warnings.filterwarnings("ignore")
from data_split import chrono_split
from pidm_losses import CombinedLoss, SSIM, ssim_metric, psnr_metric
from skimage.metrics import structural_similarity as sk_ssim

CACHE, OUT = "cache_128", "simvp.pth"
T_IN, T_OUT, C = 4, 3, 6


class CachedSeq(Dataset):
    def __init__(self, files): self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = torch.load(self.files[i], map_location="cpu", weights_only=False)
        return d["input"].float(), d["target"].float()


def conv(cin, cout, k=3, s=1, t=False):
    p = k // 2
    layer = (nn.ConvTranspose2d(cin, cout, k, s, p, output_padding=s - 1) if t
             else nn.Conv2d(cin, cout, k, s, p))
    return nn.Sequential(layer, nn.GroupNorm(8, cout), nn.SiLU())


class Inception(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        b = cout // 4
        self.b1 = nn.Conv2d(cin, b, 1)
        self.b3 = nn.Conv2d(cin, b, 3, padding=1)
        self.b5 = nn.Conv2d(cin, b, 5, padding=2)
        self.b7 = nn.Conv2d(cin, cout - 3 * b, 7, padding=3)
        self.norm = nn.GroupNorm(8, cout); self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(torch.cat([self.b1(x), self.b3(x), self.b5(x), self.b7(x)], 1)))


class SimVP(nn.Module):
    """SimVP with residual-from-persistence output (standard, for a fair & strong
    baseline that doesn't collapse on small data)."""
    def __init__(self, in_ch=T_IN * C, out_ch=T_OUT * C, hid=128, n_trans=6):
        super().__init__()
        self.enc = nn.Sequential(conv(in_ch, 64), conv(64, hid, s=2), conv(hid, hid, s=2))
        self.trans = nn.Sequential(*[Inception(hid, hid) for _ in range(n_trans)])
        self.up = nn.Sequential(conv(hid, hid, s=2, t=True), conv(hid, 64, s=2, t=True))
        self.head = nn.Conv2d(64, out_ch, 3, padding=1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)  # start at persistence

    def forward(self, x):                 # x: [B, T_IN, C, H, W]
        B, T, c, H, W = x.shape
        h = self.up(self.trans(self.enc(x.reshape(B, T * c, H, W))))
        delta = self.head(h)[..., :H, :W].view(B, T_OUT, C, H, W)
        last = x[:, -1:].expand(-1, T_OUT, -1, -1, -1)
        return (last + delta).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hours", type=float, default=2.5)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
    tr, va, te = chrono_split(len(files))
    tl = DataLoader(Subset(CachedSeq(files), tr), batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=False)
    vl = DataLoader(Subset(CachedSeq(files), va), batch_size=args.batch, num_workers=2,
                    persistent_workers=False)

    model = SimVP().to(device)
    print(f"[simvp] params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = CombinedLoss(w_msssim=2.0, lead_weights=[0.6, 1.0, 1.4]).to(device)  # SAME loss as ours
    ssim_mod = SSIM().to(device)
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda" else nullcontext

    best = -1
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for inp, tgt in tl:
            inp = inp.to(device, non_blocking=True); tgt = tgt.to(device, non_blocking=True)
            if torch.rand(1).item() < 0.5:        # same flip augmentation
                inp = torch.flip(inp, [-1]); tgt = torch.flip(tgt, [-1])
            opt.zero_grad(set_to_none=True)
            with ac():
                loss, _ = crit(model(inp), tgt, inp)
            if not torch.isfinite(loss): continue
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval(); vs, nb = 0.0, 0
        with torch.no_grad():
            for inp, tgt in vl:
                inp = inp.to(device); tgt = tgt.to(device)
                with ac(): pr = model(inp)
                vs += ssim_metric(pr.float(), tgt, ssim_mod).item(); nb += 1
        vs /= nb
        el = (time.time() - t0) / 3600
        print(f"[simvp] EPOCH {ep+1}/{args.epochs} | val_ssim {vs:.4f} | {el:.2f}h", flush=True)
        if vs > best:
            best = vs; torch.save({"model": model.state_dict(), "epoch": ep, "val_ssim": vs}, OUT)
        if el >= args.hours:
            print("[simvp] time cap"); break

    # held-out TEST eval (skimage SSIM + TTA, same as ours)
    ck = torch.load(OUT, map_location=device, weights_only=False); model.load_state_dict(ck["model"]); model.eval()
    def tta(x):
        x = x.to(device); outs = [model(x)]
        for d in ([-1], [-2], [-1, -2]): outs.append(torch.flip(model(torch.flip(x, d)), d))
        return torch.stack(outs).mean(0)
    S = np.zeros((T_OUT, C)); n = 0
    with torch.no_grad():
        for idx in te:
            d = torch.load(files[idx], map_location="cpu", weights_only=False)
            inp = d["input"].float().unsqueeze(0); g = d["target"].float().numpy()
            pr = tta(inp).float().cpu().numpy()[0]
            for t in range(T_OUT):
                for c in range(C):
                    S[t, c] += sk_ssim(np.clip(g[t, c], 0, 1).astype(np.float64),
                                       np.clip(pr[t, c], 0, 1).astype(np.float64),
                                       data_range=1.0, gaussian_weights=True, win_size=11,
                                       use_sample_covariance=False)
            n += 1
    S /= n
    res = {"model": "SimVP (our protocol)", "test_ssim": float(S.mean()),
           "per_lead": [float(S[t].mean()) for t in range(T_OUT)], "val_ssim": best,
           "params": sum(p.numel() for p in model.parameters())}
    json.dump(res, open("results/baseline_simvp.json", "w"), indent=2)
    print(f"[simvp] TEST SSIM={res['test_ssim']:.4f}  per-lead={[round(x,3) for x in res['per_lead']]}")
    print(f"[simvp] (ours = 0.729; SimVP under identical protocol = {res['test_ssim']:.4f})")


if __name__ == "__main__":
    main()
