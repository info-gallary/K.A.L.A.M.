"""
ConvLSTM seq2seq learned baseline, trained under OUR protocol (same leakage-free
split, cache, augmentation, loss, budget) for a fair head-to-head. ConvLSTM
(Shi et al., NeurIPS 2015) is the canonical recurrent nowcasting baseline.

    python baseline_convlstm.py --epochs 80
"""
import os, glob, time, json, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from contextlib import nullcontext
warnings.filterwarnings("ignore")
from data_split import chrono_split
from pidm_losses import CombinedLoss, SSIM, ssim_metric
from skimage.metrics import structural_similarity as sk_ssim

CACHE, OUT = "cache_128", "convlstm.pth"
T_IN, T_OUT, C = 4, 3, 6


class CachedSeq(Dataset):
    def __init__(self, files): self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = torch.load(self.files[i], map_location="cpu", weights_only=False)
        return d["input"].float(), d["target"].float()


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv2d(in_ch + hid, 4 * hid, 3, padding=1)

    def forward(self, x, h, c):
        z = self.conv(torch.cat([x, h], 1))
        i, f, o, g = z.chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, ch=C, hid=128, layers=2):
        super().__init__()
        self.hid, self.layers = hid, layers
        self.cells = nn.ModuleList(
            [ConvLSTMCell(ch if i == 0 else hid, hid) for i in range(layers)])
        self.head = nn.Sequential(nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid),
                                  nn.SiLU(), nn.Conv2d(hid, ch, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)  # residual init

    def forward(self, x):                 # x: [B, T_IN, C, H, W]
        B, T, c, H, W = x.shape
        hs = [x.new_zeros(B, self.hid, H, W) for _ in range(self.layers)]
        cs = [x.new_zeros(B, self.hid, H, W) for _ in range(self.layers)]
        # encode observed frames
        for t in range(T):
            inp = x[:, t]
            for li, cell in enumerate(self.cells):
                hs[li], cs[li] = cell(inp, hs[li], cs[li]); inp = hs[li]
        # decode autoregressively; residual-from-persistence output (fair, no collapse)
        last_obs = x[:, -1]
        outs = []; prev = last_obs
        for _ in range(T_OUT):
            inp = prev
            for li, cell in enumerate(self.cells):
                hs[li], cs[li] = cell(inp, hs[li], cs[li]); inp = hs[li]
            y = (last_obs + self.head(hs[-1])).clamp(0, 1)
            outs.append(y); prev = y
        return torch.stack(outs, 1)        # [B, T_OUT, C, H, W]


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

    model = ConvLSTM().to(device)
    print(f"[convlstm] params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = CombinedLoss(w_msssim=2.0, lead_weights=[0.6, 1.0, 1.4]).to(device)
    ssim_mod = SSIM().to(device)
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda" else nullcontext

    best = -1; t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for inp, tgt in tl:
            inp = inp.to(device, non_blocking=True); tgt = tgt.to(device, non_blocking=True)
            if torch.rand(1).item() < 0.5:
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
        print(f"[convlstm] EPOCH {ep+1}/{args.epochs} | val_ssim {vs:.4f} | {(time.time()-t0)/3600:.2f}h", flush=True)
        if vs > best:
            best = vs; torch.save({"model": model.state_dict(), "epoch": ep, "val_ssim": vs}, OUT)
        if (time.time()-t0)/3600 >= args.hours:
            print("[convlstm] time cap"); break

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
    res = {"model": "ConvLSTM (our protocol)", "test_ssim": float(S.mean()),
           "per_lead": [float(S[t].mean()) for t in range(T_OUT)], "val_ssim": best,
           "params": sum(p.numel() for p in model.parameters())}
    json.dump(res, open("results/baseline_convlstm.json", "w"), indent=2)
    print(f"[convlstm] TEST SSIM={res['test_ssim']:.4f}  per-lead={[round(x,3) for x in res['per_lead']]}")


if __name__ == "__main__":
    main()
