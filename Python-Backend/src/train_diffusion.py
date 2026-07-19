"""Train the conditional residual diffusion refiner on top of the frozen
deterministic forecaster (pidm_best.pth). The deterministic model provides the
prior mean ŷ (conditioning); the DDPM learns the residual distribution.

    python train_diffusion.py --epochs 40
"""
import os, glob, json, time, math, argparse, warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from contextlib import nullcontext
warnings.filterwarnings("ignore")
from pidm_model import PIConditionalForecaster
from pidm_diffusion import ConditionalResidualDiffusion
from data_split import chrono_split

CACHE, DET, OUT, HIST = "cache_128", "pidm_best.pth", "diffusion.pth", "diffusion_history.json"


class CachedSeq(Dataset):
    def __init__(self, files): self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = torch.load(self.files[i], map_location="cpu", weights_only=False)
        return d["input"].float(), d["target"].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--hours", type=float, default=1.2)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
    tr, va, _ = chrono_split(len(files))
    tl = DataLoader(Subset(CachedSeq(files), tr), batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)

    ck = torch.load(DET, map_location=device, weights_only=False)
    a = ck.get("args", {})
    det = PIConditionalForecaster(base=a.get("base", 64),
                                  refine_steps=a.get("refine", 2)).to(device)
    det.load_state_dict(ck["model"]); det.eval()
    for p in det.parameters():
        p.requires_grad_(False)
    print(f"[diff] frozen deterministic model loaded ({DET})")

    diff = ConditionalResidualDiffusion(base=args.base).to(device)
    nparam = sum(p.numel() for p in diff.parameters())
    print(f"[diff] diffusion denoiser params: {nparam:,}")
    opt = torch.optim.AdamW(diff.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda" else nullcontext

    hist = {"loss": []}
    t0 = time.time()
    for ep in range(args.epochs):
        diff.train(); tot, nb = 0.0, 0
        for inp, tgt in tl:
            inp = inp.to(device, non_blocking=True); tgt = tgt.to(device, non_blocking=True)
            with torch.no_grad(), ac():
                ydet = det(inp).float()
            opt.zero_grad(set_to_none=True)
            with ac():
                loss = diff.loss(inp, ydet, tgt)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(diff.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        avg = tot / max(1, nb)
        hist["loss"].append(avg); json.dump(hist, open(HIST, "w"))
        el = (time.time() - t0) / 3600
        print(f"[diff] EPOCH {ep+1}/{args.epochs} | ddpm_loss {avg:.4f} | "
              f"lr {opt.param_groups[0]['lr']:.2e} | {el:.2f}h", flush=True)
        torch.save({"model": diff.state_dict(), "epoch": ep, "base": args.base}, OUT)
        if el >= args.hours:
            print(f"[diff] stop at {args.hours}h"); break
    print(f"[diff] done -> {OUT}  ({(time.time()-t0)/3600:.2f}h)")


if __name__ == "__main__":
    main()
