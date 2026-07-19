"""
Fine-tuning stage: resume the headline model and squeeze extra SSIM via
(i) multi-scale SSIM objective and (ii) weight EMA, at a low learning rate.
The EMA weights are evaluated and saved as pidm_finetuned.pth (used by eval if
present). Run AFTER the headline run, on the free GPU.

    python finetune.py --epochs 40 --lr 3e-5 --w_msssim 1.2
"""
import os, glob, json, time, math, argparse, warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from contextlib import nullcontext
warnings.filterwarnings("ignore")
from pidm_model import PIConditionalForecaster
from pidm_losses import CombinedLoss, SSIM, ssim_metric, psnr_metric
from data_split import chrono_split

CACHE, SRC, OUT, HIST = "cache_128", "pidm_best.pth", "pidm_finetuned.pth", "pidm_finetune_history.json"


class CachedSeq(Dataset):
    def __init__(self, files): self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = torch.load(self.files[i], map_location="cpu", weights_only=False)
        return d["input"].float(), d["target"].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--w_msssim", type=float, default=1.2)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--src", type=str, default=SRC)
    ap.add_argument("--lead_weights", type=str, default="0.6,1.0,1.4")
    args = ap.parse_args()
    lead_w = [float(x) for x in args.lead_weights.split(",")] if args.lead_weights else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
    tr_idx, va_idx, _ = chrono_split(len(files))
    tl = DataLoader(Subset(CachedSeq(files), tr_idx), batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(Subset(CachedSeq(files), va_idx), batch_size=args.batch, num_workers=2,
                    persistent_workers=True)

    ck = torch.load(args.src, map_location=device, weights_only=False)
    a = ck.get("args", {})
    model = PIConditionalForecaster(base=a.get("base", 64),
                                    refine_steps=a.get("refine", 2)).to(device)
    model.load_state_dict(ck["model"])
    print(f"[ft] loaded {args.src} (epoch {ck.get('epoch')}, "
          f"val_ssim {ck.get('val_ssim', float('nan')):.4f})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = CombinedLoss(w_msssim=args.w_msssim, lead_weights=lead_w).to(device)
    ssim_mod = SSIM().to(device)
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda" else nullcontext

    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    best, hist = -1e9, {"val_ssim": [], "val_psnr": []}
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for inp, tgt in tl:
            inp = inp.to(device, non_blocking=True); tgt = tgt.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with ac():
                loss, _ = crit(model(inp), tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(args.ema).add_(v.detach().float(), alpha=1 - args.ema)
        sched.step()

        # validate with EMA weights
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: ema[k].to(v.dtype) for k, v in backup.items()})
        model.eval()
        vs, vp, nb = 0.0, 0.0, 0
        with torch.no_grad():
            for inp, tgt in vl:
                inp = inp.to(device); tgt = tgt.to(device)
                with ac():
                    pr = model(inp)
                vs += ssim_metric(pr.float(), tgt, ssim_mod).item()
                vp += psnr_metric(pr.float(), tgt).item(); nb += 1
        vs /= nb; vp /= nb
        hist["val_ssim"].append(vs); hist["val_psnr"].append(vp)
        json.dump(hist, open(HIST, "w"))
        score = vs + vp / 50
        flag = ""
        if score > best:
            best = score
            torch.save({"model": {k: ema[k].to(backup[k].dtype) for k in ema},
                        "epoch": ck.get("epoch", 0) + ep, "args": a,
                        "val_ssim": vs, "val_psnr": vp}, OUT)
            flag = "  -> saved EMA best"
        model.load_state_dict(backup)   # restore raw weights for next epoch
        print(f"[ft] EPOCH {ep+1}/{args.epochs} | EMA val_ssim {vs:.4f} | "
              f"val_psnr {vp:.2f} | {(time.time()-t0)/3600:.2f}h{flag}", flush=True)
    print(f"[ft] done. best EMA score={best:.4f} -> {OUT}")


if __name__ == "__main__":
    main()
