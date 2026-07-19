"""
Train the Physics-Informed Conditional Forecaster + Meta-Learner.

- Loads the 128x128 tensor cache (run prep_cache.py first).
- Mixed-precision training on GPU, AdamW + warmup-cosine schedule.
- Residual-from-last-frame model => strong SSIM/PSNR prior.
- Per-epoch checkpoint (resumable) + greppable metric log + history.json.

Usage:
    python train_pidm.py --epochs 250 --batch 8 --hours 9
    python train_pidm.py --resume          # continue from pidm_last.pth
"""
import os, sys, glob, json, time, math, argparse, warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

warnings.filterwarnings("ignore")
from pidm_model import PIConditionalForecaster
from pidm_losses import CombinedLoss, SSIM, ssim_metric, psnr_metric
from data_split import chrono_split

CACHE = "cache_128"
BEST = "pidm_best.pth"
LAST = "pidm_last.pth"
HIST = "pidm_history.json"
LOG = "train_log.txt"


# --------------------------------------------------------------------------- #
class CachedSeq(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.files[i], map_location="cpu", weights_only=False)
        return d["input"].float(), d["target"].float()


from contextlib import nullcontext


def amp_ctx(device):
    """bf16 autocast on CUDA (Ampere+): full fp32 exponent range => no gradient
    overflow and NO GradScaler needed (fp16 scaler can collapse to 0)."""
    if device == "cuda":
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return lambda: nullcontext()


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hours", type=float, default=9.0)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="", help="suffix for output files (ablations)")
    ap.add_argument("--no_residual", action="store_true")
    ap.add_argument("--no_meta", action="store_true")
    ap.add_argument("--no_refine", action="store_true")
    ap.add_argument("--no_physics", action="store_true")
    ap.add_argument("--w_msssim", type=float, default=0.0)
    ap.add_argument("--lead_weights", type=str, default="",
                    help="comma-separated per-lead-time loss weights, e.g. 0.6,1.0,1.4")
    ap.add_argument("--augment", action="store_true", help="random flip augmentation")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay (0=off)")
    ap.add_argument("--w_ssl", type=float, default=0.0,
                    help="self-supervised flip-equivariance consistency weight (0=off)")
    ap.add_argument("--w_pde", type=float, default=0.0,
                    help="PINN advection-PDE residual weight (0=off)")
    args = ap.parse_args()
    lead_w = [float(x) for x in args.lead_weights.split(",")] if args.lead_weights else None

    global BEST, LAST, HIST, LOG
    if args.tag:
        BEST = f"pidm_best_{args.tag}.pth"; LAST = f"pidm_last_{args.tag}.pth"
        HIST = f"pidm_history_{args.tag}.json"; LOG = f"train_log_{args.tag}.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[init] device={device}  torch={torch.__version__}")
    if device == "cuda":
        log(f"[init] gpu={torch.cuda.get_device_name(0)}  "
            f"vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    torch.manual_seed(args.seed)

    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
    if not files:
        log(f"[error] no cache found in {CACHE}/. Run prep_cache.py first.")
        sys.exit(1)
    ds = CachedSeq(files)
    tr_idx, va_idx, te_idx = chrono_split(len(files))   # leakage-free, chronological
    tr = Subset(ds, tr_idx)
    va = Subset(ds, va_idx)
    log(f"[data] total={len(ds)}  train={len(tr)}  val={len(va)}  "
        f"test(held-out)={len(te_idx)}  (chronological, gap-buffered)")

    pin = device == "cuda"
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=pin,
                    persistent_workers=args.workers > 0, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False,
                    num_workers=max(1, args.workers // 2), pin_memory=pin,
                    persistent_workers=args.workers > 0)

    model = PIConditionalForecaster(
        base=args.base, refine_steps=args.refine,
        use_residual=not args.no_residual, use_meta=not args.no_meta,
        use_refine=not args.no_refine).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"[model] PIConditionalForecaster base={args.base} params={nparam:,} "
        f"residual={not args.no_residual} meta={not args.no_meta} "
        f"refine={not args.no_refine} physics={not args.no_physics}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5,
                            betas=(0.9, 0.999))
    warmup = 8

    def lr_at(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        p = (ep - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    autocast = amp_ctx(device)
    crit = (CombinedLoss(w_msssim=args.w_msssim, lead_weights=lead_w, w_pde=0,
                         w_grad=0, w_temporal=0, w_tv=0, w_div=0, w_adv=0)
            if args.no_physics
            else CombinedLoss(w_msssim=args.w_msssim, lead_weights=lead_w,
                              w_pde=args.w_pde)).to(device)
    ssim_mod = SSIM().to(device)

    history = {"train_loss": [], "val_ssim": [], "val_psnr": [], "lr": []}
    start_ep, best = 0, -1e9
    if args.resume and os.path.exists(LAST):
        ck = torch.load(LAST, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        sched.base_lrs = [args.lr for _ in opt.param_groups]   # honor lowered --lr
        start_ep = ck["epoch"] + 1
        best = ck.get("best", best)
        history = ck.get("history", history)
        log(f"[resume] from epoch {start_ep}  best_score={best:.4f}")

    # init EMA AFTER any resume so it copies the loaded (warm-start) weights
    ema = ({k: v.detach().clone().float() for k, v in model.state_dict().items()}
           if args.ema > 0 else None)

    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for inp, tgt in tl:
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            if args.augment:                              # flip augmentation
                if torch.rand(1).item() < 0.5:
                    inp = torch.flip(inp, [-1]); tgt = torch.flip(tgt, [-1])
                if torch.rand(1).item() < 0.5:
                    inp = torch.flip(inp, [-2]); tgt = torch.flip(tgt, [-2])
            opt.zero_grad(set_to_none=True)
            with autocast():
                pred = model(inp)
                loss, _ = crit(pred, tgt, inp)
                if args.w_ssl > 0:                        # SSL flip-equivariance (sub-batch for VRAM)
                    s = max(1, inp.shape[0] // 3)
                    pred_f = model(torch.flip(inp[:s], [-1]))
                    ssl = (pred_f - torch.flip(pred[:s], [-1]).detach()).abs().mean()
                    loss = loss + args.w_ssl * ssl
            if not torch.isfinite(loss):
                continue                                  # skip bad batch
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True); continue  # skip gradient spike
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(args.ema).add_(v.detach().float(), alpha=1 - args.ema)
            tot += loss.item(); nb += 1
        sched.step()
        train_loss = tot / max(1, nb)

        # ---- validation (with EMA weights if enabled) ----
        model.eval()
        raw = None
        if ema is not None:
            raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict({k: ema[k].to(v.dtype) for k, v in raw.items()})
        vs, vp, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for inp, tgt in vl:
                inp = inp.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                with autocast():
                    pred = model(inp)
                vs += ssim_metric(pred.float(), tgt, ssim_mod).item()
                vp += psnr_metric(pred.float(), tgt).item()
                vn += 1
        val_ssim, val_psnr = vs / vn, vp / vn
        score = val_ssim + val_psnr / 50.0

        history["train_loss"].append(train_loss)
        history["val_ssim"].append(val_ssim)
        history["val_psnr"].append(val_psnr)
        history["lr"].append(opt.param_groups[0]["lr"])
        with open(HIST, "w") as f:
            json.dump(history, f)

        elapsed = (time.time() - t0) / 3600
        flag = ""
        if val_ssim >= 0.80 and val_psnr >= 24.0:
            flag = "  <== TARGET MET"
        log(f"EPOCH {ep+1}/{args.epochs} | loss {train_loss:.4f} | "
            f"val_ssim {val_ssim:.4f} | val_psnr {val_psnr:.2f} | "
            f"lr {opt.param_groups[0]['lr']:.2e} | {elapsed:.2f}h{flag}")

        # checkpoint stores the EMA weights (the evaluated ones) when EMA is on
        save_model = ({k: ema[k].to(raw[k].dtype) for k in ema} if ema is not None
                      else model.state_dict())
        ck = {"epoch": ep, "model": save_model, "opt": opt.state_dict(),
              "sched": sched.state_dict(),
              "best": best, "history": history,
              "val_ssim": val_ssim, "val_psnr": val_psnr, "args": vars(args)}
        torch.save(ck, LAST)
        if score > best:
            best = score
            ck["best"] = best
            torch.save(ck, BEST)
            log(f"  -> new BEST score={best:.4f} (ssim={val_ssim:.4f}, psnr={val_psnr:.2f})")
        if ema is not None:                               # restore raw weights for training
            model.load_state_dict(raw)

        if elapsed >= args.hours:
            log(f"[stop] wall-clock budget {args.hours}h reached at epoch {ep+1}")
            break

    log(f"[done] best_score={best:.4f}  total={ (time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    main()
