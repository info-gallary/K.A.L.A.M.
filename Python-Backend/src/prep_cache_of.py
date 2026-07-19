"""Precompute the Lagrangian (optical-flow advection) prior for each cached
sequence: Farneback flow from the last two observed frames, used to advect the
last frame to t+1, t+2, t+3. Stored as of_prior/seq_XXXXX.pt -> [3,6,128,128].

CPU only (runs in parallel with GPU training). Idempotent/resumable.
"""
import os, glob, warnings
import numpy as np
import torch
import cv2
warnings.filterwarnings("ignore")

SRC, DST, T, C, HW = "cache_128", "of_prior", 3, 6, 128


def of_prior(inp):                      # inp: [4,C,H,W] float[0,1] -> [T,C,H,W]
    out = np.zeros((T, C, HW, HW), np.float32)
    gx, gy = np.meshgrid(np.arange(HW), np.arange(HW))
    for c in range(C):
        prev = (np.clip(inp[-2, c], 0, 1) * 255).astype(np.uint8)
        curr = (np.clip(inp[-1, c], 0, 1) * 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        base = np.clip(inp[-1, c], 0, 1).astype(np.float32)
        for t in range(T):
            mx = (gx + (t + 1) * flow[..., 0]).astype(np.float32)
            my = (gy + (t + 1) * flow[..., 1]).astype(np.float32)
            out[t, c] = cv2.remap(base, mx, my, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    return np.clip(out, 0, 1)


def main():
    os.makedirs(DST, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.pt")))
    print(f"Building advection prior for {len(files)} sequences -> {DST}/")
    done = 0
    for i, f in enumerate(files):
        out = os.path.join(DST, os.path.basename(f))
        if os.path.exists(out):
            done += 1; continue
        d = torch.load(f, map_location="cpu", weights_only=False)
        p = of_prior(d["input"].float().numpy())
        torch.save({"of_prior": torch.from_numpy(p).half()}, out)
        done += 1
        if (i + 1) % 150 == 0 or i == len(files) - 1:
            print(f"  {i+1}/{len(files)} done", flush=True)
    print(f"Done. {done} priors in {DST}/")


if __name__ == "__main__":
    main()
