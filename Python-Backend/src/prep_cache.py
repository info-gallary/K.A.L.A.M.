"""One-time preprocessing: resize the tensor sequences in diffusion_dataset/ to a
compact 128x128 float16 cache so training epochs are fast (1.3 MB vs 33 MB/file).

Idempotent / resumable: skips sequences already cached.
"""
import os, glob, sys, warnings
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

SRC = "diffusion_dataset"
DST = "cache_128"
SIZE = 128


def resize(x, size=SIZE):                      # x: [T, C, H, W] -> [T, C, s, s]
    T, C, H, W = x.shape
    x = F.interpolate(x.reshape(T * C, 1, H, W).float(), size=(size, size),
                      mode="bilinear", align_corners=False)
    return x.reshape(T, C, size, size)


def main():
    os.makedirs(DST, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.pt")))
    print(f"Found {len(files)} source sequences -> caching to {DST}/ at {SIZE}px")
    done = 0
    for i, f in enumerate(files):
        out = os.path.join(DST, f"seq_{i:05d}.pt")
        if os.path.exists(out):
            done += 1
            continue
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
            inp = resize(d["input"].float()).clamp(0, 1).half()
            tgt = resize(d["target"].float()).clamp(0, 1).half()
            torch.save({"input": inp, "target": tgt}, out)
            done += 1
        except Exception as e:
            print(f"  ! skip {f}: {e}")
        if (i + 1) % 100 == 0 or i == len(files) - 1:
            print(f"  {i+1}/{len(files)} cached ({done} ready)", flush=True)
    print(f"Done. {done} cached sequences in {DST}/")


if __name__ == "__main__":
    main()
