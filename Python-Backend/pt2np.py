import os
import torch
import numpy as np
from glob import glob

SOURCE_DIR = "./diffusion_dataset"
DEST_DIR = "./cloudchase_dataset"
os.makedirs(DEST_DIR, exist_ok=True)

# Get all .pt files
pt_files = sorted(glob(os.path.join(SOURCE_DIR, "sequence_*.pt")))

for idx, pt_file in enumerate(pt_files):
    data = torch.load(pt_file)
    input_tensor = data["input"]   # [4, 6, H, W]
    target_tensor = data["target"] # [3, 6, H, W]

    sample_dir = os.path.join(DEST_DIR, f"sample_{idx:03d}")
    os.makedirs(sample_dir, exist_ok=True)

    # Save input frames
    for i in range(input_tensor.shape[0]):
        np.save(os.path.join(sample_dir, f"input_{i}.npy"), input_tensor[i].numpy())

    # Save target frames
    for i in range(target_tensor.shape[0]):
        np.save(os.path.join(sample_dir, f"target_{i}.npy"), target_tensor[i].numpy())

    print(f"✅ Converted {pt_file} → {sample_dir}")

print("\n🎉 All sequences converted to .npy format successfully!")
