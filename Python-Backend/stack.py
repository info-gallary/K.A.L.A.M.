# import os
# import numpy as np
# from glob import glob
# from PIL import Image
# import torch
# from tqdm import tqdm

# BASE_DIR = r"/home/23bce370/Desktop/ISRO/pre_final/processed_samples/"
# BAND_NAMES = ["IMG_VIS", "IMG_MIR", "IMG_SWIR", "IMG_TIR1", "IMG_TIR2", "IMG_WV"]
# FRAMES_IN = 4
# FRAMES_OUT = 3
# SAVE_DIR = r"./diffusion_dataset"

# os.makedirs(SAVE_DIR, exist_ok=True)

# # Sort folders by timestamp
# sample_folders = sorted(glob(os.path.join(BASE_DIR, "sample_*")))
# assert len(sample_folders) >= FRAMES_IN + FRAMES_OUT, "Not enough samples!"

# def load_6_band_tensor(sample_path):
#     bands = []
#     for band in BAND_NAMES:
#         path = os.path.join(sample_path, "diffusion_pngs", f"{band}.png")
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Missing {path}")
#         img = Image.open(path).convert("L")
#         arr = np.array(img, dtype=np.float32) / 255.0  # Normalize
#         bands.append(arr)
#     stacked = np.stack(bands, axis=0)  # Shape: [6, H, W]
#     return stacked

# sequence_id = 0
# for i in tqdm(range(len(sample_folders) - FRAMES_IN - FRAMES_OUT + 1), desc="Building sequences"):
#     try:
#         input_frames = []
#         target_frames = []

#         # Capture names
#         input_names = sample_folders[i:i+FRAMES_IN]
#         target_names = sample_folders[i+FRAMES_IN:i+FRAMES_IN+FRAMES_OUT]

#         print(f"\n📦 Sequence {sequence_id:05d}")
#         print("🟦 Inputs :", [os.path.basename(x) for x in input_names])
#         print("🟥 Targets:", [os.path.basename(x) for x in target_names])

#         for path in input_names:
#             input_tensor = load_6_band_tensor(path)
#             input_frames.append(input_tensor)

#         for path in target_names:
#             target_tensor = load_6_band_tensor(path)
#             target_frames.append(target_tensor)

#         input_stack = torch.tensor(np.stack(input_frames, axis=0))   # [4, 6, H, W]
#         target_stack = torch.tensor(np.stack(target_frames, axis=0)) # [3, 6, H, W]

#         # Save with metadata
#         save_path = os.path.join(SAVE_DIR, f"sequence_{sequence_id:05d}.pt")
#         torch.save({
#             "input": input_stack,
#             "target": target_stack,
#             "input_names": [os.path.basename(x) for x in input_names],
#             "target_names": [os.path.basename(x) for x in target_names]
#         }, save_path)

#         sequence_id += 1

#     except Exception as e:
#         print(f"⚠️ Skipping sequence {i}: {e}")

# print(f"\n✅ Done! Saved {sequence_id} sequences in '{SAVE_DIR}'")
import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

BASE_DIR = r"D:\Hackathon\ISRO\final\final_test_data\processed_samples"
BAND_NAMES = ["IMG_VIS", "IMG_MIR", "IMG_SWIR", "IMG_TIR1", "IMG_TIR2", "IMG_WV"]
SAVE_DIR = r"./test_sorted_bands"

# Create destination directories
for band in BAND_NAMES:
    os.makedirs(os.path.join(SAVE_DIR, band), exist_ok=True)

# Get all sample folders
sample_folders = sorted(glob(os.path.join(BASE_DIR, "sample_*")))
print(f"🔍 Found {len(sample_folders)} sample folders.")

def save_bands_as_png(sample_path, sample_name):
    for band in BAND_NAMES:
        band_path = os.path.join(sample_path, "test_diffusion_pngs", f"{band}.png")
        if not os.path.exists(band_path):
            print(f"⚠️ Missing file: {band_path}")
            continue
        try:
            img = Image.open(band_path).convert("L")
            dest_path = os.path.join(SAVE_DIR, band, f"{sample_name}.png")
            img.save(dest_path)
        except Exception as e:
            print(f"❌ Error saving {band_path}: {e}")

# Process all samples
for sample_path in tqdm(sample_folders, desc="📤 Saving band images"):
    sample_name = os.path.basename(sample_path)
    save_bands_as_png(sample_path, sample_name)

print(f"\n✅ Done! Band images saved in '{SAVE_DIR}'")
