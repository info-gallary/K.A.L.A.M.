import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

# === Set this to your actual base path ===
BASE_DIR = r"D:\Hackathon\ISRO\final\final_test_data\processed_samples"

# Target .npy files to convert
target_npy_files = [
    "IMG_VIS.npy", "IMG_MIR.npy", "IMG_SWIR.npy",
    "IMG_TIR1.npy", "IMG_TIR2.npy", "IMG_WV.npy"
]

def crop_top_bottom(arr, top=70, bottom=50):
    return arr[top:arr.shape[0]-bottom, :]

def normalize_to_uint8(array):
    array = array.astype(np.float32)
    norm = (array - np.min(array)) / (np.max(array) - np.min(array) + 1e-8)
    return (norm * 255).astype(np.uint8)

# === Loop through all sample folders ===
sample_folders = sorted(glob(os.path.join(BASE_DIR, "sample_*")))

for folder in tqdm(sample_folders, desc="Converting all samples"):
    output_dir = os.path.join(folder, "test_diffusion_pngs")
    os.makedirs(output_dir, exist_ok=True)

    for file in target_npy_files:
        npy_path = os.path.join(folder, file)
        if not os.path.exists(npy_path):
            print(f"❌ Missing: {file} in {folder}")
            continue

        arr = np.load(npy_path)
        arr = np.squeeze(arr)

        if arr.ndim != 2:
            print(f"⚠️ Skipped {file} (bad shape: {arr.shape}) in {folder}")
            continue

        # Crop top and bottom
        arr = crop_top_bottom(arr, top=70, bottom=50)
        print(f"📐 Shape after crop for {file}: {arr.shape}")

        img_uint8 = normalize_to_uint8(arr)
        img_pil = Image.fromarray(img_uint8)
        save_path = os.path.join(output_dir, file.replace(".npy", ".png"))
        img_pil.save(save_path)

print("✅ All samples converted to PNG.")
