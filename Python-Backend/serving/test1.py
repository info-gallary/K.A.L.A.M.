import os
import gc
import time
import torch
from inference import run_model_inference
from train4 import SatelliteSequenceDataset

# Device setup — keep CPU to avoid CUDA OOM
device = 'cpu'

# === Load dataset (only once) ===
dataset = SatelliteSequenceDataset(
    base_dir="./test_sorted_bands",
    channels=["IMG_TIR1", "IMG_TIR2", "IMG_WV", "IMG_VIS", "IMG_MIR", "IMG_SWIR"],
    img_size=(128, 128)
)

# === Choose a single sequence index ===
sequence_index = 0  # 🔹 Change this number for another sequence

# === Output directory setup ===
output_root = "inference_outputs/single_sequence"
sequence_dir = os.path.join(output_root, f"sequence_{sequence_index:04d}")
os.makedirs(sequence_dir, exist_ok=True)

print(f"🔄 Running inference on sequence {sequence_index}...")

sample = input_tensor = target_tensor = None
try:
    # Load the sequence sample
    sample = dataset[sequence_index]
    input_tensor = sample['input'].unsqueeze(0).to(device)   # [1, 4, 6, H, W]
    target_tensor = sample['target'].unsqueeze(0).to(device) # Optional

    # === Run inference ===
    start_time = time.time()
    run_model_inference(
        model_path="models/best_model.pth",
        input_tensor=input_tensor,
        target_tensor=target_tensor,
        sequence_id=sequence_index,
        device=device,
        save_dir=sequence_dir
    )
    print(f"✅ Inference complete for sequence {sequence_index} in {time.time() - start_time:.2f}s")

except Exception as e:
    print(f"❌ Error during inference: {str(e)}")

finally:
    # Cleanup (guarded: nothing to free if the sample never loaded)
    del sample, input_tensor, target_tensor
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 Memory cleared.")
