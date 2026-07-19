import os
import gc
import time
import torch
from inference import run_model_inference
from train4 import SatelliteSequenceDataset

# Always run on CPU to avoid CUDA OOM
device = 'cpu'

# Load full dataset once
dataset = SatelliteSequenceDataset(
    base_dir="./test_sorted_bands",
    channels=["IMG_TIR1", "IMG_TIR2", "IMG_WV", "IMG_VIS", "IMG_MIR", "IMG_SWIR"],
    img_size=(128, 128)
)

output_root = "inference_outputs/test"
os.makedirs(output_root, exist_ok=True)


print(f"🔄 Processing {len(dataset)} sequences...")
# Batched over the dataset once (starting at index 299); the previous version
# wrapped this in `while(True)` with a `break` that only exited the inner
# loop, so it silently reprocessed the whole dataset forever.
for j in range(299, len(dataset), 15):
    for i in range(j, min(j + 15, len(dataset))):
        try:
            print(f"🔄 Processing sequence {i}...")
            sample = dataset[i]
            input_tensor = sample['input'].unsqueeze(0).to(device)  # Shape: [1, 4, 6, H, W]
            target_tensor = sample['target'].unsqueeze(0).to(device)  # Optional for validation metrics
            sequence_dir = os.path.join(output_root, f"sequence_{i:04d}")
            os.makedirs(sequence_dir, exist_ok=True)

            start_time = time.time()
            run_model_inference(
                model_path="models/best_model.pth",
                input_tensor=input_tensor,
                target_tensor=target_tensor,
                sequence_id=i,
                device=device,
                save_dir=sequence_dir
            )
            print(f"✅ Sequence {i} done in {time.time() - start_time:.2f} seconds.")

            del input_tensor, target_tensor, sample
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ Error in sequence {i}: {str(e)}")

