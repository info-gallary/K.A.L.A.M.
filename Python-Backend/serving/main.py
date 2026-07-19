from fastapi import FastAPI
from pydantic import BaseModel
import os
import torch
import time
import gc
from inference import run_model_inference
from train4 import SatelliteSequenceDataset

app = FastAPI()

# Weights are gitignored (~350 MB); train via train4.py or download the release asset
# into models/best_model.pth before starting this server.
MODEL_PATH = os.path.join("models", "best_model.pth")

# Global cache to store dataset after first load
dataset_cache = {}

class InferenceRequest(BaseModel):
    base_dir: str
    img_size: tuple = (128, 128)
    channels: list = ["IMG_TIR1", "IMG_TIR2", "IMG_WV", "IMG_VIS", "IMG_MIR", "IMG_SWIR"]
    sequence_id: int | None = None  # Optional specific sequence to process

@app.post("/run_inference")
def run_full_inference(request: InferenceRequest):
    base_dir = request.base_dir
    channels = request.channels
    img_size = tuple(request.img_size)
    sequence_id = request.sequence_id

    if not os.path.exists(base_dir):
        return {"error": f"Provided directory does not exist: {base_dir}"}

    device = 'cpu'

    # Load dataset once and cache it
    if base_dir not in dataset_cache:
        dataset = SatelliteSequenceDataset(base_dir=base_dir, channels=channels, img_size=img_size)
        dataset_cache[base_dir] = dataset
    else:
        dataset = dataset_cache[base_dir]

    total_sequences = len(dataset)
    output_root = os.path.join("inference_outputs", os.path.basename(base_dir.rstrip("/\\")))
    os.makedirs(output_root, exist_ok=True)

    sequence_range = range(total_sequences) if sequence_id is None else [sequence_id]

    for i in sequence_range:
        try:
            sample = dataset[i]
            input_tensor = sample['input'].unsqueeze(0).to(device)
            target_tensor = sample.get('target', None)
            if target_tensor is not None:
                target_tensor = target_tensor.unsqueeze(0).to(device)

            sequence_dir = os.path.join(output_root, f"sequence_{i:04d}")
            os.makedirs(sequence_dir, exist_ok=True)

            start_time = time.time()
            run_model_inference(
                model_path=MODEL_PATH,
                input_tensor=input_tensor,
                target_tensor=target_tensor,
                sequence_id=i,
                device=device,
                save_dir=sequence_dir
            )
            print(f"✅ Sequence {i} done in {time.time() - start_time:.2f} seconds.")

            del input_tensor
            del target_tensor
            gc.collect()

        except Exception as e:
            print(f"❌ Error in sequence {i}: {str(e)}")

    return {"message": f"Inference completed for {len(sequence_range)} sequence(s).", "output_dir": output_root}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
