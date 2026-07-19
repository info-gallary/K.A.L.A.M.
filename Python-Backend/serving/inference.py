import os
import torch
import numpy as np
import json
import cv2
from torch.serialization import safe_globals
from torch.nn.functional import mse_loss
from skimage import exposure, filters
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from train4 import EnhancedUNet  # must match the architecture best_model.pth was saved from (train4, not train5)
import matplotlib.pyplot as plt



def sharpen_grayscale_band(band):
    band = np.clip(band, 0, 1)
    band = exposure.equalize_adapthist(band, clip_limit=0.03)
    blurred = filters.gaussian(band, sigma=0.8)
    sharp = band + 1.5 * (band - blurred)
    return np.clip(sharp, 0, 1)


import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_metrics(predicted, ground_truth):
    # Clip values
    predicted = np.clip(predicted, 0, 1)
    ground_truth = np.clip(ground_truth, 0, 1)

    # Compute metrics
    metrics = {
        "psnr": float(psnr(ground_truth, predicted, data_range=1.0)),
        "ssim": float(ssim(ground_truth, predicted, data_range=1.0, channel_axis=-1 if predicted.ndim==3 else None))
    }

    # Difference maps
    # diff = predicted - ground_truth
    # abs_diff = np.abs(diff)

    # # Plot
    # fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # # Raw difference
    # im0 = axes[0].imshow(diff, cmap='bwr', vmin=-1, vmax=1)
    # axes[0].set_title("Raw Difference (Pred - GT)")
    # plt.colorbar(im0, ax=axes[0])

    # # # Absolute difference
    # # im1 = axes[1].imshow(abs_diff, cmap='hot')
    # # axes[1].set_title("Absolute Difference")
    # # plt.colorbar(im1, ax=axes[1])

    # # Histogram
    # axes[1].hist(diff.ravel(), bins=50, color='purple', alpha=0.7)
    # axes[1].set_title("Histogram of Differences")
    # axes[1].set_xlabel("Difference Value")
    # axes[1].set_ylabel("Frequency")

    # plt.suptitle(f"PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f}")
    # plt.tight_layout()
    # plt.savefig("difference_metrics_.png")
    # plt.show()
    # plt.close(fig)

    return metrics

channels=["IMG_TIR1", "IMG_TIR2", "IMG_WV", "IMG_VIS", "IMG_MIR", "IMG_SWIR"]
def run_model_inference(model_path, input_tensor, target_tensor=None, sequence_id=None, device='cpu', save_dir='inference_outputs/new_model'):
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model = EnhancedUNet(in_channels=24, out_channels=18).to(device)
    with safe_globals([np._core.multiarray.scalar]):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        pred_tensor = model(input_tensor)

    pred_np = pred_tensor.cpu().numpy()[0]  # [3, 6, H, W]
    performance = {
        "sequence_id": sequence_id,
        "frames": [],
        "avg_mse_loss": None
    }

    total_loss = 0.0
    count = 0

    for t in range(3):  # Predicted T5, T6, T7
        for c in range(6):  # Bands 1 to 6
            band = pred_np[t, c]
            band = np.clip(band, 0, 1)
            enhanced = (band * 255).astype(np.uint8)

            file_name = f"predicted_t{t+5}_{channels[c]}.png"
            file_path = os.path.join(save_dir, file_name)
            cv2.imwrite(file_path, enhanced)

            # Metrics
            if target_tensor is not None:
                target_np = target_tensor.cpu().numpy()[0][t, c]
                metrics = compute_metrics(band, target_np)

                # MSE loss
                loss = mse_loss(torch.tensor(band), torch.tensor(target_np)).item()
                total_loss += loss
                count += 1

                performance["frames"].append({
                    "frame": f"T{t+5}_{channels[c]}",
                    "file": file_name,
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "mse_loss": round(loss, 6)
                })

    if count > 0:
        performance["avg_mse_loss"] = round(total_loss / count, 6)

    if target_tensor is not None:
        with open(os.path.join(save_dir, "performance.json"), 'w') as f:
            json.dump(performance, f, indent=4)

    return pred_tensor
