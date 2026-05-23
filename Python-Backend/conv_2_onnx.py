import os
import torch
from torch.serialization import safe_globals
from train4 import EnhancedUNet  # Make sure EnhancedUNet matches your model definition
import numpy as np
def export_model_to_onnx(
    model_path="best_model.pth",
    onnx_output_path="inference_models/kalam_m1.onnx",
    input_shape=(1, 4, 6, 128, 128),
    device='cuda'
):

    os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)

    # Initialize model
    model = EnhancedUNet(in_channels=24, out_channels=18).to(device)

    # Load checkpoint
    with safe_globals([np._core.multiarray.scalar]):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},  # batch dynamic
            "output": {0: "batch_size"}
        },
        opset_version=17,
        verbose=True
    )

    print(f"✅ Model successfully exported to: {onnx_output_path}")

if __name__ == "__main__":
    export_model_to_onnx(
        model_path="best_model.pth",
        onnx_output_path="inference_models/kalam_m1.onnx",
        input_shape=(1, 4, 6, 128, 128),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )