import torch
import matplotlib.pyplot as plt
import os

# --- Change this path to your own .pt file
pt_path = r"./diffusion_dataset/sequence_00000.pt"

# Band details
band_info = [
    ('IMG_VIS', 'Visible Light', 'grey'),
    ('IMG_MIR', 'Middle Infrared', 'grey'),
    ('IMG_SWIR', 'Short Wave Infrared', 'grey'),
    ('IMG_TIR1', 'Thermal Infrared 1', 'grey'),
    ('IMG_TIR2', 'Thermal Infrared 2', 'grey'),
    ('IMG_WV', 'Water Vapor', 'grey')
]
# band_info = [
#     ('IMG_VIS', 'Visible Light', 'viridis'),
#     ('IMG_MIR', 'Middle Infrared', 'plasma'),
#     ('IMG_SWIR', 'Short Wave Infrared', 'inferno'),
#     ('IMG_TIR1', 'Thermal Infrared 1', 'hot'),
#     ('IMG_TIR2', 'Thermal Infrared 2', 'hot'),
#     ('IMG_WV', 'Water Vapor', 'Blues')
# ]

# Load the .pt file
data = torch.load(pt_path)
input_tensor = data["input"]    # [4, 6, H, W]
target_tensor = data["target"]  # [3, 6, H, W]

print(f"\n📦 Input Tensor Shape: {input_tensor.shape} | Size: {input_tensor.element_size() * input_tensor.nelement() / (1024**2):.2f} MB")
print(f"🎯 Target Tensor Shape: {target_tensor.shape} | Size: {target_tensor.element_size() * target_tensor.nelement() / (1024**2):.2f} MB")

def plot_multiband_frames(tensor, title, num_frames):
    fig, axes = plt.subplots(num_frames, 6, figsize=(18, 3 * num_frames))
    fig.suptitle(title, fontsize=16)

    for i in range(num_frames):
        for j in range(6):
            ax = axes[i, j] if num_frames > 1 else axes[j]
            image = tensor[i, j].cpu().numpy()
            ax.imshow(image, cmap=band_info[j][2])
            ax.set_title(f"{band_info[j][0]}\n{image.shape}", fontsize=8)
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# Plot Input Frames
plot_multiband_frames(input_tensor, "Input Frames (Past 4)", num_frames=4)

# Plot Target Frames
plot_multiband_frames(target_tensor, "Target Frames (Future 3)", num_frames=3)
