import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from diffusers import UNet2DConditionModel
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# --- Dataset Loader ---
class CloudChaseDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        inputs = []
        targets = []
        for i in range(4):
            inputs.append(np.load(os.path.join(path, f"input_{i}.npy")))
        for i in range(3):
            targets.append(np.load(os.path.join(path, f"target_{i}.npy")))

        input_tensor = torch.from_numpy(np.concatenate(inputs, axis=0)).float()  # (24, H, W)
        target_tensor = torch.from_numpy(np.concatenate(targets, axis=0)).float()  # (18, H, W)
        return input_tensor / 255.0, target_tensor / 255.0

# --- Conditioning Encoder ---
class ConditioningEncoder(nn.Module):
    def __init__(self, in_channels=24, embed_dim=768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# --- Latent Encoder (Replaces VAE) ---
class SimpleLatentEncoder(nn.Module):
    def __init__(self, in_channels=18, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, 3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.encoder(x)

# --- Main Training ---
def train():
    root_dir = "./cloudchase_dataset"
    batch_size = 4
    epochs = 50
    lr = 1e-4
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    latent_encoder = SimpleLatentEncoder().cuda()
    unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet"
).cuda()

    cond_encoder = ConditioningEncoder().cuda()

    optimizer = torch.optim.Adam(
        list(unet.parameters()) + 
        list(cond_encoder.parameters()) + 
        list(latent_encoder.parameters()), 
        lr=lr
    )

    train_loader = DataLoader(CloudChaseDataset(root_dir), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()

            with torch.no_grad():
                y_latent = latent_encoder(y)  # [B, 4, H/8, W/8] assumed

            cond_emb = cond_encoder(x).unsqueeze(1)  # shape: [B, 1, 1280]
  # [B, 1280]
            noise = torch.randn_like(y_latent)
            timesteps = torch.randint(0, 1000, (x.size(0),), device=x.device).long()

            noisy_latents = noise

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=cond_emb).sample
            loss = nn.MSELoss()(model_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"[Epoch {epoch} | Step {i}] Loss: {loss.item():.4f}")

            if i % 10 == 0:
                with torch.no_grad():
                    pred_latents = unet(noise, timesteps, encoder_hidden_states=cond_emb).sample
                    pred_image = pred_latents[0].cpu().numpy()
                    np.save(f"{save_dir}/pred_latent_{epoch}_{i}.npy", pred_image)

                    # Save first 3 channels as visualization
                    img = np.transpose(pred_image[:3], (1, 2, 0))
                    plt.imsave(f"{save_dir}/pred_latent_{epoch}_{i}.png", img)

if __name__ == "__main__":
    train()
