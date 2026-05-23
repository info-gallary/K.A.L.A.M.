import os
import math
import time
import random
import logging
import warnings
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

class SatelliteSequenceDataset(Dataset):
    def __init__(self, base_dir, channels, sequence_length=4, prediction_length=3, img_size=(256, 256), augment=True):
        self.base_dir = Path(base_dir)
        self.channels = channels
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.img_size = img_size
        self.augment = augment
        self.sequences = self._load_sequences()
        logger.info("Loaded %d valid sequences", len(self.sequences))

    def _get_all_timestamps(self):
        first_channel_dir = self.base_dir / self.channels[0]
        if not first_channel_dir.exists():
            raise FileNotFoundError(f"Missing channel folder: {first_channel_dir}")

        timestamps = set()
        for file_path in first_channel_dir.glob("*.png"):
            parts = file_path.stem.split("_")
            if len(parts) >= 3:
                timestamps.add(f"{parts[-2]}_{parts[-1]}")
        return sorted(timestamps)

    def _load_image(self, channel, timestamp):
        files = list((self.base_dir / channel).glob(f"*_{timestamp}.png"))
        if not files:
            return None
        img = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return None
        img = cv2.resize(img, self.img_size)
        img = cv2.GaussianBlur(img, (3, 3), 0.3)
        return img.astype(np.float32) / 255.0

    def _augment_consistent(self, frames):
        if not self.augment:
            return frames
        if random.random() < 0.5:
            frames = [np.flip(f, axis=-1).copy() for f in frames]
        if random.random() < 0.5:
            frames = [np.flip(f, axis=-2).copy() for f in frames]
        if random.random() < 0.35:
            angle = np.random.uniform(-4, 4)
            out = []
            for frame in frames:
                new_frame = []
                for ch in frame:
                    center = (ch.shape[1] / 2, ch.shape[0] / 2)
                    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                    new_frame.append(cv2.warpAffine(ch, rot, (ch.shape[1], ch.shape[0])))
                out.append(np.stack(new_frame, axis=0))
            frames = out
        return frames

    def _load_sequences(self):
        timestamps = self._get_all_timestamps()
        total_len = self.sequence_length + self.prediction_length
        sequences = []
        for i in range(0, len(timestamps) - total_len + 1):
            frames, ok = [], True
            for ts in timestamps[i : i + total_len]:
                channels = []
                for ch in self.channels:
                    img = self._load_image(ch, ts)
                    if img is None:
                        ok = False
                        break
                    channels.append(img)
                if not ok:
                    break
                frames.append(np.stack(channels, axis=0))
            if ok:
                frames = self._augment_consistent(frames)
                sequences.append({
                    "input": np.stack(frames[: self.sequence_length], axis=0),
                    "target": np.stack(frames[self.sequence_length :], axis=0),
                })
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = self.sequences[idx]
        return {
            "input": torch.tensor(sample["input"], dtype=torch.float32),
            "target": torch.tensor(sample["target"], dtype=torch.float32),
        }

class DiffusionSchedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return a * x0 + om * noise, noise

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        emb = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch) if time_dim else None
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
        h = F.silu(self.norm1(self.conv1(x)))
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class GridGCNBottleneck(nn.Module):
    def __init__(self, channels, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(layers)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid()) for _ in range(layers)])

    def neighbour_aggregate(self, x):
        up = torch.roll(x, shifts=1, dims=2)
        down = torch.roll(x, shifts=-1, dims=2)
        left = torch.roll(x, shifts=1, dims=3)
        right = torch.roll(x, shifts=-1, dims=3)
        return (up + down + left + right) / 4.0

    def forward(self, x):
        for conv, gate in zip(self.layers, self.gates):
            agg = self.neighbour_aggregate(x)
            g = gate(torch.cat([x, agg], dim=1))
            x = x + g * F.silu(conv(agg))
        return x


class MetaCorrectionLearner(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.err_estimator = nn.Sequential(
            nn.Conv2d(channels * 2, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, channels, 1), nn.Tanh()
        )
        self.confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid()
        )

    def forward(self, raw_pred, condition_summary):
        residual = self.err_estimator(torch.cat([raw_pred, condition_summary], dim=1))
        conf = self.confidence(raw_pred)
        return torch.clamp(raw_pred + 0.15 * conf * residual, -1, 1)


class ConditionalGCNPiNNDenoiser(nn.Module):
    def __init__(self, in_frames=4, out_frames=3, bands=6, base=64, time_dim=128):
        super().__init__()
        self.out_frames = out_frames
        self.bands = bands
        noisy_ch = out_frames * bands
        cond_ch = in_frames * bands
        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_ch, base, 3, padding=1), nn.SiLU(),
            ResBlock(base, base), ResBlock(base, base)
        )
        self.in_conv = nn.Conv2d(noisy_ch + base, base, 3, padding=1)

        self.e1 = ResBlock(base, base, time_dim)
        self.e2 = ResBlock(base, base * 2, time_dim)
        self.e3 = ResBlock(base * 2, base * 4, time_dim)

        self.gcn = GridGCNBottleneck(base * 4, layers=3)
        self.bot = ResBlock(base * 4, base * 4, time_dim)

        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ResBlock(base * 4, base * 2, time_dim)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ResBlock(base * 2, base, time_dim)

        self.raw_out = nn.Conv2d(base, noisy_ch, 3, padding=1)
        self.cond_summary = nn.Conv2d(base, noisy_ch, 1)
        self.meta = MetaCorrectionLearner(noisy_ch)

    def forward(self, x_t, condition, t):
        b, tf, c, h, w = x_t.shape
        x = x_t.view(b, tf * c, h, w)
        cond = condition.view(b, -1, h, w)
        cond_feat = self.cond_encoder(cond)
        t_emb = self.time_emb(t)

        x = self.in_conv(torch.cat([x, cond_feat], dim=1))
        e1 = self.e1(x, t_emb)
        e2 = self.e2(F.avg_pool2d(e1, 2), t_emb)
        e3 = self.e3(F.avg_pool2d(e2, 2), t_emb)

        z = self.gcn(e3)
        z = self.bot(z, t_emb)

        d2 = self.u2(z)
        d2 = self.d2(torch.cat([d2, e2], dim=1), t_emb)
        d1 = self.u1(d2)
        d1 = self.d1(torch.cat([d1, e1], dim=1), t_emb)

        raw_noise = self.raw_out(d1)
        corrected_noise = self.meta(raw_noise, self.cond_summary(cond_feat))
        return corrected_noise.view(b, tf, c, h, w)

def gradient_xy(x):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


def pinn_practicality_loss(pred_x0, condition):
    dx, dy = gradient_xy(pred_x0)
    spatial = dx.abs().mean() + dy.abs().mean()

    v1 = pred_x0[:, 1] - pred_x0[:, 0]
    v2 = pred_x0[:, 2] - pred_x0[:, 1]
    accel = F.smooth_l1_loss(v2, v1)

    continuity = F.smooth_l1_loss(pred_x0[:, 0], condition[:, -1])

    spectral = torch.mean(torch.abs(pred_x0[:, :, 1:] - pred_x0[:, :, :-1]))
    return 0.20 * spatial + 0.35 * accel + 0.35 * continuity + 0.10 * spectral


def ssl_teacher_consistency_loss(student_noise, teacher_model, x_t_aug, cond_aug, t):
    with torch.no_grad():
        teacher_noise = teacher_model(x_t_aug, cond_aug, t)
    return F.smooth_l1_loss(student_noise, teacher_noise)


def reconstruct_x0_from_noise(x_t, eps_pred, t, schedule):
    a = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
    om = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
    return torch.clamp((x_t - om * eps_pred) / (a + 1e-8), 0, 1)


def update_ema(teacher, student, decay=0.995):
    with torch.no_grad():
        for tp, sp in zip(teacher.parameters(), student.parameters()):
            tp.data.mul_(decay).add_(sp.data, alpha=1 - decay)

def calculate_metrics(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    vals_psnr, vals_ssim = [], []
    b, t, c, h, w = pred_np.shape
    for bi in range(b):
        for ti in range(t):
            for ci in range(c):
                p = np.clip(pred_np[bi, ti, ci], 0, 1)
                y = np.clip(target_np[bi, ti, ci], 0, 1)
                vals_psnr.append(min(psnr(y, p, data_range=1.0), 60.0))
                vals_ssim.append(ssim(y, p, data_range=1.0, win_size=min(7, h, w)))
    return float(np.mean(vals_psnr)), float(np.mean(vals_ssim))


@torch.no_grad()
def predict_x0_fast(model, condition, schedule, steps=25):
    b, _, c, h, w = condition.shape
    x = torch.randn(b, 3, c, h, w, device=condition.device)
    time_grid = torch.linspace(schedule.timesteps - 1, 0, steps, device=condition.device).long()
    for t_scalar in time_grid:
        t = torch.full((b,), int(t_scalar.item()), device=condition.device, dtype=torch.long)
        eps = model(x, condition, t)
        x0 = reconstruct_x0_from_noise(x, eps, t, schedule)
        x = 0.85 * x + 0.15 * x0
    return torch.clamp(x, 0, 1)


def train_novel_model(
    base_dir,
    epochs=80,
    batch_size=2,
    lr=1e-4,
    img_size=(256, 256),
    diffusion_steps=1000,
    save_path="novel_cd_gcn_pinn_ssl_meta_model.pth",
):
    channels = ["IMG_TIR1", "IMG_TIR2", "IMG_WV", "IMG_VIS", "IMG_MIR", "IMG_SWIR"]
    dataset = SatelliteSequenceDataset(base_dir, channels, img_size=img_size, augment=True)
    if len(dataset) < 2:
        raise ValueError("Need at least 2 valid sequences for train/validation split.")

    train_size = max(1, int(0.85 * len(dataset)))
    val_size = len(dataset) - train_size
    if val_size == 0:
        train_size -= 1
        val_size = 1

    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=device == "cuda")
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device == "cuda")

    schedule = DiffusionSchedule(timesteps=diffusion_steps)
    model = ConditionalGCNPiNNDenoiser().to(device)
    teacher = deepcopy(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler() if device == "cuda" else None

    best_score = -1e9
    logger.info("Model parameters: %.2f M", sum(p.numel() for p in model.parameters()) / 1e6)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = []
        for batch in train_loader:
            cond = batch["input"].to(device, non_blocking=True)
            x0 = batch["target"].to(device, non_blocking=True)
            b = x0.shape[0]
            t = torch.randint(0, diffusion_steps, (b,), device=device).long()
            x_t, noise = schedule.q_sample(x0, t)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with autocast():
                    eps_pred = model(x_t, cond, t)
                    pred_x0 = reconstruct_x0_from_noise(x_t, eps_pred, t, schedule)
                    diffusion_loss = F.mse_loss(eps_pred, noise)
                    pinn_loss = pinn_practicality_loss(pred_x0, cond)
                    ssl_loss = ssl_teacher_consistency_loss(eps_pred, teacher, x_t, cond, t)
                    loss = diffusion_loss + 0.15 * pinn_loss + 0.10 * ssl_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                eps_pred = model(x_t, cond, t)
                pred_x0 = reconstruct_x0_from_noise(x_t, eps_pred, t, schedule)
                diffusion_loss = F.mse_loss(eps_pred, noise)
                pinn_loss = pinn_practicality_loss(pred_x0, cond)
                ssl_loss = ssl_teacher_consistency_loss(eps_pred, teacher, x_t, cond, t)
                loss = diffusion_loss + 0.15 * pinn_loss + 0.10 * ssl_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            update_ema(teacher, model)
            loss_meter.append(float(loss.detach().cpu()))

        scheduler.step()

        val_psnr, val_ssim = 0.0, 0.0
        if epoch % 2 == 0 or epoch == epochs:
            model.eval()
            psnr_list, ssim_list = [], []
            for batch in val_loader:
                cond = batch["input"].to(device)
                target = batch["target"].to(device)
                pred = predict_x0_fast(model, cond, schedule, steps=25)
                p, s = calculate_metrics(pred, target)
                psnr_list.append(p)
                ssim_list.append(s)
            val_psnr = float(np.mean(psnr_list))
            val_ssim = float(np.mean(ssim_list))
            score = val_psnr + 50 * val_ssim
            if score > best_score:
                best_score = score
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "teacher_state_dict": teacher.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "architecture": "Conditional DDPM + GCN Bottleneck + PiNN Loss + SSL EMA Teacher + Meta-Correction Learner",
                    "channels": channels,
                    "img_size": img_size,
                }, save_path)
                logger.info("Saved best checkpoint: %s", save_path)

        logger.info(
            "Epoch %03d/%03d | loss %.6f | val PSNR %.2f | val SSIM %.4f | lr %.2e",
            epoch, epochs, np.mean(loss_meter), val_psnr, val_ssim, optimizer.param_groups[0]["lr"]
        )

    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Conditional diffusion cloud forecasting")
    parser.add_argument("--data", type=str, required=True, help="Dataset directory containing channel folders")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--save", type=str, default="novel_cd_gcn_pinn_ssl_meta_model.pth")
    args = parser.parse_args()

    train_novel_model(
        base_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        img_size=(args.size, args.size),
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
