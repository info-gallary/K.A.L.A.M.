import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import glob
import logging
from pathlib import Path
from torch.serialization import safe_globals
import numpy as np
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

class SatelliteSequenceDataset(Dataset):
    """Robust dataset for satellite image sequences"""
    
    def __init__(self, base_dir, channels, sequence_length=4, prediction_length=3, img_size=(128, 128)):
        self.base_dir = Path(base_dir)
        self.channels = channels
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.img_size = img_size
        
        logger.info(f"Loading data from: {self.base_dir}")
        logger.info(f"Channels: {channels}")
        
        self.sequences = self._load_sequences()
        logger.info(f"✅ Total valid sequences: {len(self.sequences)}")
        
    def _get_all_timestamps(self):
        """Extract all unique timestamps from filenames"""
        timestamps = set()
        first_channel_dir = self.base_dir / self.channels[0]
        
        if not first_channel_dir.exists():
            raise FileNotFoundError(f"Channel directory not found: {first_channel_dir}")
            
        for file_path in first_channel_dir.glob("*.png"):
            # Extract timestamp from filename (assuming format: prefix_YYYYMMDD_HHMMSS.png)
            filename = file_path.stem
            parts = filename.split('_')
            if len(parts) >= 3:
                timestamp = f"{parts[-2]}_{parts[-1]}"  # YYYYMMDD_HHMMSS
                timestamps.add(timestamp)
        
        return sorted(list(timestamps))
    
    def _load_image_safely(self, channel, timestamp):
        """Safely load image with error handling"""
        channel_dir = self.base_dir / channel
        
        # Find file with this timestamp
        pattern = f"*_{timestamp}.png"
        matching_files = list(channel_dir.glob(pattern))
        
        if not matching_files:
            return None
            
        file_path = matching_files[0]
        
        try:
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                logger.warning(f"Failed to load image: {file_path}")
                return None
                
            # Resize and normalize
            img = cv2.resize(img, self.img_size)
            img = img.astype(np.float32) / 255.0
            return img
            
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            return None
    
    def _load_sequences(self):
        """Load all valid sequences"""
        timestamps = self._get_all_timestamps()
        logger.info(f"Found {len(timestamps)} timestamps")
        
        sequences = []
        total_length = self.sequence_length + self.prediction_length
        
        for i in range(len(timestamps) - total_length + 1):
            # Get timestamps for this sequence
            seq_timestamps = timestamps[i:i + total_length]
            
            # Load all frames for this sequence
            sequence_frames = []
            valid_sequence = True
            
            for t_idx, timestamp in enumerate(seq_timestamps):
                frame_channels = []
                
                # Load all channels for this timestamp
                for channel in self.channels:
                    img = self._load_image_safely(channel, timestamp)
                    if img is None:
                        valid_sequence = False
                        break
                    frame_channels.append(img)
                
                if not valid_sequence:
                    break
                    
                # Stack channels for this frame
                frame = np.stack(frame_channels, axis=0)  # [C, H, W]
                sequence_frames.append(frame)
            
            if valid_sequence and len(sequence_frames) == total_length:
                input_frames = sequence_frames[:self.sequence_length]   # First 4 frames
                target_frames = sequence_frames[self.sequence_length:]  # Next 3 frames
                
                sequences.append({
                    'input': np.stack(input_frames, axis=0),    # [4, 6, H, W]
                    'target': np.stack(target_frames, axis=0)   # [3, 6, H, W]
                })
            print(f"Loaded sequence {len(sequences)}: {seq_timestamps[0]} to {seq_timestamps[-1]}")
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sample = self.sequences[idx]
        return {
            'input': torch.tensor(sample['input'], dtype=torch.float32),
            'target': torch.tensor(sample['target'], dtype=torch.float32)
        }

class EnhancedUNet(nn.Module):
    """Enhanced U-Net for satellite frame prediction"""
    
    def __init__(self, in_channels=24, out_channels=18):  # 4*6 input, 3*6 output
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.up4 = self._up_conv(1024, 512)
        self.dec4 = self._conv_block(1024, 512)
        
        self.up3 = self._up_conv(512, 256)
        self.dec3 = self._conv_block(512, 256)
        
        self.up2 = self._up_conv(256, 128)
        self.dec2 = self._conv_block(256, 128)
        
        self.up1 = self._up_conv(128, 64)
        self.dec1 = self._conv_block(128, 64)
        
        # Output
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def _up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
    
    def forward(self, x):
        # Input: [B, 4, 6, H, W] -> reshape to [B, 24, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        out = self.final(d1)  # [B, 18, H, W]
        
        # Reshape to [B, 3, 6, H, W]
        out = out.view(B, 3, 6, H, W)
        
        return out

def combined_loss(pred, target, alpha=0.8):
    """Combined L1 + MSE loss (removed SSIM to fix gradient issue)"""
    l1_loss = F.l1_loss(pred, target)
    mse_loss = F.mse_loss(pred, target)
    
    return alpha * l1_loss + (1 - alpha) * mse_loss

def calculate_metrics(pred, target):
    """Calculate PSNR and SSIM metrics"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    psnr_values = []
    ssim_values = []
    
    B, T, C, H, W = pred_np.shape
    
    for b in range(B):
        for t in range(T):
            for c in range(C):
                p_img = pred_np[b, t, c]
                t_img = target_np[b, t, c]
                
                # Ensure values are in [0, 1]
                p_img = np.clip(p_img, 0, 1)
                t_img = np.clip(t_img, 0, 1)
                
                psnr_val = psnr(t_img, p_img, data_range=1.0)
                ssim_val = ssim(t_img, p_img, data_range=1.0)
                
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
    
    return np.mean(psnr_values), np.mean(ssim_values)

def train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-4):
    """Train the model"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_psnr = 0
    history = {'train_loss': [], 'val_psnr': [], 'val_ssim': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_frames = batch['input'].to(device)   # [B, 4, 6, H, W]
            target_frames = batch['target'].to(device) # [B, 3, 6, H, W]
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_frames = model(input_frames)
            
            # Calculate loss
            loss = combined_loss(pred_frames, target_frames)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        if epoch % 2 == 0:  # Validate every 2 epochs
            model.eval()
            val_psnr_total = 0
            val_ssim_total = 0
            val_samples = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_frames = batch['input'].to(device)
                    target_frames = batch['target'].to(device)
                    
                    pred_frames = model(input_frames)
                    
                    batch_psnr, batch_ssim = calculate_metrics(pred_frames, target_frames)
                    val_psnr_total += batch_psnr
                    val_ssim_total += batch_ssim
                    val_samples += 1
            
            avg_psnr = val_psnr_total / val_samples if val_samples > 0 else 0
            avg_ssim = val_ssim_total / val_samples if val_samples > 0 else 0
            
            history['val_psnr'].append(avg_psnr)
            history['val_ssim'].append(avg_ssim)
            
            logger.info(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val PSNR={avg_psnr:.2f}, Val SSIM={avg_ssim:.3f}')
            
            # Save best model
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_psnr': best_psnr,
                    'history': history
                }, 'best_model.pth')
                logger.info(f'✅ New best model saved! PSNR: {best_psnr:.2f}')
        
        scheduler.step()
    
    return history

def visualize_predictions(model, dataloader, num_samples=3):
    """Visualize model predictions"""
    model.eval()
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        input_frames = batch['input'][:num_samples].to(device)
        target_frames = batch['target'][:num_samples].to(device)
        
        pred_frames = model(input_frames)
        
        # Convert to numpy
        input_np = input_frames.cpu().numpy()
        target_np = target_frames.cpu().numpy()
        pred_np = pred_frames.cpu().numpy()
        
        for sample_idx in range(num_samples):
            fig, axes = plt.subplots(3, 6, figsize=(18, 9))
            
            # Show input frames (first 3 bands as RGB)
            for t in range(min(4, 6)):
                if t < 4:
                    # Use first 3 channels as RGB
                    rgb = input_np[sample_idx, t, :3].transpose(1, 2, 0)
                    rgb = np.clip(rgb, 0, 1)
                    axes[0, t].imshow(rgb)
                    axes[0, t].set_title(f'Input T{t+1}')
                    axes[0, t].axis('off')
                else:
                    axes[0, t].axis('off')
            
            # Clear remaining input plots
            for t in range(4, 6):
                axes[0, t].axis('off')
            
            # Show target frames
            for t in range(3):
                rgb = target_np[sample_idx, t, :3].transpose(1, 2, 0)
                rgb = np.clip(rgb, 0, 1)
                axes[1, t].imshow(rgb)
                axes[1, t].set_title(f'Target T{t+5}')
                axes[1, t].axis('off')
            
            # Clear remaining target plots
            for t in range(3, 6):
                axes[1, t].axis('off')
            
            # Show predicted frames
            for t in range(3):
                rgb = pred_np[sample_idx, t, :3].transpose(1, 2, 0)
                rgb = np.clip(rgb, 0, 1)
                axes[2, t].imshow(rgb)
                axes[2, t].set_title(f'Predicted T{t+5}')
                axes[2, t].axis('off')
            
            # Clear remaining prediction plots
            for t in range(3, 6):
                axes[2, t].axis('off')
            
            plt.suptitle(f'Sample {sample_idx + 1} - Satellite Frame Prediction')
            plt.tight_layout()
            plt.savefig(f'prediction_sample_{sample_idx + 1}.png', dpi=200, bbox_inches='tight')
            plt.show()

def main():
    """Main training and evaluation function"""
    
    # Configuration
    BASE_DIR = "./test_sorted_bands"  # generated by npy2png.py + stack.py; not tracked in git
    CHANNELS = ["IMG_TIR1", "IMG_TIR2", "IMG_WV", "IMG_VIS", "IMG_MIR", "IMG_SWIR"]
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 4
    NUM_EPOCHS = 200
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    try:
        # Load dataset
        logger.info("Loading dataset...")
        dataset = SatelliteSequenceDataset(
            base_dir=BASE_DIR,
            channels=CHANNELS,
            img_size=IMG_SIZE
        )
        
        if len(dataset) == 0:
            raise ValueError("No valid sequences found! Check your data directory and file structure.")
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        # Initialize model
        model = EnhancedUNet(in_channels=24, out_channels=18).to(device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        logger.info("Starting training...")
        history = train_model(model, train_loader, val_loader, NUM_EPOCHS)
        
        # Load best model for final evaluation
        # import torch


        with safe_globals([np._core.multiarray.scalar]):
            checkpoint = torch.load("best_model.pth", weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])

        
        # Final evaluation
        logger.info("Final evaluation...")
        model.eval()
        final_psnr = 0
        final_ssim = 0
        num_val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_frames = batch['input'].to(device)
                target_frames = batch['target'].to(device)
                
                pred_frames = model(input_frames)
                batch_psnr, batch_ssim = calculate_metrics(pred_frames, target_frames)
                
                final_psnr += batch_psnr
                final_ssim += batch_ssim
                num_val_samples += 1
        
        final_psnr /= num_val_samples
        final_ssim /= num_val_samples
        
        logger.info(f"🎯 Final Results:")
        logger.info(f"   PSNR: {final_psnr:.2f} dB (Target: ≥30 dB) {'✅' if final_psnr >= 30 else '❌'}")
        logger.info(f"   SSIM: {final_ssim:.3f} (Target: ≥0.6) {'✅' if final_ssim >= 0.6 else '❌'}")
        
        # Save results
        results = {
            'final_psnr': final_psnr,
            'final_ssim': final_ssim,
            # 'target_psnr_met': final_psnr >= 30,
            # 'target_ssim_met': final_ssim >= 0.6,
            # 'training_history': history,
            # 'model_params': sum(p.numel() for p in model.parameters())
        }
        
        with open('outputs/results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualizations
        logger.info("Generating sample predictions...")
        visualize_predictions(model, val_loader, num_samples=3)
        
        # Plot training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(history['val_psnr'])
        plt.axhline(y=30, color='r', linestyle='--', alpha=0.7, label='Target PSNR=30')
        plt.title('Validation PSNR')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(history['val_ssim'])
        plt.axhline(y=0.6, color='r', linestyle='--', alpha=0.7, label='Target SSIM=0.6')
        plt.title('Validation SSIM')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('outputs/training_curves.png', dpi=200, bbox_inches='tight')
        plt.show()
        
        logger.info("✅ Training completed successfully!")
        logger.info("📁 Check 'outputs/' folder for saved model and results")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")

        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🚀 Model training completed successfully!")
    else:
        print("💥 Training failed. Check the logs above.")