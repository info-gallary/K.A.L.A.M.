import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import logging
import json
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SatelliteDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.data_files.sort()
        logger.info(f"Found {len(self.data_files)} data files")
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data = torch.load(self.data_files[idx], map_location='cpu')
        input_frames = data['input']  # [4, 6, H, W]
        target_frames = data['target']  # [3, 6, H, W]
        
        # Normalize to [0, 1] range for HF diffusion models
        input_frames = (input_frames - input_frames.min()) / (input_frames.max() - input_frames.min() + 1e-8)
        target_frames = (target_frames - target_frames.min()) / (target_frames.max() - target_frames.min() + 1e-8)
        
        # Convert to RGB format by selecting first 3 bands for visualization
        input_rgb = input_frames[:, :3]  # [4, 3, H, W] 
        target_rgb = target_frames[:, :3]  # [3, 3, H, W]
        
        return {
            'input': input_frames.float(),
            'target': target_frames.float(),
            'input_rgb': input_rgb.float(),
            'target_rgb': target_rgb.float()
        }

class MultiSpectralAdapter(nn.Module):
    """Adapter to handle multi-spectral satellite data with pretrained RGB models"""
    def __init__(self, input_channels=6, output_channels=3):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Channel adaptation layers
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, output_channels, 3, padding=1),
            nn.Sigmoid()  # Ensure output is in [0, 1] range
        )
        
        # Reverse adapter for reconstruction
        self.reverse_adapter = nn.Sequential(
            nn.Conv2d(output_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def to_rgb(self, multispectral):
        """Convert multi-spectral to RGB-like format"""
        return self.channel_adapter(multispectral)
    
    def to_multispectral(self, rgb):
        """Convert RGB back to multi-spectral format"""
        return self.reverse_adapter(rgb)

class TemporalConditioningModule(nn.Module):
    """Module to encode temporal sequence information"""
    def __init__(self, num_frames=4, embedding_dim=512):
        super().__init__()
        self.num_frames = num_frames
        self.embedding_dim = embedding_dim
        
        # Temporal positional embeddings
        self.temporal_embeddings = nn.Parameter(torch.randn(num_frames, embedding_dim))
        
        # Frame encoder
        self.frame_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Flatten(),
            nn.Linear(32*32*3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Sequence encoder
        self.sequence_encoder = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, 77)  # CLIP text embedding dimension
    
    def forward(self, frames):
        """
        frames: [B, T, 3, H, W] - RGB frames
        """
        batch_size, seq_len, channels, height, width = frames.shape
        
        # Encode each frame
        frame_features = []
        for t in range(seq_len):
            frame = frames[:, t]  # [B, 3, H, W]
            feat = self.frame_encoder(frame)  # [B, embedding_dim]
            feat = feat + self.temporal_embeddings[t].unsqueeze(0)  # Add temporal position
            frame_features.append(feat)
        
        # Stack and process sequence
        sequence = torch.stack(frame_features, dim=1)  # [B, T, embedding_dim]
        sequence_out, (hidden, _) = self.sequence_encoder(sequence)
        
        # Use final hidden state as conditioning
        conditioning = self.output_proj(hidden[-1])  # [B, 77]
        
        return conditioning.unsqueeze(1)  # [B, 1, 77] for CLIP compatibility

class SatelliteDiffusionPredictor:
    """Complete satellite frame prediction system using HuggingFace diffusion models"""
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.device = device
        self.model_id = model_id
        
        logger.info(f"Loading pretrained diffusion model: {model_id}")
        
        # Load components
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Move to device
        self.vae.to(device)
        self.text_encoder.to(device)
        self.unet.to(device)
        
        # Set to eval mode
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()
        
        # Initialize adapters
        self.spectral_adapter = MultiSpectralAdapter().to(device)
        self.temporal_conditioner = TemporalConditioningModule().to(device)
        
        logger.info("Model loaded successfully!")
    
    def encode_text_prompt(self, prompt):
        """Encode text prompt for conditioning"""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        return text_embeddings
    
    def preprocess_frames(self, frames):
        """Convert frames to format expected by diffusion model"""
        # frames: [B, T, 6, H, W] -> convert to RGB and resize
        batch_size, seq_len, channels, height, width = frames.shape
        
        # Convert multi-spectral to RGB using adapter
        rgb_frames = []
        for t in range(seq_len):
            rgb_frame = self.spectral_adapter.to_rgb(frames[:, t])  # [B, 3, H, W]
            rgb_frames.append(rgb_frame)
        
        rgb_frames = torch.stack(rgb_frames, dim=1)  # [B, T, 3, H, W]
        
        # Resize to 512x512 for Stable Diffusion
        resized_frames = []
        for b in range(batch_size):
            batch_frames = []
            for t in range(seq_len):
                frame = rgb_frames[b, t]  # [3, H, W]
                frame_pil = Image.fromarray((frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                frame_resized = frame_pil.resize((512, 512))
                frame_tensor = torch.from_numpy(np.array(frame_resized)).float() / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1)  # [3, 512, 512]
                batch_frames.append(frame_tensor)
            resized_frames.append(torch.stack(batch_frames))
        
        return torch.stack(resized_frames).to(self.device)  # [B, T, 3, 512, 512]
    
    def predict_next_frames(self, input_frames, num_predictions=3, num_inference_steps=50):
        """
        Predict next frames using the diffusion model
        input_frames: [B, 4, 6, H, W] - 4 input frames with 6 spectral bands each
        Returns: [B, 3, 6, H, W] - 3 predicted frames with 6 spectral bands each
        """
        batch_size = input_frames.shape[0]
        original_size = input_frames.shape[-2:]
        
        # Preprocess input frames
        rgb_input = self.preprocess_frames(input_frames)  # [B, 4, 3, 512, 512]
        
        # Generate temporal conditioning
        temporal_conditioning = self.temporal_conditioner(rgb_input)  # [B, 1, 77]
        
        # Create text prompt for satellite imagery
        text_prompt = "high resolution satellite imagery, temporal sequence, earth observation, clear weather"
        text_embeddings = self.encode_text_prompt([text_prompt] * batch_size)  # [B, 77, 768]
        
        # Combine text and temporal conditioning
        combined_conditioning = text_embeddings + temporal_conditioning.expand(-1, 77, -1)[:, :, :768]
        
        predictions = []
        
        for pred_idx in range(num_predictions):
            logger.info(f"Generating prediction {pred_idx + 1}/{num_predictions}")
            
            # Set up scheduler
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            
            # Start from noise
            latent_shape = (batch_size, 4, 64, 64)  # Stable Diffusion latent space
            latents = torch.randn(latent_shape, device=self.device)
            latents = latents * self.scheduler.init_noise_sigma
            
            # Denoising loop
            for i, t in enumerate(tqdm(timesteps, desc=f"Denoising frame {pred_idx + 1}")):
                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Prepare conditioning
                empty_embeddings = self.encode_text_prompt([""] * batch_size)
                encoder_hidden_states = torch.cat([empty_embeddings, combined_conditioning])
                
                # Predict noise
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )[0]
                
                # Classifier-free guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                guidance_scale = 7.5
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute previous latents
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # Decode latents to image
            with torch.no_grad():
                images = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            
            # Convert back to [0, 1] range
            images = (images + 1) / 2
            images = torch.clamp(images, 0, 1)
            
            # Resize back to original size and convert to multi-spectral
            resized_images = F.interpolate(images, size=original_size, mode='bilinear', align_corners=False)
            multispectral_pred = self.spectral_adapter.to_multispectral(resized_images)
            
            predictions.append(multispectral_pred)
        
        return torch.stack(predictions, dim=1)  # [B, 3, 6, H, W]

def calculate_metrics(pred, target):
    """Calculate PSNR and SSIM metrics"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    psnr_values = []
    ssim_values = []
    
    for i in range(pred_np.shape[0]):
        for j in range(pred_np.shape[1]):
            for k in range(pred_np.shape[2]):
                p_img = pred_np[i, j, k]
                t_img = target_np[i, j, k]
                
                # Ensure values are in [0, 1] range
                p_img = np.clip(p_img, 0, 1)
                t_img = np.clip(t_img, 0, 1)
                
                psnr_val = peak_signal_noise_ratio(t_img, p_img, data_range=1.0)
                ssim_val = structural_similarity(t_img, p_img, data_range=1.0)
                
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
    
    return np.mean(psnr_values), np.mean(ssim_values)

def validate_model(predictor, dataloader, device, num_samples=None):
    """Validate the model on test data"""
    predictor.vae.eval()
    predictor.text_encoder.eval()
    predictor.unet.eval()
    predictor.spectral_adapter.eval()
    predictor.temporal_conditioner.eval()
    
    total_psnr = 0
    total_ssim = 0
    total_samples = 0
    
    logger.info("Starting validation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            if num_samples and batch_idx >= num_samples:
                break
                
            input_frames = batch['input'].to(device)  # [B, 4, 6, H, W]
            target_frames = batch['target'].to(device)  # [B, 3, 6, H, W]
            
            # Generate predictions
            try:
                generated = predictor.predict_next_frames(input_frames, num_predictions=3)
                
                # Calculate metrics
                psnr, ssim = calculate_metrics(generated, target_frames)
                total_psnr += psnr
                total_ssim += ssim
                total_samples += 1
                
                logger.info(f"Batch {batch_idx + 1}: PSNR={psnr:.2f}, SSIM={ssim:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
    
    if total_samples > 0:
        avg_psnr = total_psnr / total_samples
        avg_ssim = total_ssim / total_samples
    else:
        avg_psnr = avg_ssim = 0
    
    return avg_psnr, avg_ssim

def visualize_predictions(predictor, dataloader, device, num_samples=3, save_path="predictions"):
    """Generate and visualize sample predictions"""
    predictor.vae.eval()
    predictor.text_encoder.eval()
    predictor.unet.eval()
    
    os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        input_frames = batch['input'][:num_samples].to(device)
        target_frames = batch['target'][:num_samples].to(device)
        
        logger.info("Generating sample predictions...")
        generated = predictor.predict_next_frames(input_frames, num_predictions=3)
        
        # Convert to numpy for visualization
        input_np = input_frames.cpu().numpy()
        target_np = target_frames.cpu().numpy()
        generated_np = generated.cpu().numpy()
        
        for sample_idx in range(num_samples):
            fig, axes = plt.subplots(3, 7, figsize=(21, 9))
            
            # Plot input frames (first 3 bands as RGB)
            for frame_idx in range(min(4, 7)):
                if frame_idx < 4:
                    rgb_input = input_np[sample_idx, frame_idx, :3].transpose(1, 2, 0)
                    rgb_input = np.clip(rgb_input, 0, 1)
                    axes[0, frame_idx].imshow(rgb_input)
                    axes[0, frame_idx].set_title(f'Input F{frame_idx+1}')
                    axes[0, frame_idx].axis('off')
                else:
                    axes[0, frame_idx].axis('off')
            
            # Plot target frames
            for frame_idx in range(min(3, 7)):
                rgb_target = target_np[sample_idx, frame_idx, :3].transpose(1, 2, 0)
                rgb_target = np.clip(rgb_target, 0, 1)
                axes[1, frame_idx].imshow(rgb_target)
                axes[1, frame_idx].set_title(f'Target F{frame_idx+1}')
                axes[1, frame_idx].axis('off')
            
            # Clear unused target subplots
            for i in range(3, 7):
                axes[1, i].axis('off')
            
            # Plot generated frames
            for frame_idx in range(min(3, 7)):
                rgb_generated = generated_np[sample_idx, frame_idx, :3].transpose(1, 2, 0)
                rgb_generated = np.clip(rgb_generated, 0, 1)
                axes[2, frame_idx].imshow(rgb_generated)
                axes[2, frame_idx].set_title(f'Generated F{frame_idx+1}')
                axes[2, frame_idx].axis('off')
            
            # Clear unused generated subplots
            for i in range(3, 7):
                axes[2, i].axis('off')
            
            plt.suptitle(f'Sample {sample_idx + 1} - HF Diffusion Satellite Prediction')
            plt.tight_layout()
            plt.savefig(f'{save_path}/prediction_sample_{sample_idx + 1}.png', dpi=300, bbox_inches='tight')
            plt.show()

def main():
    # Configuration
    data_dir = "./diffusion_dataset"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_id = "runwayml/stable-diffusion-v1-5"  # Can also use "stabilityai/stable-diffusion-2"
    
    logger.info(f"Using device: {device}")
    logger.info(f"Using model: {model_id}")
    
    # Dataset
    dataset = SatelliteDataset(data_dir)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders (smaller batch size for inference)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Initialize predictor with pretrained HuggingFace model
    try:
        predictor = SatelliteDiffusionPredictor(model_id=model_id, device=device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Falling back to CPU...")
        device = torch.device('cpu')
        predictor = SatelliteDiffusionPredictor(model_id=model_id, device=device)
    
    # Run validation
    logger.info("Starting validation with pretrained HuggingFace Diffusion model...")
    avg_psnr, avg_ssim = validate_model(predictor, val_loader, device, num_samples=5)
    
    logger.info(f"Validation Results:")
    logger.info(f"Average PSNR: {avg_psnr:.2f} dB")
    logger.info(f"Average SSIM: {avg_ssim:.3f}")
    
    # Check target achievement
    target_psnr = avg_psnr >= 30.0
    target_ssim = avg_ssim >= 0.6
    logger.info(f"Target PSNR ≥ 30 dB: {'✅' if target_psnr else '❌'}")
    logger.info(f"Target SSIM ≥ 0.6: {'✅' if target_ssim else '❌'}")
    logger.info(f"Overall targets met: {'✅' if target_psnr and target_ssim else '❌'}")
    
    # Generate sample predictions
    logger.info("Generating sample predictions...")
    visualize_predictions(predictor, val_loader, device, num_samples=3)
    
    # Save results
    results = {
        'model_id': model_id,
        'device': str(device),
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'target_psnr_met': target_psnr,
        'target_ssim_met': target_ssim,
        'overall_success': target_psnr and target_ssim,
        'validation_samples': len(val_dataset)
    }
    
    with open('hf_diffusion_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Validation completed! Results saved to hf_diffusion_results.json")
    
    return results

if __name__ == "__main__":
    results = main()