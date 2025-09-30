"""
Evaluation script for Simple DDPM on Fashion-MNIST
Computes FID, Inception Score, and shows sample grid.
"""

import os
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

import config
from src.model import create_model
from src.diffusion import create_diffusion, DDPMSampler
from src.utils import load_checkpoint, get_device, save_image_grid, unnormalize_to_zero_to_one
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------------------
# Config
# ---------------------------
DEVICE = get_device()
BATCH_SIZE = 64
NUM_SAMPLES = 1000  # Number of images to generate for evaluation

# ---------------------------
# Load test dataset
# ---------------------------
# IMPORTANT: Match the training transforms (without data augmentation)
transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Same as training: normalize to [-1, 1]
])

test_dataset = datasets.FashionMNIST(root=config.DATA_PATH, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------
# Load model and diffusion
# ---------------------------
checkpoint_path = os.path.join(config.CHECKPOINT_PATH, 'best.pth')
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

print("Creating model and diffusion process...")
model = create_model().to(DEVICE)
load_checkpoint(model, None, checkpoint_path, DEVICE)

diffusion = create_diffusion().to(DEVICE)
sampler = DDPMSampler(diffusion)

model.eval()

# ---------------------------
# Generate samples
# ---------------------------
@torch.no_grad()
def generate_samples(num_samples=NUM_SAMPLES):
    print(f"Generating {num_samples} samples...")
    samples = sampler.sample(
        model=model,
        num_samples=num_samples,
        image_size=config.IMAGE_SIZE,
        channels=config.CHANNELS,
        device=str(DEVICE)  
    )
    # Samples are in [-1, 1] range from the model
    # Map back to [0,1] for metrics and visualization
    samples = unnormalize_to_zero_to_one(samples)
    samples = torch.clamp(samples, 0., 1.)
    
    return samples

generated_images = generate_samples()

# ---------------------------
# Convert grayscale to RGB for metrics
# ---------------------------
def prepare_images_for_metrics(images):
    """
    Convert grayscale to 3-channel RGB and denormalize to [0,1] for Inception metrics.
    Input images are expected to be in [-1, 1] range.
    """
    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1.0) / 2.0
    images = torch.clamp(images, 0., 1.)
    
    # Convert grayscale to RGB
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    return images

# ---------------------------
# FID
# ---------------------------
print("Computing FID...")
fid_metric = FrechetInceptionDistance(normalize=True).to(DEVICE)

# Update with real images (denormalized)
print("Processing real images for FID...")
with torch.no_grad():
    for batch, _ in test_loader:
        batch = batch.to(DEVICE)
        batch_rgb = prepare_images_for_metrics(batch)
        fid_metric.update(batch_rgb, real=True)

# Update with generated images in batches
print("Processing generated images for FID...")
# Generated images are already in [0, 1], just need to convert to RGB
gen_rgb = generated_images
if gen_rgb.shape[1] == 1:
    gen_rgb = gen_rgb.repeat(1, 3, 1, 1)

with torch.no_grad():
    for i in range(0, len(gen_rgb), BATCH_SIZE):
        batch = gen_rgb[i:i+BATCH_SIZE].to(DEVICE)
        fid_metric.update(batch, real=False)

fid_score = fid_metric.compute()
print(f"FID: {fid_score.item():.4f}")

# ---------------------------
# Inception Score
# ---------------------------
print("Computing Inception Score...")
is_metric = InceptionScore(normalize=True).to(DEVICE)

# Update in batches
with torch.no_grad():
    for i in range(0, len(gen_rgb), BATCH_SIZE):
        batch = gen_rgb[i:i+BATCH_SIZE].to(DEVICE)
        is_metric.update(batch)

is_score, std = is_metric.compute()
print(f"Inception Score: {is_score.item():.4f} ± {std.item():.4f}")

# ---------------------------
# Visual inspection
# ---------------------------
print("Displaying sample grid...")
grid = make_grid(generated_images[:64], nrow=8, normalize=False, value_range=(0, 1))
plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray' if config.CHANNELS == 1 else None)
plt.axis('off')
plt.tight_layout()
plt.show()

# ---------------------------
# Save sample grid
# ---------------------------
output_dir = os.path.join(config.OUTPUT_PATH, 'eval')
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'generated_grid.png')
save_image_grid(generated_images[:64], save_path, nrow=8)
print(f"Sample grid saved to {save_path}")

# ---------------------------
# Print summary
# ---------------------------
print("\n" + "="*50)
print("EVALUATION SUMMARY")
print("="*50)
print(f"FID Score: {fid_score.item():.4f}")
print(f"Inception Score: {is_score.item():.4f} ± {std.item():.4f}")
print(f"Generated {NUM_SAMPLES} samples")
print(f"Results saved to: {output_dir}")
print("="*50)

