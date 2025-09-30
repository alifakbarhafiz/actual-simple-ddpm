"""
Configuration file for Simple DDPM project
"""
import torch
import os

# Training parameters
BATCH_SIZE = 64  # Reduced for memory efficiency
LEARNING_RATE = 2e-4
EPOCHS = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 2

# Model parameters
IMAGE_SIZE = 32
CHANNELS = 1
TIME_STEPS = 500

# U-Net architecture parameters
UNET_DIM = 32
UNET_DIM_MULTS = [1, 2, 4, 8]
UNET_RESNET_BLOCK_GROUPS = 8

# Diffusion parameters
BETA_START = 0.0001
BETA_END = 0.02
SCHEDULE_TYPE = 'cosine'  # 'linear' or 'cosine'

# Paths
DATA_PATH = '/content/data/fashion-mnist'
CHECKPOINT_PATH = '/content/drive/MyDrive/ddpm_checkpoints_4'  # Save to Drive
OUTPUT_PATH = '/content/drive/MyDrive/ddpm_outputs'
LOG_PATH = '/content/logs'

# Training settings
SAVE_EVERY = 5  # Save more frequently
SAMPLE_EVERY = 10
LOG_EVERY = 50
GRAD_CLIP = 1.0

# Sampling parameters
NUM_SAMPLES = 64  # Number of samples to generate
SAMPLE_GRID_SIZE = 8  # 8x8 grid

# Create directories if they don't exist
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)