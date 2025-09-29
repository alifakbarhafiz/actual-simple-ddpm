# Simple DDPM Implementation

A clean, minimal implementation of Denoising Diffusion Probabilistic Models (DDPM) trained on Tiny ImageNet dataset.

## Features

- **Simple & Clean**: Focused on core DDPM concepts without unnecessary complexity
- **Robust Implementation**: Proper error handling, checkpointing, and logging
- **Tiny ImageNet Support**: Automatically downloads and processes the dataset
- **GPU Accelerated**: CUDA support with automatic device detection
- **Tensorboard Logging**: Track training progress and visualize samples
- **EMA Support**: Exponential Moving Average for better sample quality
- **Flexible Sampling**: Generate samples, interpolations, and grids

## Project Structure

```
simple-ddpm/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                   # Configuration settings
├── train.py                    # Training script
├── sample.py                   # Sampling script
├── src/
│   ├── dataset.py              # Data loading utilities
│   ├── model.py                # U-Net architecture
│   ├── diffusion.py            # DDPM implementation
│   └── utils.py                # Helper functions
├── data/                       # Dataset directory (auto-created)
├── checkpoints/                # Model checkpoints (auto-created)
├── outputs/                    # Generated samples (auto-created)
└── logs/                       # Training logs (auto-created)
```

## Quick Start

### 1. Setup Environment

```bash
# Clone or download the project
git clone <repository_url>
cd simple-ddpm

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Start training (will automatically download Tiny ImageNet)
python train.py
```

The training script will:
- Download Tiny ImageNet dataset (~237MB) if not present
- Create necessary directories
- Start training with progress bars and logging
- Save checkpoints every 10 epochs
- Generate sample images every 5 epochs
- Log progress to Tensorboard

### 3. Generate Samples

```bash
# Generate samples from best checkpoint
python sample.py --checkpoint checkpoints/best.pth --num_samples 64

# Generate interpolation between samples
python sample.py --checkpoint checkpoints/best.pth --interpolate --interp_steps 10
```

### 4. Monitor Training

```bash
# View tensorboard logs
tensorboard --logdir logs/tensorboard
```

## Configuration

All settings are in `config.py`. Key parameters:

```python
# Training
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 100

# Model
IMAGE_SIZE = 64
TIME_STEPS = 1000
UNET_DIM = 128

# Paths
DATA_PATH = './data/tiny-imagenet-200'
CHECKPOINT_PATH = './checkpoints'
OUTPUT_PATH = './outputs'
```

## Model Architecture

- **U-Net Backbone**: ResNet blocks with attention mechanisms
- **Time Embedding**: Sinusoidal positional encodings
- **Skip Connections**: Multi-scale feature fusion
- **GroupNorm + SiLU**: Modern normalization and activation
- **~90M Parameters**: Manageable size for most GPUs

## Training Details

- **Dataset**: Tiny ImageNet (200 classes, 64×64 RGB images)
- **Diffusion Steps**: 1000 timesteps with linear noise schedule
- **Loss Function**: Simple MSE between predicted and actual noise
- **Optimization**: Adam with gradient clipping
- **EMA**: Exponential moving average for stable sampling
- **Memory**: ~8GB VRAM required for default settings

## File Descriptions

### Core Files

- **`train.py`**: Main training loop with checkpointing, logging, and sample generation
- **`sample.py`**: Generate samples from trained models with various options
- **`config.py`**: Central configuration with all hyperparameters

### Source Code (`src/`)

- **`dataset.py`**: Tiny ImageNet dataset loading with automatic download
- **`model.py`**: U-Net implementation optimized for 64×64 images  
- **`diffusion.py`**: DDPM forward/reverse processes with sampling methods
- **`utils.py`**: Helper functions for training, logging, and visualization

## Usage Examples

### Basic Training
```bash
python train.py
```

### Resume from Checkpoint
```bash
# Training automatically resumes from latest.pth if it exists
python train.py
```

### Custom Sampling
```bash
# Generate 100 samples
python sample.py --num_samples 100

# Use specific checkpoint
python sample.py --checkpoint checkpoints/epoch_050.pth

# Save to custom directory
python sample.py --output_dir my_samples/

# Generate interpolation
python sample.py --interpolate --interp_steps 8
```

### Test Components
```bash
# Test dataset loading
python -c "from src.dataset import test_dataset; test_dataset()"

# Test model architecture
python -c "from src.model import test_model; test_model()"

# Test diffusion process
python -c "from src.diffusion import test_diffusion; test_diffusion()"
```

## Expected Results

After training for 100 epochs (~6-12 hours on modern GPU):

- **Training Loss**: Should decrease from ~0.1 to ~0.01
- **Sample Quality**: Recognizable objects and textures
- **Diversity**: Wide variety of generated content
- **Stability**: Consistent generation without mode collapse

Sample progression:
- **Epoch 10**: Basic shapes and colors
- **Epoch 30**: Simple textures and patterns  
- **Epoch 50**: Recognizable objects
- **Epoch 100**: High-quality diverse images

## Troubleshooting

### Memory Issues
```python
# In config.py, reduce:
BATCH_SIZE = 8  # or 4
NUM_SAMPLES = 16  # instead of 64
```

### Slow Training
```python
# In config.py:
NUM_WORKERS = 0  # if on Windows
TIME_STEPS = 500  # reduce timesteps
```

### Dataset Download Issues
```bash
# Manual download:
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d data/
```

### CUDA Issues
```python
# In config.py:
DEVICE = 'cpu'  # force CPU if CUDA problems
```

## Advanced Usage

### Custom Dataset
Modify `src/dataset.py` to load your own dataset:

```python
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # Your dataset loading code here
        pass
```

### Different Image Sizes
```python
# In config.py:
IMAGE_SIZE = 32  # or 128, 256 (requires model changes)
```

### Conditional Generation
The current implementation is unconditional. For class-conditional generation, you would need to:

1. Modify the U-Net to accept class embeddings
2. Update the diffusion process to handle conditioning
3. Change the dataset to return class labels

## Performance Notes

- **Training Time**: ~10 hours for 100 epochs (RTX 3080)
- **Memory Usage**: ~8GB VRAM at default settings
- **Dataset Size**: ~2GB after extraction
- **Sample Generation**: ~30 seconds for 64 samples

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021)
- [Tiny ImageNet Dataset](https://www.kaggle.com/c/tiny-imagenet)

## License

This project is for educational purposes. Please ensure you comply with the Tiny ImageNet dataset license terms.

## Contributing

This is a minimal educational implementation. For production use, consider:
- More sophisticated architectures
- Better sampling methods (DDIM, DPM-Solver)
- Conditional generation
- Better evaluation metrics
- Multi-GPU training support