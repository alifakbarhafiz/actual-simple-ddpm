<div align="center">

# Simple DDPM

A clean, minimal implementation of Denoising Diffusion Probabilistic Models (DDPM) for image generation on Fashion-MNIST.

<img src="path/to/your/generated_samples_grid.png" alt="Generated Fashion-MNIST Samples" width="400">

*Sample grid of generated Fashion-MNIST images after training*

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Sampling](#sampling)
  - [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## 🔍 Overview

This project implements Denoising Diffusion Probabilistic Models (DDPM), a class of generative models that learn to generate images by gradually denoising random noise. The implementation is trained on Fashion-MNIST with 32x32 grayscale images.

**Key Features:**
- Clean, educational implementation focused on core concepts
- Flexible noise scheduling (linear and cosine)
- Exponential Moving Average (EMA) for stable sampling
- Comprehensive evaluation metrics (FID, Inception Score)
- TensorBoard integration for monitoring
- Checkpoint management and resumable training

## ✨ Features

- 🎨 **Image Generation**: Generate high-quality 32x32 images from pure noise
- 🏗️ **U-Net Architecture**: ResNet blocks with self-attention mechanisms
- 📊 **Multiple Metrics**: FID score and Inception Score evaluation
- 💾 **Smart Checkpointing**: Automatic saving of best models and resume capability
- 📈 **Training Monitoring**: Real-time loss tracking with TensorBoard
- 🔄 **Flexible Sampling**: Generate individual samples, grids, and interpolations
- ⚡ **GPU Accelerated**: CUDA support with automatic device detection
- 🎯 **Production Ready**: Proper error handling and logging

## 📁 Project Structure

```
.
├── config.py                # Central configuration file
├── train.py                 # Main training script
├── sample.py                # Sampling and generation script
├── eval.py                  # Evaluation with FID/IS metrics
├── requirements.txt         # Python dependencies
├── src/
│   ├── dataset.py           # Data loading utilities
│   ├── model.py             # U-Net architecture implementation
│   ├── diffusion.py         # DDPM forward/reverse diffusion process
│   └── utils.py             # Helper functions and utilities
├── data/                    # Dataset directory (auto-created)
│   └── fashion-mnist/       # Fashion-MNIST data
├── checkpoints/             # Model checkpoints
│   ├── best.pth             # Best model (lowest loss)
│   ├── latest.pth           # Latest checkpoint for resuming
│   └── final.pth            # Final epoch checkpoint
├── outputs/                 # Generated samples
│   ├── samples_epoch_*.png  # Training samples
│   ├── losses.png           # Loss curve plot
│   └── eval/                # Evaluation results
└── logs/                    # Training logs
    └── tensorboard/         # TensorBoard logs
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works too)
- 8GB+ VRAM for default settings

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/simple-ddpm.git
cd simple-ddpm
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Required packages:**
- `torch>=2.0.0` - PyTorch framework
- `torchvision>=0.15.0` - Vision utilities
- `numpy>=1.21.0` - Numerical operations
- `matplotlib>=3.5.0` - Visualization
- `tqdm>=4.64.0` - Progress bars
- `tensorboard>=2.10.0` - Training monitoring
- `einops>=0.6.0` - Tensor operations
- `torchmetrics` - Evaluation metrics (for eval.py)

## 💻 Usage

### Training

Start training with default configuration:

```bash
python train.py
```

The script will:
- Automatically download Fashion-MNIST dataset
- Create necessary directories
- Train the model with progress bars
- Save checkpoints every 5 epochs
- Generate sample images every 10 epochs
- Log metrics to TensorBoard

**Resume training from checkpoint:**
Training automatically resumes from `checkpoints/latest.pth` if it exists.

**Monitor training:**
```bash
tensorboard --logdir logs/tensorboard
```
Then open http://localhost:6006 in your browser.

### Sampling

Generate samples from a trained model:

**Generate 64 samples:**
```bash
python sample.py --checkpoint checkpoints/best.pth --num_samples 64
```

**Generate interpolation between samples:**
```bash
python sample.py --checkpoint checkpoints/best.pth --interpolate --interp_steps 10
```

**Custom output directory:**
```bash
python sample.py --num_samples 100 --output_dir custom_outputs/
```

**Available arguments:**
- `--checkpoint`: Path to model checkpoint (default: `checkpoints/best.pth`)
- `--num_samples`: Number of samples to generate (default: 64)
- `--output_dir`: Output directory for samples
- `--interpolate`: Generate interpolation sequence
- `--interp_steps`: Number of interpolation steps (default: 10)

### Evaluation

Compute quantitative metrics (FID and Inception Score):

```bash
python eval.py
```

This will:
- Generate 1000 samples from the best checkpoint
- Compute FID (Fréchet Inception Distance) score
- Compute Inception Score (IS)
- Display and save a sample grid
- Save results to `outputs/eval/`

## ⚙️ Configuration

All hyperparameters are defined in `config.py`. Key settings:

### Training Parameters

```python
BATCH_SIZE = 64           # Training batch size
LEARNING_RATE = 1e-3      # Adam learning rate
EPOCHS = 200              # Total training epochs
GRAD_CLIP = 1.0           # Gradient clipping threshold
```

### Model Parameters

```python
IMAGE_SIZE = 32           # Image resolution (32x32)
CHANNELS = 1              # Grayscale images
TIME_STEPS = 1000         # Diffusion timesteps
UNET_DIM = 32             # Base U-Net dimensions
UNET_DIM_MULTS = [1, 2, 4, 8]  # Channel multipliers per level
```

### Diffusion Parameters

```python
BETA_START = 0.0001       # Starting noise level
BETA_END = 0.02           # Ending noise level
SCHEDULE_TYPE = 'cosine'  # Noise schedule: 'linear' or 'cosine'
```

### Paths

```python
DATA_PATH = './data/fashion-mnist'
CHECKPOINT_PATH = './checkpoints'
OUTPUT_PATH = './outputs'
LOG_PATH = './logs'
```

## 🏗️ Model Architecture

### U-Net Backbone

The model uses a U-Net architecture optimized for diffusion models:

**Encoder (Downsampling):**
- 4 levels with progressive channel multiplication: [1×, 2×, 4×, 8×]
- Each level contains ResNet blocks with time embeddings
- Self-attention at resolutions 16×16 and 8×8
- Downsampling via strided convolutions

**Middle Block:**
- Two ResNet blocks with self-attention
- Processes features at lowest resolution

**Decoder (Upsampling):**
- Mirror of encoder with skip connections
- TransposedConv2d for upsampling
- Concatenates features from corresponding encoder levels

### Key Components

1. **Time Embeddings**: Sinusoidal positional encodings for timestep information
2. **ResNet Blocks**: Two-layer blocks with GroupNorm and SiLU activation
3. **Self-Attention**: Multi-head attention for capturing long-range dependencies
4. **Skip Connections**: U-Net style connections preserving spatial information

**Model Size:** ~2M parameters

## 🎓 Training Details

### Forward Diffusion Process

Images are gradually corrupted with Gaussian noise over T=1000 timesteps:

```
q(x_t | x_0) = N(x_t; sqrt(α̅_t)x_0, (1-α̅_t)I)
```

Where α̅_t is the cumulative product of noise schedule coefficients.

### Reverse Diffusion Process

The model learns to denoise images step by step:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

The network predicts noise ε_θ(x_t, t) at each timestep.

### Loss Function

Simple MSE loss between predicted and actual noise:

```
L = ||ε - ε_θ(x_t, t)||²
```

### Optimization

- **Optimizer**: Adam with default betas
- **Learning Rate**: 1e-3 (configurable)
- **Gradient Clipping**: Norm clipping at 1.0
- **EMA**: Exponential moving average (β=0.995) for stable sampling

## 📈 Results

### Training Progress

Typical training on Fashion-MNIST (200 epochs):

| Epoch | Loss | Sample Quality |
|-------|------|----------------|
| 10 | ~0.08 | Basic shapes and patterns |
| 50 | ~0.04 | Recognizable clothing items |
| 100 | ~0.02 | Clear textures and details |
| 200 | ~0.01 | High-quality diverse samples |

### Evaluation Metrics

After 200 epochs of training:
- **FID Score**: Lower is better (measures distribution similarity)
- **Inception Score**: Higher is better (measures quality and diversity)

Results vary based on dataset and training duration.

### Sample Visualization

Generated samples show:
- Good diversity across different classes
- Coherent structure and textures
- Some artifacts at lower training epochs
- Smooth interpolations between samples

Check `outputs/` directory for generated images after training.

## ⚡ Performance Notes

### Hardware Requirements

- **Minimum**: 8GB VRAM GPU (RTX 2060 or equivalent)
- **Recommended**: 12GB+ VRAM GPU (RTX 3080 or better)
- **CPU Training**: Possible but ~50x slower

### Training Time

On NVIDIA RTX 3080:
- 200 epochs: ~3-4 hours
- Single epoch: ~1 minute
- Sample generation (64 images): ~30 seconds

### Memory Optimization

If you encounter OOM errors:

```python
# In config.py, reduce:
BATCH_SIZE = 32  # or 16
NUM_SAMPLES = 16  # instead of 64
TIME_STEPS = 500  # fewer diffusion steps
```

## 🔧 Troubleshooting

### Common Issues

**Dataset download fails:**
```bash
# Manually download Fashion-MNIST
# It will be automatically extracted to data/fashion-mnist/
```

**CUDA out of memory:**
- Reduce `BATCH_SIZE` in config.py
- Reduce `NUM_SAMPLES` for generation
- Close other GPU applications

**Training is too slow:**
- Ensure GPU is being used: check TensorBoard or console output
- Reduce `TIME_STEPS` for faster training (may affect quality)
- Use mixed precision training (requires code modification)

**Poor sample quality:**
- Train for more epochs (200+)
- Check if loss is decreasing steadily
- Try cosine noise schedule instead of linear
- Ensure EMA is being used for sampling

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Ideas for Contributions

- Support for additional datasets (CIFAR-10, CelebA, etc.)
- Conditional generation (class-conditional, text-conditional)
- Advanced sampling methods (DDIM, DPM-Solver)
- Model architecture improvements
- Better evaluation metrics
- Multi-GPU training support
- Web demo interface

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

### Papers

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
  - Original DDPM paper
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021)
  - Improved training techniques
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) (Dhariwal & Nichol, 2021)
  - Architecture improvements

### Resources

- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

## 🙏 Acknowledgments

- Original DDPM implementation by Jonathan Ho et al.
- PyTorch team for the excellent deep learning framework
- Fashion-MNIST dataset by Zalando Research
- Open-source community for inspiration and code references

---

**Author**: Alif Akbar Hafiz

If you find this implementation helpful, please consider giving it a ⭐!

For questions or issues, please open an issue on GitHub.

