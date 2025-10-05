import React, { useState } from 'react';
import { Download, Copy, Check } from 'lucide-react';

export default function ReadmeGenerator() {
  const [copied, setCopied] = useState(false);
  const [config, setConfig] = useState({
    projectName: 'Simple DDPM',
    description: 'A clean, minimal implementation of Denoising Diffusion Probabilistic Models (DDPM) for image generation on Fashion-MNIST.',
    author: 'Your Name',
    dataset: 'Fashion-MNIST',
    imageSize: '32x32',
    includeTOC: true,
    includeVisuals: true,
    includeContributing: true,
    includeLicense: true,
    licenseType: 'MIT'
  });

  const generateReadme = () => {
    let readme = '';

    // Title and description
    if (config.includeVisuals) {
      readme += `<div align="center">\n\n`;
    }
    readme += `# ${config.projectName}\n\n`;
    readme += `${config.description}\n\n`;
    
    if (config.includeVisuals) {
      readme += `<img src="path/to/your/generated_samples_grid.png" alt="Generated Fashion-MNIST Samples" width="400">\n\n`;
      readme += `*Sample grid of generated ${config.dataset} images after training*\n\n`;
      readme += `</div>\n\n`;
    }

    // Badges or visual separator
    if (config.includeVisuals) {
      readme += `---\n\n`;
    }

    // Table of Contents
    if (config.includeTOC) {
      readme += `## 📋 Table of Contents\n\n`;
      readme += `- [Overview](#overview)\n`;
      readme += `- [Features](#features)\n`;
      readme += `- [Project Structure](#project-structure)\n`;
      readme += `- [Installation](#installation)\n`;
      readme += `- [Usage](#usage)\n`;
      readme += `  - [Training](#training)\n`;
      readme += `  - [Sampling](#sampling)\n`;
      readme += `  - [Evaluation](#evaluation)\n`;
      readme += `- [Configuration](#configuration)\n`;
      readme += `- [Model Architecture](#model-architecture)\n`;
      readme += `- [Results](#results)\n`;
      if (config.includeContributing) readme += `- [Contributing](#contributing)\n`;
      if (config.includeLicense) readme += `- [License](#license)\n`;
      readme += `- [References](#references)\n\n`;
    }

    // Overview
    readme += `## 🔍 Overview\n\n`;
    readme += `This project implements Denoising Diffusion Probabilistic Models (DDPM), a class of generative models that learn to generate images by gradually denoising random noise. The implementation is trained on ${config.dataset} with ${config.imageSize} grayscale images.\n\n`;
    readme += `**Key Features:**\n`;
    readme += `- Clean, educational implementation focused on core concepts\n`;
    readme += `- Flexible noise scheduling (linear and cosine)\n`;
    readme += `- Exponential Moving Average (EMA) for stable sampling\n`;
    readme += `- Comprehensive evaluation metrics (FID, Inception Score)\n`;
    readme += `- TensorBoard integration for monitoring\n`;
    readme += `- Checkpoint management and resumable training\n\n`;

    // Features
    readme += `## ✨ Features\n\n`;
    readme += `- 🎨 **Image Generation**: Generate high-quality ${config.imageSize} images from pure noise\n`;
    readme += `- 🏗️ **U-Net Architecture**: ResNet blocks with self-attention mechanisms\n`;
    readme += `- 📊 **Multiple Metrics**: FID score and Inception Score evaluation\n`;
    readme += `- 💾 **Smart Checkpointing**: Automatic saving of best models and resume capability\n`;
    readme += `- 📈 **Training Monitoring**: Real-time loss tracking with TensorBoard\n`;
    readme += `- 🔄 **Flexible Sampling**: Generate individual samples, grids, and interpolations\n`;
    readme += `- ⚡ **GPU Accelerated**: CUDA support with automatic device detection\n`;
    readme += `- 🎯 **Production Ready**: Proper error handling and logging\n\n`;

    // Project Structure
    readme += `## 📁 Project Structure\n\n`;
    readme += `\`\`\`\n`;
    readme += `.\n`;
    readme += `├── config.py                # Central configuration file\n`;
    readme += `├── train.py                 # Main training script\n`;
    readme += `├── sample.py                # Sampling and generation script\n`;
    readme += `├── eval.py                  # Evaluation with FID/IS metrics\n`;
    readme += `├── requirements.txt         # Python dependencies\n`;
    readme += `├── src/\n`;
    readme += `│   ├── dataset.py           # Data loading utilities\n`;
    readme += `│   ├── model.py             # U-Net architecture implementation\n`;
    readme += `│   ├── diffusion.py         # DDPM forward/reverse diffusion process\n`;
    readme += `│   └── utils.py             # Helper functions and utilities\n`;
    readme += `├── data/                    # Dataset directory (auto-created)\n`;
    readme += `│   └── fashion-mnist/       # Fashion-MNIST data\n`;
    readme += `├── checkpoints/             # Model checkpoints\n`;
    readme += `│   ├── best.pth             # Best model (lowest loss)\n`;
    readme += `│   ├── latest.pth           # Latest checkpoint for resuming\n`;
    readme += `│   └── final.pth            # Final epoch checkpoint\n`;
    readme += `├── outputs/                 # Generated samples\n`;
    readme += `│   ├── samples_epoch_*.png  # Training samples\n`;
    readme += `│   ├── losses.png           # Loss curve plot\n`;
    readme += `│   └── eval/                # Evaluation results\n`;
    readme += `└── logs/                    # Training logs\n`;
    readme += `    └── tensorboard/         # TensorBoard logs\n`;
    readme += `\`\`\`\n\n`;

    // Installation
    readme += `## 🚀 Installation\n\n`;
    readme += `### Prerequisites\n\n`;
    readme += `- Python 3.8+\n`;
    readme += `- CUDA-capable GPU (recommended, but CPU works too)\n`;
    readme += `- 8GB+ VRAM for default settings\n\n`;
    readme += `### Setup\n\n`;
    readme += `1. **Clone the repository:**\n`;
    readme += `\`\`\`bash\n`;
    readme += `git clone https://github.com/yourusername/simple-ddpm.git\n`;
    readme += `cd simple-ddpm\n`;
    readme += `\`\`\`\n\n`;
    readme += `2. **Install dependencies:**\n`;
    readme += `\`\`\`bash\n`;
    readme += `pip install -r requirements.txt\n`;
    readme += `\`\`\`\n\n`;
    readme += `**Required packages:**\n`;
    readme += `- \`torch>=2.0.0\` - PyTorch framework\n`;
    readme += `- \`torchvision>=0.15.0\` - Vision utilities\n`;
    readme += `- \`numpy>=1.21.0\` - Numerical operations\n`;
    readme += `- \`matplotlib>=3.5.0\` - Visualization\n`;
    readme += `- \`tqdm>=4.64.0\` - Progress bars\n`;
    readme += `- \`tensorboard>=2.10.0\` - Training monitoring\n`;
    readme += `- \`einops>=0.6.0\` - Tensor operations\n`;
    readme += `- \`torchmetrics\` - Evaluation metrics (for eval.py)\n\n`;

    // Usage
    readme += `## 💻 Usage\n\n`;
    
    // Training
    readme += `### Training\n\n`;
    readme += `Start training with default configuration:\n\n`;
    readme += `\`\`\`bash\n`;
    readme += `python train.py\n`;
    readme += `\`\`\`\n\n`;
    readme += `The script will:\n`;
    readme += `- Automatically download ${config.dataset} dataset\n`;
    readme += `- Create necessary directories\n`;
    readme += `- Train the model with progress bars\n`;
    readme += `- Save checkpoints every 5 epochs\n`;
    readme += `- Generate sample images every 10 epochs\n`;
    readme += `- Log metrics to TensorBoard\n\n`;
    readme += `**Resume training from checkpoint:**\n`;
    readme += `Training automatically resumes from \`checkpoints/latest.pth\` if it exists.\n\n`;
    readme += `**Monitor training:**\n`;
    readme += `\`\`\`bash\n`;
    readme += `tensorboard --logdir logs/tensorboard\n`;
    readme += `\`\`\`\n`;
    readme += `Then open http://localhost:6006 in your browser.\n\n`;

    // Sampling
    readme += `### Sampling\n\n`;
    readme += `Generate samples from a trained model:\n\n`;
    readme += `**Generate 64 samples:**\n`;
    readme += `\`\`\`bash\n`;
    readme += `python sample.py --checkpoint checkpoints/best.pth --num_samples 64\n`;
    readme += `\`\`\`\n\n`;
    readme += `**Generate interpolation between samples:**\n`;
    readme += `\`\`\`bash\n`;
    readme += `python sample.py --checkpoint checkpoints/best.pth --interpolate --interp_steps 10\n`;
    readme += `\`\`\`\n\n`;
    readme += `**Custom output directory:**\n`;
    readme += `\`\`\`bash\n`;
    readme += `python sample.py --num_samples 100 --output_dir custom_outputs/\n`;
    readme += `\`\`\`\n\n`;
    readme += `**Available arguments:**\n`;
    readme += `- \`--checkpoint\`: Path to model checkpoint (default: \`checkpoints/best.pth\`)\n`;
    readme += `- \`--num_samples\`: Number of samples to generate (default: 64)\n`;
    readme += `- \`--output_dir\`: Output directory for samples\n`;
    readme += `- \`--interpolate\`: Generate interpolation sequence\n`;
    readme += `- \`--interp_steps\`: Number of interpolation steps (default: 10)\n\n`;

    // Evaluation
    readme += `### Evaluation\n\n`;
    readme += `Compute quantitative metrics (FID and Inception Score):\n\n`;
    readme += `\`\`\`bash\n`;
    readme += `python eval.py\n`;
    readme += `\`\`\`\n\n`;
    readme += `This will:\n`;
    readme += `- Generate 1000 samples from the best checkpoint\n`;
    readme += `- Compute FID (Fréchet Inception Distance) score\n`;
    readme += `- Compute Inception Score (IS)\n`;
    readme += `- Display and save a sample grid\n`;
    readme += `- Save results to \`outputs/eval/\`\n\n`;

    // Configuration
    readme += `## ⚙️ Configuration\n\n`;
    readme += `All hyperparameters are defined in \`config.py\`. Key settings:\n\n`;
    readme += `### Training Parameters\n\n`;
    readme += `\`\`\`python\n`;
    readme += `BATCH_SIZE = 64           # Training batch size\n`;
    readme += `LEARNING_RATE = 1e-3      # Adam learning rate\n`;
    readme += `EPOCHS = 200              # Total training epochs\n`;
    readme += `GRAD_CLIP = 1.0           # Gradient clipping threshold\n`;
    readme += `\`\`\`\n\n`;
    readme += `### Model Parameters\n\n`;
    readme += `\`\`\`python\n`;
    readme += `IMAGE_SIZE = 32           # Image resolution (32x32)\n`;
    readme += `CHANNELS = 1              # Grayscale images\n`;
    readme += `TIME_STEPS = 1000         # Diffusion timesteps\n`;
    readme += `UNET_DIM = 32             # Base U-Net dimensions\n`;
    readme += `UNET_DIM_MULTS = [1, 2, 4, 8]  # Channel multipliers per level\n`;
    readme += `\`\`\`\n\n`;
    readme += `### Diffusion Parameters\n\n`;
    readme += `\`\`\`python\n`;
    readme += `BETA_START = 0.0001       # Starting noise level\n`;
    readme += `BETA_END = 0.02           # Ending noise level\n`;
    readme += `SCHEDULE_TYPE = 'cosine'  # Noise schedule: 'linear' or 'cosine'\n`;
    readme += `\`\`\`\n\n`;
    readme += `### Paths\n\n`;
    readme += `\`\`\`python\n`;
    readme += `DATA_PATH = './data/fashion-mnist'\n`;
    readme += `CHECKPOINT_PATH = './checkpoints'\n`;
    readme += `OUTPUT_PATH = './outputs'\n`;
    readme += `LOG_PATH = './logs'\n`;
    readme += `\`\`\`\n\n`;

    // Architecture
    readme += `## 🏗️ Model Architecture\n\n`;
    readme += `### U-Net Backbone\n\n`;
    readme += `The model uses a U-Net architecture optimized for diffusion models:\n\n`;
    readme += `**Encoder (Downsampling):**\n`;
    readme += `- 4 levels with progressive channel multiplication: [1×, 2×, 4×, 8×]\n`;
    readme += `- Each level contains ResNet blocks with time embeddings\n`;
    readme += `- Self-attention at resolutions 16×16 and 8×8\n`;
    readme += `- Downsampling via strided convolutions\n\n`;
    readme += `**Middle Block:**\n`;
    readme += `- Two ResNet blocks with self-attention\n`;
    readme += `- Processes features at lowest resolution\n\n`;
    readme += `**Decoder (Upsampling):**\n`;
    readme += `- Mirror of encoder with skip connections\n`;
    readme += `- TransposedConv2d for upsampling\n`;
    readme += `- Concatenates features from corresponding encoder levels\n\n`;
    readme += `### Key Components\n\n`;
    readme += `1. **Time Embeddings**: Sinusoidal positional encodings for timestep information\n`;
    readme += `2. **ResNet Blocks**: Two-layer blocks with GroupNorm and SiLU activation\n`;
    readme += `3. **Self-Attention**: Multi-head attention for capturing long-range dependencies\n`;
    readme += `4. **Skip Connections**: U-Net style connections preserving spatial information\n\n`;
    readme += `**Model Size:** ~${config.imageSize === '32x32' ? '2M' : '90M'} parameters\n\n`;

    // Training Details
    readme += `## 🎓 Training Details\n\n`;
    readme += `### Forward Diffusion Process\n\n`;
    readme += `Images are gradually corrupted with Gaussian noise over T=1000 timesteps:\n\n`;
    readme += `\`\`\`\n`;
    readme += `q(x_t | x_0) = N(x_t; sqrt(α̅_t)x_0, (1-α̅_t)I)\n`;
    readme += `\`\`\`\n\n`;
    readme += `Where α̅_t is the cumulative product of noise schedule coefficients.\n\n`;
    readme += `### Reverse Diffusion Process\n\n`;
    readme += `The model learns to denoise images step by step:\n\n`;
    readme += `\`\`\`\n`;
    readme += `p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))\n`;
    readme += `\`\`\`\n\n`;
    readme += `The network predicts noise ε_θ(x_t, t) at each timestep.\n\n`;
    readme += `### Loss Function\n\n`;
    readme += `Simple MSE loss between predicted and actual noise:\n\n`;
    readme += `\`\`\`\n`;
    readme += `L = ||ε - ε_θ(x_t, t)||²\n`;
    readme += `\`\`\`\n\n`;
    readme += `### Optimization\n\n`;
    readme += `- **Optimizer**: Adam with default betas\n`;
    readme += `- **Learning Rate**: 1e-3 (configurable)\n`;
    readme += `- **Gradient Clipping**: Norm clipping at 1.0\n`;
    readme += `- **EMA**: Exponential moving average (β=0.995) for stable sampling\n\n`;

    // Results
    readme += `## 📈 Results\n\n`;
    readme += `### Training Progress\n\n`;
    readme += `Typical training on ${config.dataset} (200 epochs):\n\n`;
    readme += `| Epoch | Loss | Sample Quality |\n`;
    readme += `|-------|------|----------------|\n`;
    readme += `| 10 | ~0.08 | Basic shapes and patterns |\n`;
    readme += `| 50 | ~0.04 | Recognizable clothing items |\n`;
    readme += `| 100 | ~0.02 | Clear textures and details |\n`;
    readme += `| 200 | ~0.01 | High-quality diverse samples |\n\n`;
    readme += `### Evaluation Metrics\n\n`;
    readme += `After 200 epochs of training:\n`;
    readme += `- **FID Score**: Lower is better (measures distribution similarity)\n`;
    readme += `- **Inception Score**: Higher is better (measures quality and diversity)\n\n`;
    readme += `Results vary based on dataset and training duration.\n\n`;
    readme += `### Sample Visualization\n\n`;
    readme += `Generated samples show:\n`;
    readme += `- Good diversity across different classes\n`;
    readme += `- Coherent structure and textures\n`;
    readme += `- Some artifacts at lower training epochs\n`;
    readme += `- Smooth interpolations between samples\n\n`;
    readme += `Check \`outputs/\` directory for generated images after training.\n\n`;

    // Performance Notes
    readme += `## ⚡ Performance Notes\n\n`;
    readme += `### Hardware Requirements\n\n`;
    readme += `- **Minimum**: 8GB VRAM GPU (RTX 2060 or equivalent)\n`;
    readme += `- **Recommended**: 12GB+ VRAM GPU (RTX 3080 or better)\n`;
    readme += `- **CPU Training**: Possible but ~50x slower\n\n`;
    readme += `### Training Time\n\n`;
    readme += `On NVIDIA RTX 3080:\n`;
    readme += `- 200 epochs: ~3-4 hours\n`;
    readme += `- Single epoch: ~1 minute\n`;
    readme += `- Sample generation (64 images): ~30 seconds\n\n`;
    readme += `### Memory Optimization\n\n`;
    readme += `If you encounter OOM errors:\n\n`;
    readme += `\`\`\`python\n`;
    readme += `# In config.py, reduce:\n`;
    readme += `BATCH_SIZE = 32  # or 16\n`;
    readme += `NUM_SAMPLES = 16  # instead of 64\n`;
    readme += `TIME_STEPS = 500  # fewer diffusion steps\n`;
    readme += `\`\`\`\n\n`;

    // Troubleshooting
    readme += `## 🔧 Troubleshooting\n\n`;
    readme += `### Common Issues\n\n`;
    readme += `**Dataset download fails:**\n`;
    readme += `\`\`\`bash\n`;
    readme += `# Manually download Fashion-MNIST\n`;
    readme += `# It will be automatically extracted to data/fashion-mnist/\n`;
    readme += `\`\`\`\n\n`;
    readme += `**CUDA out of memory:**\n`;
    readme += `- Reduce \`BATCH_SIZE\` in config.py\n`;
    readme += `- Reduce \`NUM_SAMPLES\` for generation\n`;
    readme += `- Close other GPU applications\n\n`;
    readme += `**Training is too slow:**\n`;
    readme += `- Ensure GPU is being used: check TensorBoard or console output\n`;
    readme += `- Reduce \`TIME_STEPS\` for faster training (may affect quality)\n`;
    readme += `- Use mixed precision training (requires code modification)\n\n`;
    readme += `**Poor sample quality:**\n`;
    readme += `- Train for more epochs (200+)\n`;
    readme += `- Check if loss is decreasing steadily\n`;
    readme += `- Try cosine noise schedule instead of linear\n`;
    readme += `- Ensure EMA is being used for sampling\n\n`;

    // Contributing
    if (config.includeContributing) {
      readme += `## 🤝 Contributing\n\n`;
      readme += `Contributions are welcome! Here's how you can help:\n\n`;
      readme += `1. Fork the repository\n`;
      readme += `2. Create a feature branch (\`git checkout -b feature/amazing-feature\`)\n`;
      readme += `3. Commit your changes (\`git commit -m 'Add amazing feature'\`)\n`;
      readme += `4. Push to the branch (\`git push origin feature/amazing-feature\`)\n`;
      readme += `5. Open a Pull Request\n\n`;
      readme += `### Ideas for Contributions\n\n`;
      readme += `- Support for additional datasets (CIFAR-10, CelebA, etc.)\n`;
      readme += `- Conditional generation (class-conditional, text-conditional)\n`;
      readme += `- Advanced sampling methods (DDIM, DPM-Solver)\n`;
      readme += `- Model architecture improvements\n`;
      readme += `- Better evaluation metrics\n`;
      readme += `- Multi-GPU training support\n`;
      readme += `- Web demo interface\n\n`;
    }

    // License
    if (config.includeLicense) {
      readme += `## 📄 License\n\n`;
      readme += `This project is licensed under the ${config.licenseType} License - see the LICENSE file for details.\n\n`;
    }

    // References
    readme += `## 📚 References\n\n`;
    readme += `### Papers\n\n`;
    readme += `- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)\n`;
    readme += `  - Original DDPM paper\n`;
    readme += `- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021)\n`;
    readme += `  - Improved training techniques\n`;
    readme += `- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) (Dhariwal & Nichol, 2021)\n`;
    readme += `  - Architecture improvements\n\n`;
    readme += `### Resources\n\n`;
    readme += `- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)\n`;
    readme += `- [PyTorch Documentation](https://pytorch.org/docs/)\n`;
    readme += `- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)\n\n`;

    // Acknowledgments
    readme += `## 🙏 Acknowledgments\n\n`;
    readme += `- Original DDPM implementation by Jonathan Ho et al.\n`;
    readme += `- PyTorch team for the excellent deep learning framework\n`;
    readme += `- Fashion-MNIST dataset by Zalando Research\n`;
    readme += `- Open-source community for inspiration and code references\n\n`;

    // Footer
    readme += `---\n\n`;
    readme += `**Author**: ${config.author}\n\n`;
    readme += `If you find this implementation helpful, please consider giving it a ⭐!\n\n`;
    readme += `For questions or issues, please open an issue on GitHub.\n`;

    return readme;
  };

  const handleCopy = () => {
    const readme = generateReadme();
    navigator.clipboard.writeText(readme);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const readme = generateReadme();
    const blob = new Blob([readme], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'README.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-xl shadow-lg p-8 mb-6">
          <h1 className="text-3xl font-bold text-slate-800 mb-2">
            DDPM README Generator
          </h1>
          <p className="text-slate-600 mb-6">
            Customize and generate a professional README for your Diffusion Model project
          </p>

          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Project Name
              </label>
              <input
                type="text"
                value={config.projectName}
                onChange={(e) => setConfig({...config, projectName: e.target.value})}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Author Name
              </label>
              <input
                type="text"
                value={config.author}
                onChange={(e) => setConfig({...config, author: e.target.value})}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Dataset
              </label>
              <input
                type="text"
                value={config.dataset}
                onChange={(e) => setConfig({...config, dataset: e.target.value})}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Image Size
              </label>
              <input
                type="text"
                value={config.imageSize}
                onChange={(e) => setConfig({...config, imageSize: e.target.value})}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
            </div>
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Description
            </label>
            <textarea
              value={config.description}
              onChange={(e) => setConfig({...config, description: e.target.value})}
              rows={3}
              className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>

          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <label className="flex items-center space-x-3 p-3 bg-slate-50 rounded-lg cursor-pointer hover:bg-slate-100 transition">
              <input
                type="checkbox"
                checked={config.includeTOC}
                onChange={(e) => setConfig({...config, includeTOC: e.target.checked})}
                className="w-4 h-4 text-purple-600"
              />
              <span className="text-sm text-slate-700">Include Table of Contents</span>
            </label>

            <label className="flex items-center space-x-3 p-3 bg-slate-50 rounded-lg cursor-pointer hover:bg-slate-100 transition">
              <input
                type="checkbox"
                checked={config.includeVisuals}
                onChange={(e) => setConfig({...config, includeVisuals: e.target.checked})}
                className="w-4 h-4 text-purple-600"
              />
              <span className="text-sm text-slate-700">Include Visual Elements</span>
            </label>

            <label className="flex items-center space-x-3 p-3 bg-slate-50 rounded-lg cursor-pointer hover:bg-slate-100 transition">
              <input
                type="checkbox"
                checked={config.includeContributing}
                onChange={(e) => setConfig({...config, includeContributing: e.target.checked})}
                className="w-4 h-4 text-purple-600"
              />
              <span className="text-sm text-slate-700">Include Contributing Section</span>
            </label>

            <label className="flex items-center space-x-3 p-3 bg-slate-50 rounded-lg cursor-pointer hover:bg-slate-100 transition">
              <input
                type="checkbox"
                checked={config.includeLicense}
                onChange={(e) => setConfig({...config, includeLicense: e.target.checked})}
                className="w-4 h-4 text-purple-600"
              />
              <span className="text-sm text-slate-700">Include License Section</span>
            </label>
          </div>

          {config.includeLicense && (
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                License Type
              </label>
              <select
                value={config.licenseType}
                onChange={(e) => setConfig({...config, licenseType: e.target.value})}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              >
                <option>MIT</option>
                <option>Apache 2.0</option>
                <option>GPL-3.0</option>
                <option>BSD-3-Clause</option>
              </select>
            </div>
          )}

          <div className="flex gap-4">
            <button
              onClick={handleCopy}
              className="flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition font-medium"
            >
              {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
              {copied ? 'Copied!' : 'Copy to Clipboard'}
            </button>
            <button
              onClick={handleDownload}
              className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium"
            >
              <Download className="w-5 h-5" />
              Download README.md
            </button>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-xl font-bold text-slate-800 mb-4">Preview</h2>
          <div className="bg-slate-50 rounded-lg p-6 max-h-96 overflow-y-auto">
            <pre className="text-sm text-slate-700 whitespace-pre-wrap font-mono">
              {generateReadme()}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
