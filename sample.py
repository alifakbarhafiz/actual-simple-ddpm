"""
Sampling script for Simple DDPM
"""
import os
import torch
import argparse
from datetime import datetime

import config
from src.model import create_model
from src.diffusion import create_diffusion, DDPMSampler
from src.utils import load_checkpoint, save_image_grid, get_device

def generate_samples(model_path, num_samples=64, output_dir=None, use_ema=True):
    """Generate samples from trained model"""
    device = get_device()
    
    # Create model
    print("Creating model...")
    model = create_model()
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading model from {model_path}...")
    load_checkpoint(model, None, model_path, device)
    
    # Create diffusion process
    print("Creating diffusion process...")
    diffusion = create_diffusion()
    diffusion = diffusion.to(device)
    
    # Create sampler
    sampler = DDPMSampler(diffusion)
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    samples = sampler.sample(
        model=model,
        num_samples=num_samples,
        image_size=config.IMAGE_SIZE,
        channels=config.CHANNELS,
        device=str(device)
    )
    
    # Save samples
    if output_dir is None:
        output_dir = config.OUTPUT_PATH
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as grid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_path = os.path.join(output_dir, f'generated_samples_{timestamp}.png')
    
    nrow = int(num_samples ** 0.5)  # Square grid
    if nrow * nrow < num_samples:
        nrow += 1
    
    save_image_grid(samples, grid_path, nrow=nrow)
    print(f"Samples saved to {grid_path}")
    
    # Save individual samples
    individual_dir = os.path.join(output_dir, f'individual_samples_{timestamp}')
    os.makedirs(individual_dir, exist_ok=True)
    
    for i, sample in enumerate(samples):
        sample_path = os.path.join(individual_dir, f'sample_{i:04d}.png')
        save_image_grid(sample.unsqueeze(0), sample_path, nrow=1)
    
    print(f"Individual samples saved to {individual_dir}")
    
    return samples


def interpolate_samples(model_path, num_steps=10, output_dir=None):
    """Generate interpolation between random samples"""
    device = get_device()
    
    # Create model and load checkpoint
    model = create_model()
    model = model.to(device)
    load_checkpoint(model, None, model_path, device)
    
    # Create diffusion process
    diffusion = create_diffusion()
    diffusion = diffusion.to(device)
    
    print(f"Generating interpolation with {num_steps} steps...")
    
    # Generate two random starting points
    shape = (1, config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    start_noise = torch.randn(shape, device=device)
    end_noise = torch.randn(shape, device=device)
    
    # Interpolate in noise space
    interpolated_samples = []
    
    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        interpolated_noise = (1 - alpha) * start_noise + alpha * end_noise
        
        # Run reverse process
        model.eval()
        with torch.no_grad():
            sample = diffusion.p_sample_loop(model, interpolated_noise.shape, device)
        
        interpolated_samples.append(sample)
    
    # Concatenate samples
    all_samples = torch.cat(interpolated_samples, dim=0)
    
    # Save interpolation
    if output_dir is None:
        output_dir = config.OUTPUT_PATH
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    interp_path = os.path.join(output_dir, f'interpolation_{timestamp}.png')
    
    save_image_grid(all_samples, interp_path, nrow=num_steps)
    print(f"Interpolation saved to {interp_path}")
    
    return all_samples

def main():
    """Main sampling function"""
    parser = argparse.ArgumentParser(description='Generate samples from trained DDPM')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: config.OUTPUT_PATH)')
    parser.add_argument('--interpolate', action='store_true',
                        help='Generate interpolation between samples')
    parser.add_argument('--interp_steps', type=int, default=10,
                        help='Number of interpolation steps')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Available checkpoints:")
        checkpoint_dir = os.path.dirname(args.checkpoint)
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pth'):
                    print(f"  {os.path.join(checkpoint_dir, f)}")
        return
    
    try:
        if args.interpolate:
            # Generate interpolation
            interpolate_samples(
                model_path=args.checkpoint,
                num_steps=args.interp_steps,
                output_dir=args.output_dir
            )
        else:
            # Generate regular samples
            generate_samples(
                model_path=args.checkpoint,
                num_samples=args.num_samples,
                output_dir=args.output_dir
            )
            
    except Exception as e:
        print(f"Sampling failed: {e}")
        raise
    
    print("Sampling completed!")

if __name__ == "__main__":
    main()