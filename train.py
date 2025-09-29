"""
Training script for Simple DDPM
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

import config
from src.dataset import create_dataloader
from src.model import create_model
from src.diffusion import create_diffusion
from src.utils import (
    save_checkpoint, load_checkpoint, save_image_grid, 
    plot_losses, count_parameters, setup_logging, 
    log_images_to_tensorboard, get_device, EMA
)

def train_one_epoch(model, diffusion, dataloader, optimizer, device, epoch, writer=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for step, batch in enumerate(pbar):
        # Handle (images, labels) tuple
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        
        # Move data to device
        batch = batch.to(device)
        
        # Random timesteps
        t = torch.randint(0, config.TIME_STEPS, (batch.shape[0],), device=device).long()
        
        # Compute loss
        loss = diffusion.get_loss(model, batch, t)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'avg_loss': f'{avg_loss:.6f}'
        })
        
        # Log to tensorboard
        if writer and step % config.LOG_EVERY == 0:
            global_step = epoch * num_batches + step
            writer.add_scalar('Loss/Train', loss.item(), global_step)
            writer.add_scalar('Loss/Average', avg_loss, global_step)
    
    return total_loss / num_batches


@torch.no_grad()
def sample_and_save(model, diffusion, device, epoch, writer=None):
    """Generate and save samples"""
    model.eval()
    
    print(f"Generating samples for epoch {epoch}...")
    
    # Generate samples
    samples = diffusion.sample(
        model=model,
        image_size=config.IMAGE_SIZE,
        batch_size=config.NUM_SAMPLES,
        channels=config.CHANNELS
    )
    
    # Save as grid
    save_path = os.path.join(config.OUTPUT_PATH, f'samples_epoch_{epoch:04d}.png')
    save_image_grid(samples, save_path, nrow=config.SAMPLE_GRID_SIZE)
    print(f"Samples saved to {save_path}")
    
    # Log to tensorboard
    if writer:
        log_images_to_tensorboard(writer, samples, epoch, "samples")


def main():
    """Main training function"""
    # Setup
    print("Setting up training...")
    device = get_device()
    
    # Create model
    print("Creating model...")
    model = create_model()
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create diffusion process
    print("Creating diffusion process...")
    diffusion = create_diffusion()
    diffusion = diffusion.to(device)
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = create_dataloader(train=True)
    dataset = getattr(dataloader, 'dataset', None)
    if dataset is not None and hasattr(dataset, '__len__'):
        print(f"Dataset size: {len(dataset)}")
    else:
        print("Dataset size: Unknown (dataset does not implement __len__)")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Setup EMA
    ema = EMA(model, beta=0.995)
    
    # Setup logging
    writer = setup_logging()
    
    # Training state
    start_epoch = 0
    losses = []
    best_loss = float('inf')
    epoch = start_epoch  # Ensure epoch is always defined
    avg_loss = None  # Ensure avg_loss is always defined
    
    # Load checkpoint if exists
    checkpoint_path = os.path.join(config.CHECKPOINT_PATH, 'latest.pth')
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        start_epoch, best_loss = load_checkpoint(model, optimizer, checkpoint_path, device)
        start_epoch += 1
        epoch = start_epoch  # Update epoch if checkpoint is loaded
    
    # Training loop
    print(f"Starting training from epoch {start_epoch}...")
    print(f"Training on device: {device}")
    
    try:
        for epoch in range(start_epoch, config.EPOCHS):
            epoch_start_time = time.time()
            
            # Train one epoch
            avg_loss = train_one_epoch(model, diffusion, dataloader, optimizer, device, epoch, writer)
            losses.append(avg_loss)
            
            # Update EMA
            ema.update()
            
            # Log epoch info
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time:.1f}s, Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if epoch % config.SAVE_EVERY == 0 or avg_loss < best_loss:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_path = os.path.join(config.CHECKPOINT_PATH, 'best.pth')
                    print(f"New best loss: {best_loss:.6f}")
                else:
                    save_path = os.path.join(config.CHECKPOINT_PATH, 'latest.pth')
                
                save_checkpoint(model, optimizer, epoch, avg_loss, save_path)
            
            # Generate samples
            if epoch % config.SAMPLE_EVERY == 0 or epoch == config.EPOCHS - 1:
                # Use EMA weights for sampling
                ema.apply_shadow()
                sample_and_save(model, diffusion, device, epoch, writer)
                ema.restore()
            
            # Log to tensorboard
            writer.add_scalar('Loss/Epoch', avg_loss, epoch)
            
            # Plot losses
            if epoch % (config.SAVE_EVERY * 2) == 0:
                plot_path = os.path.join(config.OUTPUT_PATH, 'losses.png')
                plot_losses(losses, plot_path)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    
    finally:
        # Final save
        print("Saving final checkpoint...")
        final_path = os.path.join(config.CHECKPOINT_PATH, 'final.pth')
        save_checkpoint(model, optimizer, epoch, avg_loss, final_path)
        
        # Final samples with EMA
        print("Generating final samples...")
        ema.apply_shadow()
        sample_and_save(model, diffusion, device, config.EPOCHS - 1, writer)
        
        # Close writer
        writer.close()
        
        print("Training completed!")


if __name__ == "__main__":
    main()