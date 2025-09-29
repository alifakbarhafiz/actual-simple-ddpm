"""
Utility functions for Simple DDPM
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config

def exists(x):
    """Check if value exists (not None)"""
    return x is not None


def default(val, d):
    """Return val if it exists, otherwise return d"""
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """
    Extract values from a 1-D tensor `a` at positions `t` (batch of indices),
    and reshape to match x_shape for broadcasting.
    """
    # Ensure t has correct shape and is on the same device
    b = t.shape[0]
    out = a.gather(-1, t.to(a.device))  # FIX: keep on same device
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """Create linear beta schedule"""
    return torch.linspace(start, end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """Create cosine beta schedule"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def get_beta_schedule(schedule_type, timesteps):
    """Get beta schedule based on type"""
    if schedule_type == 'linear':
        return linear_beta_schedule(timesteps, config.BETA_START, config.BETA_END)
    elif schedule_type == 'cosine':
        return cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def unnormalize_to_zero_to_one(t):
    """Convert from [-1, 1] to [0, 1]"""
    return (t + 1) * 0.5


def normalize_to_neg_one_to_one(img):
    """Convert from [0, 1] to [-1, 1]"""
    return img * 2 - 1


def save_image_grid(images, path, nrow=8, normalize=True):
    """Save a grid of images"""
    import torchvision.utils as vutils
    
    if normalize:
        images = unnormalize_to_zero_to_one(images)
    
    # Ensure images are in [0, 1] range
    images = torch.clamp(images, 0., 1.)
    
    # Create directory if it doesn't exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save grid
    vutils.save_image(images, path, nrow=nrow, padding=2, pad_value=1)


def plot_losses(losses, save_path=None):
    """Plot training losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {path} (epoch {epoch}, loss {loss:.6f})")
    return epoch, loss


class EMA:
    """Exponential Moving Average for model parameters"""
    
    def __init__(self, model, beta=0.9999):
        self.model = model
        self.beta = beta
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.beta * self.shadow[name] + (1 - self.beta) * param.data
    
    def apply_shadow(self):
        """Apply shadow parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def setup_logging():
    """Setup logging directory"""
    from torch.utils.tensorboard import SummaryWriter
    
    log_dir = Path(config.LOG_PATH) / 'tensorboard'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(log_dir))
    return writer


def log_images_to_tensorboard(writer, images, step, tag="samples", nrow=8):
    """Log images to tensorboard"""
    import torchvision.utils as vutils
    
    if isinstance(images, torch.Tensor):
        # Normalize to [0, 1]
        images = unnormalize_to_zero_to_one(images)
        images = torch.clamp(images, 0., 1.)
        
        # Create grid
        grid = vutils.make_grid(images, nrow=nrow, padding=2, pad_value=1)
        
        # Log to tensorboard
        writer.add_image(tag, grid, step)


def get_device():
    """Get the device to use for training"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon) device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test device detection
    device = get_device()
    print(f"Device: {device}")
    
    # Test beta schedules
    linear_betas = linear_beta_schedule(1000)
    cosine_betas = cosine_beta_schedule(1000)
    print(f"Linear betas range: {linear_betas.min():.6f} - {linear_betas.max():.6f}")
    print(f"Cosine betas range: {cosine_betas.min():.6f} - {cosine_betas.max():.6f}")
    
    print("Utilities test passed!")