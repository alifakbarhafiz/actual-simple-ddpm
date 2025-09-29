"""
DDPM diffusion process implementation
"""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import config
from .utils import extract, get_beta_schedule

class GaussianDiffusion:
    """Gaussian Diffusion Process for DDPM"""
    
    def __init__(self, timesteps=1000, schedule_type='linear'):
        self.timesteps = timesteps
        self.schedule_type = schedule_type

        # Get beta schedule, ensure torch.float32 tensor
        betas = get_beta_schedule(schedule_type, timesteps)
        if not isinstance(betas, torch.Tensor):
            betas = torch.tensor(betas, dtype=torch.float32)
        else:
            betas = betas.to(dtype=torch.float32)

        # Pre-compute values
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # previous cumulative product (prepend 1.0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=self.alphas_cumprod.dtype), self.alphas_cumprod[:-1]]
        )

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Sample x_t from x_0 and t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def get_loss(self, model, x_start, t, noise=None):
        """
        Compute the loss for training
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward process
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Predict noise
        predicted_noise = model(x_noisy, t)
        
        # Simple MSE loss
        loss = F.mse_loss(noise, predicted_noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """
        Sample x_{t-1} from x_t using the model
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Use our model to predict the noise
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, device):
        """
        Generate samples by running the reverse process
        """
        b = shape[0]
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        
        return img
    
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        """
        Generate samples from the model
        """
        device = next(model.parameters()).device
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), device=device)
    
    def to(self, device):
        """Move all tensors to device"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self


class DDPMSampler:
    """DDPM sampling interface"""
    
    def __init__(self, diffusion_model):
        self.diffusion = diffusion_model
    
    def sample(self, model, num_samples, image_size, channels=3, device='cuda'):
        """Generate samples"""
        model.eval()
        
        with torch.no_grad():
            # Generate samples in batches to avoid memory issues
            all_samples = []
            batch_size = min(num_samples, 16)  # Process in batches
            
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                samples = self.diffusion.sample(
                    model=model,
                    image_size=image_size,
                    batch_size=current_batch_size,
                    channels=channels
                )
                all_samples.append(samples)
            
            if len(all_samples) == 1:
                return all_samples[0]
            else:
                return torch.cat(all_samples, dim=0)


def create_diffusion():
    """Create diffusion process with config parameters"""
    diffusion = GaussianDiffusion(
        timesteps=config.TIME_STEPS,
        schedule_type=config.SCHEDULE_TYPE
    )
    return diffusion


def test_diffusion():
    """Test the diffusion process"""
    print("Testing diffusion process...")
    
    # Create diffusion
    diffusion = create_diffusion()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = diffusion.to(device)
    
    # Create dummy data
    batch_size = 4
    x = torch.randn(batch_size, config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    t = torch.randint(0, config.TIME_STEPS, (batch_size,)).to(device)
    
    # Test forward process
    noise = torch.randn_like(x)
    x_noisy = diffusion.q_sample(x, t, noise)
    
    print(f"Original shape: {x.shape}")
    print(f"Noisy shape: {x_noisy.shape}")
    print(f"Time steps shape: {t.shape}")
    
    # Test that noise level increases with time
    t_early = torch.zeros(batch_size, dtype=torch.long).to(device)
    t_late = torch.full((batch_size,), config.TIME_STEPS - 1, dtype=torch.long).to(device)
    
    x_early = diffusion.q_sample(x, t_early, noise)
    x_late = diffusion.q_sample(x, t_late, noise)
    
    early_noise = torch.mean((x - x_early) ** 2).item()
    late_noise = torch.mean((x - x_late) ** 2).item()
    
    print(f"Early timestep noise level: {early_noise:.6f}")
    print(f"Late timestep noise level: {late_noise:.6f}")
    
    assert late_noise > early_noise, "Noise should increase with timestep"
    print("Diffusion test passed!")
    
    return diffusion


if __name__ == "__main__":
    test_diffusion()