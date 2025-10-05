"""
Robust U-Net model for DDPM - Clean implementation without type issues
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        inv_freq = math.log(10000) / (half_dim - 1)
        inv_freq = torch.exp(torch.arange(half_dim, device=device) * -inv_freq)
        emb = time[:, None].float() * inv_freq[None, :]  
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConvBlock(nn.Module):
    """Basic convolution block with GroupNorm and SiLU"""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        gn = max(1, min(groups, out_channels))
        self.norm = nn.GroupNorm(gn, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block with time embedding"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, groups: int = 8):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.block1 = ConvBlock(in_channels, out_channels, groups)
        self.block2 = ConvBlock(out_channels, out_channels, groups)

        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)

        time_emb_proj = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb_proj

        h = self.block2(h)
        return h + self.residual(x)

class AttentionBlock(nn.Module):
    """Self-attention block (simple, memory-conscious implementation)"""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        # use a safe number of groups
        gn = max(1, min(groups, channels))
        self.norm = nn.GroupNorm(gn, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        # Flatten spatial dims
        q = self.q(h).view(B, C, H * W)         # (B, C, N)
        k = self.k(h).view(B, C, H * W)         # (B, C, N)
        v = self.v(h).view(B, C, H * W)         # (B, C, N)

        # compute attention: (B, N, N)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / math.sqrt(C), dim=-1)

        # apply attention to values -> (B, C, N)
        h_attn = torch.bmm(v, attn.transpose(1, 2))

        h_attn = h_attn.view(B, C, H, W)
        return x + self.out(h_attn)

class DownBlock(nn.Module):
    """Downsampling block: two residuals (in->out, out->out), optional attn, then downsample"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 has_attn: bool = False, groups: int = 8):
        super().__init__()
        self.resnet1 = ResidualBlock(in_channels, out_channels, time_emb_dim, groups)
        self.resnet2 = ResidualBlock(out_channels, out_channels, time_emb_dim, groups)
        self.attn = AttentionBlock(out_channels, groups) if has_attn else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        h1 = self.resnet1(x, time_emb)   
        h2 = self.resnet2(h1, time_emb)  
        h2 = self.attn(h2)
        out = self.downsample(h2)
        return out, h1, h2

class UpBlock(nn.Module):
    """Upsampling block: ConvTranspose2d upsample, then two residuals mixing skips"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 has_attn: bool = False, groups: int = 8):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.resnet1 = ResidualBlock(out_channels + out_channels, out_channels, time_emb_dim, groups)
        self.resnet2 = ResidualBlock(out_channels + out_channels, out_channels, time_emb_dim, groups)
        self.attn = AttentionBlock(out_channels, groups) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, skip1: torch.Tensor, skip2: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)  
        h = torch.cat([x, skip2], dim=1)
        h = self.resnet1(h, time_emb)
        h = torch.cat([h, skip1], dim=1)
        h = self.resnet2(h, time_emb)
        h = self.attn(h)
        return h

class MiddleBlock(nn.Module):
    """Middle block with attention"""

    def __init__(self, channels: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.resnet1 = ResidualBlock(channels, channels, time_emb_dim, groups)
        self.attn = AttentionBlock(channels, groups)
        self.resnet2 = ResidualBlock(channels, channels, time_emb_dim, groups)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.resnet1(x, time_emb)
        h = self.attn(h)
        h = self.resnet2(h, time_emb)
        return h

class UNet(nn.Module):
    """Clean U-Net implementation for DDPM"""

    def __init__(
        self,
        in_channels: int = 1,
        model_channels: int = 32,
        out_channels: int = 1,
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        channel_mult: tuple = (1, 2, 4, 8),
        num_groups: int = 8
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = tuple(channel_mult)

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Use channel_mult directly 
        ch_mult = list(self.channel_mult)
        self.num_resolutions = len(ch_mult)

        # Build down blocks
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [ch]

        for level in range(self.num_resolutions):
            out_ch = model_channels * ch_mult[level]
            has_attn = (64 // (2 ** level)) in attention_resolutions
            self.down_blocks.append(
                DownBlock(ch, out_ch, time_embed_dim, has_attn, num_groups)
            )
            ch = out_ch
            input_block_chans.append(ch)

        # Middle block
        self.middle_block = MiddleBlock(ch, time_embed_dim, num_groups)

        # Build up blocks (reverse of down)
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            out_ch = model_channels * ch_mult[level]  
            has_attn = (64 // (2 ** level)) in attention_resolutions
            self.up_blocks.append(
                UpBlock(ch, out_ch, time_embed_dim, has_attn, num_groups)
            )
            ch = out_ch

        # Final layers
        gn = max(1, min(num_groups, ch))
        self.final_conv = nn.Sequential(
            nn.GroupNorm(gn, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )

        # Store skip connection channels for reference
        self.input_block_chans = input_block_chans

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # Time embedding
        time_emb = self.time_embed(timesteps)

        # Initial conv
        h = self.input_conv(x)

        # Store skip connections
        skip_connections = [h]

        # Downsampling
        for down_block in self.down_blocks:
            h, skip1, skip2 = down_block(h, time_emb)
            skip_connections.extend([skip1, skip2])

        # Middle
        h = self.middle_block(h, time_emb)

        # Upsampling
        for up_block in self.up_blocks:
            skip2 = skip_connections.pop()
            skip1 = skip_connections.pop()
            h = up_block(h, skip1, skip2, time_emb)

        # Final output
        return self.final_conv(h)

def create_model():
    """Create U-Net model with config parameters"""
    model = UNet(
        in_channels=config.CHANNELS,
        model_channels=config.UNET_DIM,
        out_channels=config.CHANNELS,
        channel_mult=tuple(config.UNET_DIM_MULTS),
        num_groups=config.UNET_RESNET_BLOCK_GROUPS
    )
    return model

def test_model():
    """Test the model"""
    print("Testing U-Net model...")

    model = create_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    batch_size = 4
    x = torch.randn(batch_size, config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)

    with torch.no_grad():
        output = model(x, t)

    print(f"Input shape: {x.shape}")
    print(f"Time shape: {t.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input shape {x.shape}"
    print("Model test passed!")

    return model