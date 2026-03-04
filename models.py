"""
Neural network models for flow matching - Production Version

Provides U-Net architecture with classifier-free guidance for conditional generation.
"""

from abc import ABC, abstractmethod
from typing import List
import math
import torch


class ConditionalVectorField(torch.nn.Module, ABC):
    """
    Abstract interface for conditional vector field u_t^θ(x|y).
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute conditional vector field.
        
        Args:
            x: Input state, shape (batch_size, *dims)
            t: Time, shape (batch_size, 1, 1, 1)
            y: Conditioning labels, shape (batch_size,)
        
        Returns:
            Vector field, shape (batch_size, *dims)
        """
        pass


class FourierEncoder(torch.nn.Module):
    """
    Fourier feature time embedding.
    
    Encodes time t ∈ [0,1] into high-dimensional features:
        embedding = [sin(2π·t·w), cos(2π·t·w)] · √2
    
    Args:
        dim: Embedding dimension (must be even)
    """
    
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"
        self.half_dim = dim // 2
        self.register_buffer('weights', torch.randn(1, self.half_dim))
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode time into Fourier features.
        
        Args:
            t: Time tensor, shape (batch_size, 1, 1, 1)
        
        Returns:
            Fourier embeddings, shape (batch_size, dim)
        """
        t = t.view(-1, 1)
        freqs = t * self.weights * 2 * math.pi
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1) * math.sqrt(2)


class ResidualLayer(torch.nn.Module):
    """
    Residual block with FiLM conditioning.
    
    Args:
        channels: Number of feature channels
        time_embed_dim: Time embedding dimension
        y_embed_dim: Label embedding dimension
    """
    
    def __init__(self, channels: int, time_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.BatchNorm2d(channels),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.BatchNorm2d(channels),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.time_adapter = torch.nn.Sequential(
            torch.nn.Linear(time_embed_dim, time_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim, channels)
        )
        self.y_adapter = torch.nn.Sequential(
            torch.nn.Linear(y_embed_dim, y_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(y_embed_dim, channels)
        )
    
    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """Forward pass with conditioning."""
        res = x.clone()
        x = self.block1(x)
        x = x + self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1)
        x = x + self.y_adapter(y_embed).unsqueeze(-1).unsqueeze(-1)
        x = self.block2(x)
        return x + res


class Encoder(torch.nn.Module):
    """U-Net encoder: residual blocks + downsampling."""
    
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int,
                 t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = torch.nn.ModuleList([
            ResidualLayer(channels_in, t_embed_dim, y_embed_dim)
            for _ in range(num_residual_layers)
        ])
        self.downsample = torch.nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
        return self.downsample(x)


class Midcoder(torch.nn.Module):
    """U-Net bottleneck: residual blocks at lowest resolution."""
    
    def __init__(self, channels: int, num_residual_layers: int,
                 t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = torch.nn.ModuleList([
            ResidualLayer(channels, t_embed_dim, y_embed_dim)
            for _ in range(num_residual_layers)
        ])
    
    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
        return x


class Decoder(torch.nn.Module):
    """U-Net decoder: upsampling + residual blocks."""
    
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int,
                 t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.upsample = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            torch.nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)
        )
        self.res_blocks = torch.nn.ModuleList([
            ResidualLayer(channels_out, t_embed_dim, y_embed_dim)
            for _ in range(num_residual_layers)
        ])
    
    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
        return x


class UNet(ConditionalVectorField):
    """
    U-Net for conditional image generation with classifier-free guidance.
    
    Implements u_t^θ(x|y) using U-Net architecture with skip connections
    and FiLM conditioning.
    
    Args:
        in_channels: Input image channels (e.g., 1 for grayscale, 3 for RGB)
        out_channels: Output channels (typically same as in_channels)
        channels: Channel progression, e.g., [32, 64, 128]
        num_residual_layers: Residual blocks per encoder/decoder
        t_embed_dim: Time embedding dimension
        y_embed_dim: Label embedding dimension
        num_classes: Number of classes + 1 for null label (e.g., 11 for MNIST)
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 channels: List[int] = [32, 64, 128], num_residual_layers: int = 2,
                 t_embed_dim: int = 40, y_embed_dim: int = 40, num_classes: int = 11):
        super().__init__()
        
        self.init_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(channels[0]),
            torch.nn.SiLU()
        )
        
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.y_embedder = torch.nn.Embedding(num_embeddings=num_classes, embedding_dim=y_embed_dim)
        
        # Build encoder-decoder pairs
        encoders = []
        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(Decoder(next_c, curr_c, num_residual_layers, t_embed_dim, y_embed_dim))
        
        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(reversed(decoders))
        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)
        
        self.final_conv = torch.nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy images, shape (batch_size, in_channels, H, W)
            t: Time, shape (batch_size, 1, 1, 1)
            y: Class labels, shape (batch_size,)
        
        Returns:
            Vector field, shape (batch_size, out_channels, H, W)
        """
        t_embed = self.time_embedder(t)
        y_embed = self.y_embedder(y)
        
        x = self.init_conv(x)
        
        # Encoder with skip connections
        residuals = []
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed)
            residuals.append(x.clone())
        
        # Bottleneck
        x = self.midcoder(x, t_embed, y_embed)
        
        # Decoder with skip connections
        for decoder in self.decoders:
            x = x + residuals.pop()
            x = decoder(x, t_embed, y_embed)
        
        return self.final_conv(x)
