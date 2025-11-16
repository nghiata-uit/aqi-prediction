"""
Baseline Models for Speech Enhancement

This module implements baseline comparison models:
- CNN-based U-Net
- Transformer-U-Net
- Conformer-U-Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math

from .unet_components import UNetEncoder, UNetDecoder


class CNNUNet(nn.Module):
    """
    Traditional CNN-based U-Net for speech enhancement.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        channels: List of channel numbers for each stage
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
        activation: Activation function
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: List[int] = [64, 128, 256, 512],
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = 'relu',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            channels=channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
            dropout=dropout,
        )
        
        # Decoder
        decoder_channels = channels[::-1]
        self.decoder = UNetDecoder(
            channels=decoder_channels,
            out_channels=out_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN U-Net."""
        bottleneck, skip_connections = self.encoder(x)
        output = self.decoder(bottleneck, skip_connections)
        return output
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class TransformerBlock(nn.Module):
    """
    Transformer block for sequence modeling.
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = x + ff_out
        x = self.norm2(x)
        
        return x


class TransformerUNet(nn.Module):
    """
    U-Net with Transformer blocks in the bottleneck.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        channels: List of channel numbers for each stage
        num_transformer_layers: Number of Transformer layers
        num_heads: Number of attention heads
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
        activation: Activation function
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: List[int] = [64, 128, 256, 512],
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = 'relu',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            channels=channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
            dropout=dropout,
        )
        
        # Transformer blocks in bottleneck
        bottleneck_dim = channels[-1]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=bottleneck_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_transformer_layers)
        ])
        
        # Decoder
        decoder_channels = channels[::-1]
        self.decoder = UNetDecoder(
            channels=decoder_channels,
            out_channels=out_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Transformer U-Net."""
        # Encoder
        bottleneck, skip_connections = self.encoder(x)
        
        # Reshape for Transformer: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, h, w = bottleneck.shape
        bottleneck_seq = bottleneck.permute(0, 3, 1, 2).reshape(b, w, c * h)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            bottleneck_seq = transformer_block(bottleneck_seq)
        
        # Reshape back: (batch, time, channels*freq) -> (batch, channels, freq, time)
        bottleneck = bottleneck_seq.reshape(b, w, c, h).permute(0, 2, 3, 1)
        
        # Decoder
        output = self.decoder(bottleneck, skip_connections)
        
        return output
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class ConformerConvModule(nn.Module):
    """
    Convolution module for Conformer.
    
    Args:
        dim: Model dimension
        kernel_size: Convolution kernel size
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dim,
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolution module.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # x: (batch, seq_len, dim)
        x = x.transpose(1, 2)  # (batch, dim, seq_len)
        
        # Pointwise convolution and GLU activation
        x = self.pointwise_conv1(x)  # (batch, 2*dim, seq_len)
        x = F.glu(x, dim=1)  # (batch, dim, seq_len)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        
        # Pointwise convolution
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # (batch, seq_len, dim)
        return x


class ConformerBlock(nn.Module):
    """
    Conformer block combining self-attention and convolution.
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        dim_feedforward: Dimension of feedforward network
        conv_kernel_size: Convolution kernel size
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Feed-forward module 1
        self.feed_forward1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )
        
        # Self-attention module
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(dim)
        self.attention_dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.conv_module = ConformerConvModule(
            dim=dim,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.conv_norm = nn.LayerNorm(dim)
        
        # Feed-forward module 2
        self.feed_forward2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )
        
        self.final_norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Conformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Feed-forward module 1 with residual connection (half-step residual)
        x = x + 0.5 * self.feed_forward1(x)
        
        # Self-attention with residual connection
        attn_out, _ = self.attention(
            self.attention_norm(x),
            self.attention_norm(x),
            self.attention_norm(x),
        )
        x = x + self.attention_dropout(attn_out)
        
        # Convolution module with residual connection
        x = x + self.conv_module(self.conv_norm(x))
        
        # Feed-forward module 2 with residual connection (half-step residual)
        x = x + 0.5 * self.feed_forward2(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x


class ConformerUNet(nn.Module):
    """
    U-Net with Conformer blocks in the bottleneck.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        channels: List of channel numbers for each stage
        num_conformer_blocks: Number of Conformer blocks
        num_heads: Number of attention heads
        dim_feedforward: Dimension of feedforward network
        conv_kernel_size: Convolution kernel size for Conformer
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
        activation: Activation function
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: List[int] = [64, 128, 256, 512],
        num_conformer_blocks: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = 'relu',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            channels=channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
            dropout=dropout,
        )
        
        # Conformer blocks in bottleneck
        bottleneck_dim = channels[-1]
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                dim=bottleneck_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_conformer_blocks)
        ])
        
        # Decoder
        decoder_channels = channels[::-1]
        self.decoder = UNetDecoder(
            channels=decoder_channels,
            out_channels=out_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Conformer U-Net."""
        # Encoder
        bottleneck, skip_connections = self.encoder(x)
        
        # Reshape for Conformer: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, h, w = bottleneck.shape
        bottleneck_seq = bottleneck.permute(0, 3, 1, 2).reshape(b, w, c * h)
        
        # Conformer blocks
        for conformer_block in self.conformer_blocks:
            bottleneck_seq = conformer_block(bottleneck_seq)
        
        # Reshape back: (batch, time, channels*freq) -> (batch, channels, freq, time)
        bottleneck = bottleneck_seq.reshape(b, w, c, h).permute(0, 2, 3, 1)
        
        # Decoder
        output = self.decoder(bottleneck, skip_connections)
        
        return output
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test all baseline models
    batch_size = 2
    in_channels = 1
    freq_bins = 257
    time_steps = 100
    
    x = torch.randn(batch_size, in_channels, freq_bins, time_steps)
    
    print("Testing baseline models...")
    print(f"Input shape: {x.shape}\n")
    
    # Test CNN U-Net
    print("--- CNN U-Net ---")
    cnn_unet = CNNUNet()
    output = cnn_unet(x)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {cnn_unet.get_num_params():,}\n")
    
    # Test Transformer U-Net
    print("--- Transformer U-Net ---")
    transformer_unet = TransformerUNet(num_transformer_layers=4)
    output = transformer_unet(x)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {transformer_unet.get_num_params():,}\n")
    
    # Test Conformer U-Net
    print("--- Conformer U-Net ---")
    conformer_unet = ConformerUNet(num_conformer_blocks=4)
    output = conformer_unet(x)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {conformer_unet.get_num_params():,}\n")
