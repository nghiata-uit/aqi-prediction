"""
U-Net Components for Mamba-SEUNet

This module implements encoder and decoder blocks for the U-Net architecture
with multi-scale feature learning and skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class EncoderBlock(nn.Module):
    """
    Encoder block with downsampling for U-Net.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 3)
        stride: Stride for downsampling (default: 2)
        padding: Padding for convolution (default: 1)
        use_batch_norm: Whether to use batch normalization (default: True)
        activation: Activation function (default: 'relu')
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        use_batch_norm: bool = True,
        activation: str = 'relu',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        layers = []
        
        # First convolution
        layers.append(nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batch_norm,
        ))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        # Second convolution (same spatial size)
        layers.append(nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=not use_batch_norm,
        ))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder block.
        
        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch, out_channels, height//stride, width//stride)
        """
        return self.conv_block(x)


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling for U-Net.
    
    Args:
        in_channels: Number of input channels
        skip_channels: Number of channels from skip connection
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 3)
        stride: Stride for upsampling (default: 2)
        padding: Padding for convolution (default: 1)
        use_batch_norm: Whether to use batch normalization (default: True)
        activation: Activation function (default: 'relu')
        dropout: Dropout rate (default: 0.1)
        upsample_mode: Upsampling mode ('transpose' or 'bilinear') (default: 'transpose')
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        use_batch_norm: bool = True,
        activation: str = 'relu',
        dropout: float = 0.1,
        upsample_mode: str = 'transpose',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        
        # Upsampling
        if upsample_mode == 'transpose':
            self.upsample = nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=stride,
                stride=stride,
            )
        elif upsample_mode == 'bilinear':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown upsample_mode: {upsample_mode}")
        
        # Convolution after concatenation
        layers = []
        concat_channels = in_channels + skip_channels
        
        layers.append(nn.Conv2d(
            concat_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=not use_batch_norm,
        ))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        # Second convolution
        layers.append(nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=not use_batch_norm,
        ))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder block.
        
        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
            skip: Skip connection tensor of shape (batch, skip_channels, height*stride, width*stride)
            
        Returns:
            Output tensor of shape (batch, out_channels, height*stride, width*stride)
        """
        # Upsample
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Convolution
        x = self.conv_block(x)
        
        return x


class UNetEncoder(nn.Module):
    """
    U-Net Encoder with multiple downsampling stages.
    
    Args:
        in_channels: Number of input channels
        channels: List of channel numbers for each stage
        num_stages: Number of encoder stages
        kernel_size: Convolution kernel size
        stride: Stride for downsampling
        padding: Padding for convolution
        use_batch_norm: Whether to use batch normalization
        activation: Activation function
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        channels: List[int] = [64, 128, 256, 512],
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        use_batch_norm: bool = True,
        activation: str = 'relu',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_stages = len(channels)
        
        self.encoder_blocks = nn.ModuleList()
        
        current_channels = in_channels
        for out_channels in channels:
            self.encoder_blocks.append(
                EncoderBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    use_batch_norm=use_batch_norm,
                    activation=activation,
                    dropout=dropout,
                )
            )
            current_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
            
        Returns:
            Tuple of (bottleneck tensor, list of skip connection tensors)
        """
        skip_connections = []
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
        
        # Last skip connection is the bottleneck
        bottleneck = skip_connections.pop()
        
        return bottleneck, skip_connections


class UNetDecoder(nn.Module):
    """
    U-Net Decoder with multiple upsampling stages.
    
    Args:
        channels: List of channel numbers for each stage (in reverse order from encoder)
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Stride for upsampling
        padding: Padding for convolution
        use_batch_norm: Whether to use batch normalization
        activation: Activation function
        dropout: Dropout rate
        upsample_mode: Upsampling mode
    """
    
    def __init__(
        self,
        channels: List[int] = [512, 256, 128, 64],
        out_channels: int = 1,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        use_batch_norm: bool = True,
        activation: str = 'relu',
        dropout: float = 0.1,
        upsample_mode: str = 'transpose',
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.num_stages = len(channels)
        
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            skip_ch = channels[i + 1]
            out_ch = channels[i + 1]
            
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    use_batch_norm=use_batch_norm,
                    activation=activation,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                )
            )
        
        # Final convolution to output channels
        self.final_conv = nn.Conv2d(
            channels[-1],
            out_channels,
            kernel_size=1,
        )
    
    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Bottleneck tensor of shape (batch, channels[0], height, width)
            skip_connections: List of skip connection tensors (in reverse order)
            
        Returns:
            Output tensor of shape (batch, out_channels, height*stride^n, width*stride^n)
        """
        # Reverse skip connections to match decoder order
        skip_connections = skip_connections[::-1]
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x, skip_connections[i])
        
        # Final convolution
        x = self.final_conv(x)
        
        return x


if __name__ == "__main__":
    # Test encoder and decoder
    batch_size = 2
    in_channels = 1
    height = 256
    width = 256
    
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Test encoder
    encoder = UNetEncoder(
        in_channels=in_channels,
        channels=[64, 128, 256, 512],
    )
    
    bottleneck, skip_connections = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Bottleneck shape: {bottleneck.shape}")
    print(f"Number of skip connections: {len(skip_connections)}")
    for i, skip in enumerate(skip_connections):
        print(f"  Skip {i} shape: {skip.shape}")
    
    # Test decoder
    decoder = UNetDecoder(
        channels=[512, 256, 128, 64],
        out_channels=in_channels,
    )
    
    output = decoder(bottleneck, skip_connections)
    print(f"\nOutput shape: {output.shape}")
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters())}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters())}")
