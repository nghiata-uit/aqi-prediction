"""
Mamba-SEUNet: Mamba UNet for Monaural Speech Enhancement

This module implements the complete Mamba-SEUNet architecture combining
U-Net encoder-decoder with Mamba state-space model blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .mamba_block import MambaBlock, StackedMambaBlocks
from .unet_components import UNetEncoder, UNetDecoder


class MambaSEUNet(nn.Module):
    """
    Mamba-SEUNet model for speech enhancement.
    
    Architecture:
        1. Input: Noisy speech magnitude spectrogram
        2. U-Net Encoder: Multi-scale feature extraction with downsampling
        3. TS-Mamba Blocks: Bidirectional temporal modeling in bottleneck
        4. U-Net Decoder: Multi-scale reconstruction with upsampling
        5. Output: Clean speech magnitude spectrogram
    
    Args:
        in_channels: Number of input channels (default: 1 for magnitude spectrogram)
        out_channels: Number of output channels (default: 1)
        encoder_channels: List of encoder channel numbers (default: [64, 128, 256, 512])
        num_mamba_blocks: Number of TS-Mamba blocks in bottleneck (default: 6)
        mamba_state_dim: State dimension for Mamba SSM (default: 16)
        mamba_conv_kernel: Convolution kernel size for Mamba (default: 4)
        mamba_expand_factor: Expansion factor for Mamba (default: 2)
        dropout: Dropout rate (default: 0.1)
        use_batch_norm: Whether to use batch normalization (default: True)
        activation: Activation function (default: 'relu')
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: List[int] = [64, 128, 256, 512],
        num_mamba_blocks: int = 6,
        mamba_state_dim: int = 16,
        mamba_conv_kernel: int = 4,
        mamba_expand_factor: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = 'relu',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channels = encoder_channels
        self.num_mamba_blocks = num_mamba_blocks
        
        # U-Net Encoder
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            channels=encoder_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
            dropout=dropout,
        )
        
        # Bottleneck dimension (last encoder channel)
        bottleneck_channels = encoder_channels[-1]
        
        # Note: We'll flatten the frequency dimension with channels for Mamba processing
        # This will be handled dynamically in forward pass
        # The Mamba blocks will process sequences where each timestep contains all frequency bins
        
        # Placeholder for bottleneck feature dimension (will be channels * freq_bins)
        # This will be determined at runtime based on input shape
        self.bottleneck_channels = bottleneck_channels
        self.mamba_state_dim = mamba_state_dim
        self.mamba_conv_kernel = mamba_conv_kernel
        self.mamba_expand_factor = mamba_expand_factor
        self.num_mamba_blocks = num_mamba_blocks
        self.mamba_dropout = dropout
        
        # TS-Mamba Blocks will be created lazily on first forward pass
        self.mamba_blocks = None
        
        # U-Net Decoder
        decoder_channels = encoder_channels[::-1]  # Reverse encoder channels
        self.decoder = UNetDecoder(
            channels=decoder_channels,
            out_channels=out_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
            dropout=dropout,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba-SEUNet.
        
        Args:
            x: Input noisy spectrogram of shape (batch, in_channels, freq, time)
            
        Returns:
            Enhanced spectrogram of shape (batch, out_channels, freq, time)
        """
        batch_size = x.shape[0]
        
        # Encoder: Extract multi-scale features
        bottleneck, skip_connections = self.encoder(x)
        # bottleneck shape: (batch, channels[-1], freq_reduced, time_reduced)
        
        # Store spatial dimensions for reshaping
        b, c, h, w = bottleneck.shape
        
        # Initialize Mamba blocks on first forward pass with correct dimension
        if self.mamba_blocks is None:
            mamba_dim = c * h  # channels * freq
            self.mamba_blocks = StackedMambaBlocks(
                num_blocks=self.num_mamba_blocks,
                dim=mamba_dim,
                state_dim=self.mamba_state_dim,
                conv_kernel=self.mamba_conv_kernel,
                expand_factor=self.mamba_expand_factor,
                bidirectional=True,
                dropout=self.mamba_dropout,
            ).to(x.device)
        
        # Reshape for Mamba: (batch, channels, freq, time) -> (batch, time, channels*freq)
        # Treat frequency bins as part of the feature dimension
        bottleneck_seq = bottleneck.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        bottleneck_seq = bottleneck_seq.reshape(b, w, c * h)  # (batch, time, channels*freq)
        
        # Mamba blocks: Temporal modeling
        bottleneck_seq = self.mamba_blocks(bottleneck_seq)
        # bottleneck_seq shape: (batch, time, channels*freq)
        
        # Reshape back to 2D: (batch, time, channels*freq) -> (batch, channels, freq, time)
        bottleneck = bottleneck_seq.reshape(b, w, c, h)  # (batch, time, channels, freq)
        bottleneck = bottleneck.permute(0, 2, 3, 1)  # (batch, channels, freq, time)
        
        # Decoder: Reconstruct enhanced spectrogram
        output = self.decoder(bottleneck, skip_connections)
        # output shape: (batch, out_channels, freq, time)
        
        return output
    
    def forward_with_phase(
        self,
        noisy_mag: torch.Tensor,
        noisy_phase: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also handles phase information.
        
        Args:
            noisy_mag: Noisy magnitude spectrogram (batch, 1, freq, time)
            noisy_phase: Noisy phase spectrogram (batch, 1, freq, time)
            
        Returns:
            Tuple of (enhanced_magnitude, noisy_phase)
        """
        enhanced_mag = self.forward(noisy_mag)
        return enhanced_mag, noisy_phase
    
    def enhance_audio(
        self,
        noisy_stft: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enhance audio using STFT with magnitude estimation and phase reconstruction.
        
        Args:
            noisy_stft: Complex STFT of noisy audio (batch, freq, time)
            
        Returns:
            Complex STFT of enhanced audio (batch, freq, time)
        """
        # Extract magnitude and phase
        noisy_mag = torch.abs(noisy_stft).unsqueeze(1)  # (batch, 1, freq, time)
        noisy_phase = torch.angle(noisy_stft)  # (batch, freq, time)
        
        # Enhance magnitude
        enhanced_mag = self.forward(noisy_mag)  # (batch, 1, freq, time)
        enhanced_mag = enhanced_mag.squeeze(1)  # (batch, freq, time)
        
        # Reconstruct complex STFT with enhanced magnitude and original phase
        enhanced_stft = enhanced_mag * torch.exp(1j * noisy_phase)
        
        return enhanced_stft
    
    @staticmethod
    def load_from_checkpoint(checkpoint_path: str, map_location: str = 'cpu') -> 'MambaSEUNet':
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            map_location: Device to map the model to
            
        Returns:
            Loaded MambaSEUNet model
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Extract model configuration from checkpoint
        if 'model_config' in checkpoint:
            model = MambaSEUNet(**checkpoint['model_config'])
        else:
            # Use default configuration
            model = MambaSEUNet()
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MambaSEUNetConfig:
    """Configuration class for Mamba-SEUNet."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: List[int] = [64, 128, 256, 512],
        num_mamba_blocks: int = 6,
        mamba_state_dim: int = 16,
        mamba_conv_kernel: int = 4,
        mamba_expand_factor: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = 'relu',
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channels = encoder_channels
        self.num_mamba_blocks = num_mamba_blocks
        self.mamba_state_dim = mamba_state_dim
        self.mamba_conv_kernel = mamba_conv_kernel
        self.mamba_expand_factor = mamba_expand_factor
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.activation = activation
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'encoder_channels': self.encoder_channels,
            'num_mamba_blocks': self.num_mamba_blocks,
            'mamba_state_dim': self.mamba_state_dim,
            'mamba_conv_kernel': self.mamba_conv_kernel,
            'mamba_expand_factor': self.mamba_expand_factor,
            'dropout': self.dropout,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'MambaSEUNetConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


if __name__ == "__main__":
    # Test Mamba-SEUNet
    batch_size = 2
    in_channels = 1
    freq_bins = 257  # For n_fft=512
    time_steps = 100
    
    x = torch.randn(batch_size, in_channels, freq_bins, time_steps)
    
    # Create model with default configuration
    model = MambaSEUNet(
        in_channels=1,
        out_channels=1,
        encoder_channels=[64, 128, 256, 512],
        num_mamba_blocks=6,
    )
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {model.get_num_params():,}")
    print(f"Number of trainable parameters: {model.get_num_trainable_params():,}")
    
    # Test with different number of Mamba blocks (for ablation study)
    print("\n--- Ablation Study: Different numbers of Mamba blocks ---")
    for num_blocks in [2, 4, 6, 8, 10]:
        model = MambaSEUNet(num_mamba_blocks=num_blocks)
        num_params = model.get_num_params()
        print(f"Blocks: {num_blocks}, Parameters: {num_params:,}")
    
    # Test phase-aware enhancement
    print("\n--- Phase-aware enhancement test ---")
    model = MambaSEUNet()
    noisy_mag = torch.randn(2, 1, 257, 100)
    noisy_phase = torch.randn(2, 1, 257, 100)
    
    enhanced_mag, phase = model.forward_with_phase(noisy_mag, noisy_phase)
    print(f"Enhanced magnitude shape: {enhanced_mag.shape}")
    print(f"Phase shape: {phase.shape}")
