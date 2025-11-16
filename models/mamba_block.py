"""
Mamba Block Implementation

Bidirectional Mamba State-Space Model block with linear O(n) complexity.
This module implements the TS-Mamba (Time-Series Mamba) block used in Mamba-SEUNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. Using simplified version.")


class SimplifiedMamba(nn.Module):
    """
    Simplified Mamba module for environments where mamba_ssm is not available.
    Uses GRU as a fallback mechanism.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # Projection layers
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # GRU for sequential processing
        self.gru = nn.GRU(
            input_size=self.d_inner,
            hidden_size=self.d_inner,
            num_layers=1,
            batch_first=True,
        )
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, length, d_model)
        Returns:
            output: (batch, length, d_model)
        """
        batch, length, dim = x.shape
        
        # Project and split
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each
        
        # Convolution
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :length]  # (B, d_inner, L)
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = self.activation(x)
        
        # Sequential processing with GRU
        x, _ = self.gru(x)  # (B, L, d_inner)
        
        # Gate and project
        x = x * self.activation(z)
        output = self.out_proj(x)  # (B, L, d_model)
        
        return output


class MambaBlock(nn.Module):
    """
    Bidirectional Mamba block for time-series modeling.
    
    This block processes the input sequence in both forward and backward directions
    and combines the results for enhanced temporal modeling with linear O(n) complexity.
    
    Args:
        dim: Model dimension
        state_dim: State dimension for SSM (default: 16)
        conv_kernel: Convolution kernel size (default: 4)
        expand_factor: Expansion factor for internal dimension (default: 2)
        bidirectional: Whether to use bidirectional processing (default: True)
        dropout: Dropout rate (default: 0.1)
        use_residual: Whether to use residual connection (default: True)
        use_layer_norm: Whether to use layer normalization (default: True)
    """
    
    def __init__(
        self,
        dim: int,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expand_factor: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.conv_kernel = conv_kernel
        self.expand_factor = expand_factor
        self.bidirectional = bidirectional
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Layer normalization
        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(dim)
            if bidirectional:
                self.norm2 = nn.LayerNorm(dim)
        
        # Mamba SSM for forward direction
        if MAMBA_AVAILABLE:
            self.mamba_forward = Mamba(
                d_model=dim,
                d_state=state_dim,
                d_conv=conv_kernel,
                expand=expand_factor,
            )
        else:
            self.mamba_forward = SimplifiedMamba(
                d_model=dim,
                d_state=state_dim,
                d_conv=conv_kernel,
                expand=expand_factor,
            )
        
        # Mamba SSM for backward direction (if bidirectional)
        if bidirectional:
            if MAMBA_AVAILABLE:
                self.mamba_backward = Mamba(
                    d_model=dim,
                    d_state=state_dim,
                    d_conv=conv_kernel,
                    expand=expand_factor,
                )
            else:
                self.mamba_backward = SimplifiedMamba(
                    d_model=dim,
                    d_state=state_dim,
                    d_conv=conv_kernel,
                    expand=expand_factor,
                )
            
            # Fusion layer to combine forward and backward
            self.fusion = nn.Linear(dim * 2, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Mamba block.
        
        Args:
            x: Input tensor of shape (batch, length, dim)
            
        Returns:
            Output tensor of shape (batch, length, dim)
        """
        # Store residual
        residual = x if self.use_residual else None
        
        # Forward direction
        if self.use_layer_norm:
            x_forward = self.norm1(x)
        else:
            x_forward = x
        x_forward = self.mamba_forward(x_forward)
        x_forward = self.dropout(x_forward)
        
        if self.bidirectional:
            # Backward direction
            if self.use_layer_norm:
                x_backward = self.norm2(x)
            else:
                x_backward = x
            
            # Reverse the sequence
            x_backward = torch.flip(x_backward, dims=[1])
            x_backward = self.mamba_backward(x_backward)
            # Reverse back
            x_backward = torch.flip(x_backward, dims=[1])
            x_backward = self.dropout(x_backward)
            
            # Concatenate and fuse
            x_combined = torch.cat([x_forward, x_backward], dim=-1)
            output = self.fusion(x_combined)
        else:
            output = x_forward
        
        # Add residual connection
        if self.use_residual and residual is not None:
            output = output + residual
        
        return output


class StackedMambaBlocks(nn.Module):
    """
    Stack of multiple Mamba blocks.
    
    Args:
        num_blocks: Number of Mamba blocks to stack
        dim: Model dimension
        state_dim: State dimension for SSM
        conv_kernel: Convolution kernel size
        expand_factor: Expansion factor
        bidirectional: Whether to use bidirectional processing
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_blocks: int,
        dim: int,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expand_factor: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        
        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=dim,
                state_dim=state_dim,
                conv_kernel=conv_kernel,
                expand_factor=expand_factor,
                bidirectional=bidirectional,
                dropout=dropout,
                use_residual=True,
                use_layer_norm=True,
            )
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stacked Mamba blocks.
        
        Args:
            x: Input tensor of shape (batch, length, dim)
            
        Returns:
            Output tensor of shape (batch, length, dim)
        """
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    # Test Mamba block
    batch_size = 4
    seq_len = 100
    dim = 256
    
    x = torch.randn(batch_size, seq_len, dim)
    
    # Test single Mamba block
    mamba_block = MambaBlock(
        dim=dim,
        state_dim=16,
        conv_kernel=4,
        expand_factor=2,
        bidirectional=True,
    )
    
    output = mamba_block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in mamba_block.parameters())}")
    
    # Test stacked Mamba blocks
    stacked_blocks = StackedMambaBlocks(
        num_blocks=6,
        dim=dim,
        state_dim=16,
        bidirectional=True,
    )
    
    output = stacked_blocks(x)
    print(f"\nStacked blocks output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in stacked_blocks.parameters())}")
