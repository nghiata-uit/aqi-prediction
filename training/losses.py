"""
Loss Functions for Speech Enhancement

This module implements various loss functions including:
- L1 Loss (MAE)
- L2 Loss (MSE)
- SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
- Perceptual Loss
- Combined Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class L1Loss(nn.Module):
    """L1 Loss (Mean Absolute Error)."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L1 loss.
        
        Args:
            predicted: Predicted magnitude spectrogram
            target: Target magnitude spectrogram
            
        Returns:
            L1 loss value
        """
        return F.l1_loss(predicted, target, reduction=self.reduction)


class L2Loss(nn.Module):
    """L2 Loss (Mean Squared Error)."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L2 loss.
        
        Args:
            predicted: Predicted magnitude spectrogram
            target: Target magnitude spectrogram
            
        Returns:
            L2 loss value
        """
        return F.mse_loss(predicted, target, reduction=self.reduction)


class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) Loss.
    
    SI-SNR is commonly used for speech enhancement and separation tasks.
    """
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SI-SNR loss (negative SI-SNR for minimization).
        
        Args:
            predicted: Predicted audio waveform or spectrogram
            target: Target audio waveform or spectrogram
            
        Returns:
            Negative SI-SNR loss value
        """
        # Flatten to waveform if needed
        if predicted.dim() > 2:
            predicted = predicted.flatten(1)
            target = target.flatten(1)
        
        # Zero-mean normalization
        predicted = predicted - predicted.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        
        # Compute SI-SNR
        s_target = (torch.sum(predicted * target, dim=-1, keepdim=True) /
                    (torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps)) * target
        
        e_noise = predicted - s_target
        
        si_snr = 10 * torch.log10(
            (torch.sum(s_target ** 2, dim=-1) + self.eps) /
            (torch.sum(e_noise ** 2, dim=-1) + self.eps)
        )
        
        # Return negative for minimization
        if self.reduction == 'mean':
            return -si_snr.mean()
        elif self.reduction == 'sum':
            return -si_snr.sum()
        else:
            return -si_snr


class SDRLoss(nn.Module):
    """
    Signal-to-Distortion Ratio (SDR) Loss.
    """
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SDR loss (negative SDR for minimization).
        
        Args:
            predicted: Predicted audio
            target: Target audio
            
        Returns:
            Negative SDR loss value
        """
        if predicted.dim() > 2:
            predicted = predicted.flatten(1)
            target = target.flatten(1)
        
        noise = predicted - target
        
        sdr = 10 * torch.log10(
            (torch.sum(target ** 2, dim=-1) + self.eps) /
            (torch.sum(noise ** 2, dim=-1) + self.eps)
        )
        
        if self.reduction == 'mean':
            return -sdr.mean()
        elif self.reduction == 'sum':
            return -sdr.sum()
        else:
            return -sdr


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using a pre-trained model.
    
    Note: This is a simplified version. For full implementation,
    you would use features from a pre-trained audio model.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        # In a full implementation, load a pre-trained model here
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        For now, this is a placeholder that returns spectral convergence loss.
        
        Args:
            predicted: Predicted spectrogram
            target: Target spectrogram
            
        Returns:
            Perceptual loss value
        """
        # Spectral convergence as a simple perceptual metric
        numerator = torch.norm(target - predicted, p='fro', dim=(-2, -1))
        denominator = torch.norm(target, p='fro', dim=(-2, -1)) + 1e-8
        
        spectral_convergence = numerator / denominator
        
        if self.reduction == 'mean':
            return spectral_convergence.mean()
        elif self.reduction == 'sum':
            return spectral_convergence.sum()
        else:
            return spectral_convergence


class SpectralConvergenceLoss(nn.Module):
    """Spectral Convergence Loss."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute spectral convergence loss.
        
        Args:
            predicted: Predicted spectrogram
            target: Target spectrogram
            
        Returns:
            Spectral convergence loss
        """
        numerator = torch.norm(target - predicted, p='fro', dim=(-2, -1))
        denominator = torch.norm(target, p='fro', dim=(-2, -1)) + 1e-8
        
        loss = numerator / denominator
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined loss with multiple loss functions.
    
    Args:
        losses: Dictionary of loss functions and their weights
        reduction: Reduction method ('mean' or 'sum')
    """
    
    def __init__(
        self,
        losses: Optional[Dict[str, tuple]] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.reduction = reduction
        
        # Default losses if none provided
        if losses is None:
            losses = {
                'l1': (L1Loss(reduction), 1.0),
                'l2': (L2Loss(reduction), 0.5),
                'si_snr': (SISNRLoss(reduction), 0.1),
            }
        
        self.loss_fns = nn.ModuleDict()
        self.loss_weights = {}
        
        for name, (loss_fn, weight) in losses.items():
            self.loss_fns[name] = loss_fn
            self.loss_weights[name] = weight
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predicted: Predicted output
            target: Target output
            
        Returns:
            Dictionary containing individual losses and total loss
        """
        losses = {}
        total_loss = 0.0
        
        for name, loss_fn in self.loss_fns.items():
            weight = self.loss_weights[name]
            loss_value = loss_fn(predicted, target)
            losses[name] = loss_value
            total_loss = total_loss + weight * loss_value
        
        losses['total'] = total_loss
        
        return losses


def create_loss_function(config: Dict) -> nn.Module:
    """
    Create loss function from configuration.
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Loss function module
    """
    loss_type = config.get('type', 'combined')
    reduction = config.get('reduction', 'mean')
    
    if loss_type == 'l1':
        return L1Loss(reduction=reduction)
    elif loss_type == 'l2':
        return L2Loss(reduction=reduction)
    elif loss_type == 'si_snr':
        return SISNRLoss(reduction=reduction)
    elif loss_type == 'perceptual':
        return PerceptualLoss(reduction=reduction)
    elif loss_type == 'combined':
        # Create combined loss from config
        losses = {}
        loss_configs = config.get('losses', {})
        
        for name, loss_config in loss_configs.items():
            weight = loss_config.get('weight', 1.0)
            loss_reduction = loss_config.get('reduction', reduction)
            
            if name == 'l1':
                loss_fn = L1Loss(reduction=loss_reduction)
            elif name == 'l2':
                loss_fn = L2Loss(reduction=loss_reduction)
            elif name == 'si_snr':
                loss_fn = SISNRLoss(reduction=loss_reduction)
            elif name == 'perceptual':
                loss_fn = PerceptualLoss(reduction=loss_reduction)
            else:
                continue
            
            if weight > 0:
                losses[name] = (loss_fn, weight)
        
        return CombinedLoss(losses=losses, reduction=reduction)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    batch_size = 4
    channels = 1
    freq = 257
    time = 100
    
    predicted = torch.randn(batch_size, channels, freq, time)
    target = torch.randn(batch_size, channels, freq, time)
    
    # Test L1 loss
    l1_loss = L1Loss()
    loss_value = l1_loss(predicted, target)
    print(f"L1 Loss: {loss_value.item():.4f}")
    
    # Test L2 loss
    l2_loss = L2Loss()
    loss_value = l2_loss(predicted, target)
    print(f"L2 Loss: {loss_value.item():.4f}")
    
    # Test SI-SNR loss
    si_snr_loss = SISNRLoss()
    loss_value = si_snr_loss(predicted, target)
    print(f"SI-SNR Loss: {loss_value.item():.4f}")
    
    # Test combined loss
    combined_loss = CombinedLoss()
    losses = combined_loss(predicted, target)
    print("\nCombined Loss:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
    
    # Test with waveforms
    print("\n--- Testing with waveforms ---")
    waveform_pred = torch.randn(batch_size, 16000)
    waveform_target = torch.randn(batch_size, 16000)
    
    si_snr_loss = SISNRLoss()
    loss_value = si_snr_loss(waveform_pred, waveform_target)
    print(f"SI-SNR Loss (waveform): {loss_value.item():.4f}")
