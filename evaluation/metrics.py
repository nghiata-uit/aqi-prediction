"""
Evaluation Metrics for Speech Enhancement

This module implements evaluation metrics including:
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
- Parameter count and FLOPs computation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Union
import warnings

# Try to import optional dependencies
try:
    from pesq import pesq as pesq_score
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    warnings.warn("pesq not available. Install with: pip install pesq")

try:
    from pystoi import stoi as stoi_score
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    warnings.warn("pystoi not available. Install with: pip install pystoi")


def compute_pesq(
    clean: Union[torch.Tensor, np.ndarray],
    enhanced: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 16000,
    mode: str = 'wb',
) -> float:
    """
    Compute PESQ score.
    
    Args:
        clean: Clean reference speech
        enhanced: Enhanced speech
        sample_rate: Sample rate (8000 for nb, 16000 for wb)
        mode: PESQ mode ('nb' for narrowband or 'wb' for wideband)
        
    Returns:
        PESQ score (1.0-4.5 for wb, -0.5-4.5 for nb)
    """
    if not PESQ_AVAILABLE:
        warnings.warn("PESQ not available, returning 0.0")
        return 0.0
    
    # Convert to numpy if needed
    if isinstance(clean, torch.Tensor):
        clean = clean.cpu().numpy()
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.cpu().numpy()
    
    # Ensure 1D
    if clean.ndim > 1:
        clean = clean.flatten()
    if enhanced.ndim > 1:
        enhanced = enhanced.flatten()
    
    # Ensure same length
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]
    
    try:
        score = pesq_score(sample_rate, clean, enhanced, mode)
        return float(score)
    except Exception as e:
        warnings.warn(f"PESQ computation failed: {e}")
        return 0.0


def compute_stoi(
    clean: Union[torch.Tensor, np.ndarray],
    enhanced: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 16000,
    extended: bool = False,
) -> float:
    """
    Compute STOI score.
    
    Args:
        clean: Clean reference speech
        enhanced: Enhanced speech
        sample_rate: Sample rate
        extended: Whether to use extended STOI
        
    Returns:
        STOI score (0.0-1.0)
    """
    if not STOI_AVAILABLE:
        warnings.warn("STOI not available, returning 0.0")
        return 0.0
    
    # Convert to numpy if needed
    if isinstance(clean, torch.Tensor):
        clean = clean.cpu().numpy()
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.cpu().numpy()
    
    # Ensure 1D
    if clean.ndim > 1:
        clean = clean.flatten()
    if enhanced.ndim > 1:
        enhanced = enhanced.flatten()
    
    # Ensure same length
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]
    
    try:
        score = stoi_score(clean, enhanced, sample_rate, extended=extended)
        return float(score)
    except Exception as e:
        warnings.warn(f"STOI computation failed: {e}")
        return 0.0


def compute_si_sdr(
    clean: Union[torch.Tensor, np.ndarray],
    enhanced: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-8,
) -> float:
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio.
    
    Args:
        clean: Clean reference speech
        enhanced: Enhanced speech
        eps: Small value to avoid division by zero
        
    Returns:
        SI-SDR score in dB (higher is better)
    """
    # Convert to torch if needed
    if isinstance(clean, np.ndarray):
        clean = torch.from_numpy(clean)
    if isinstance(enhanced, np.ndarray):
        enhanced = torch.from_numpy(enhanced)
    
    # Ensure 1D
    if clean.ndim > 1:
        clean = clean.flatten()
    if enhanced.ndim > 1:
        enhanced = enhanced.flatten()
    
    # Ensure same length
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]
    
    # Zero-mean normalization
    clean = clean - clean.mean()
    enhanced = enhanced - enhanced.mean()
    
    # Compute SI-SDR
    s_target = (torch.sum(enhanced * clean) / (torch.sum(clean ** 2) + eps)) * clean
    e_noise = enhanced - s_target
    
    si_sdr = 10 * torch.log10(
        (torch.sum(s_target ** 2) + eps) / (torch.sum(e_noise ** 2) + eps)
    )
    
    return float(si_sdr.item())


def compute_snr(
    clean: Union[torch.Tensor, np.ndarray],
    noisy: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-8,
) -> float:
    """
    Compute Signal-to-Noise Ratio.
    
    Args:
        clean: Clean signal
        noisy: Noisy signal
        eps: Small value to avoid division by zero
        
    Returns:
        SNR in dB
    """
    if isinstance(clean, np.ndarray):
        clean = torch.from_numpy(clean)
    if isinstance(noisy, np.ndarray):
        noisy = torch.from_numpy(noisy)
    
    if clean.ndim > 1:
        clean = clean.flatten()
    if noisy.ndim > 1:
        noisy = noisy.flatten()
    
    min_len = min(len(clean), len(noisy))
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    
    noise = noisy - clean
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean(noise ** 2)
    
    snr = 10 * torch.log10(signal_power / (noise_power + eps))
    return float(snr.item())


def compute_all_metrics(
    clean: Union[torch.Tensor, np.ndarray],
    enhanced: Union[torch.Tensor, np.ndarray],
    noisy: Optional[Union[torch.Tensor, np.ndarray]] = None,
    sample_rate: int = 16000,
) -> Dict[str, float]:
    """
    Compute all available metrics.
    
    Args:
        clean: Clean reference speech
        enhanced: Enhanced speech
        noisy: Noisy speech (optional, for computing improvements)
        sample_rate: Sample rate
        
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    # PESQ
    try:
        metrics['pesq'] = compute_pesq(clean, enhanced, sample_rate)
    except Exception as e:
        warnings.warn(f"PESQ computation failed: {e}")
        metrics['pesq'] = 0.0
    
    # STOI
    try:
        metrics['stoi'] = compute_stoi(clean, enhanced, sample_rate)
    except Exception as e:
        warnings.warn(f"STOI computation failed: {e}")
        metrics['stoi'] = 0.0
    
    # SI-SDR
    try:
        metrics['si_sdr'] = compute_si_sdr(clean, enhanced)
    except Exception as e:
        warnings.warn(f"SI-SDR computation failed: {e}")
        metrics['si_sdr'] = 0.0
    
    # If noisy audio is provided, compute improvements
    if noisy is not None:
        try:
            noisy_pesq = compute_pesq(clean, noisy, sample_rate)
            metrics['pesq_improvement'] = metrics['pesq'] - noisy_pesq
        except:
            pass
        
        try:
            noisy_stoi = compute_stoi(clean, noisy, sample_rate)
            metrics['stoi_improvement'] = metrics['stoi'] - noisy_stoi
        except:
            pass
        
        try:
            noisy_si_sdr = compute_si_sdr(clean, noisy)
            metrics['si_sdr_improvement'] = metrics['si_sdr'] - noisy_si_sdr
        except:
            pass
    
    return metrics


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
    }


def compute_flops(
    model: nn.Module,
    input_shape: tuple = (1, 1, 257, 100),
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Estimate FLOPs for the model.
    
    This is a simplified estimation. For accurate FLOPs counting,
    use tools like fvcore or ptflops.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run on
        
    Returns:
        Dictionary with FLOPs estimates
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        
        model = model.to(device)
        inputs = torch.randn(*input_shape).to(device)
        
        flops = FlopCountAnalysis(model, inputs)
        total_flops = flops.total()
        
        return {
            'total_flops': total_flops,
            'gflops': total_flops / 1e9,
            'mflops': total_flops / 1e6,
        }
    except ImportError:
        warnings.warn("fvcore not available. Install with: pip install fvcore")
        return {'total_flops': 0, 'gflops': 0.0, 'mflops': 0.0}
    except Exception as e:
        warnings.warn(f"FLOPs computation failed: {e}")
        return {'total_flops': 0, 'gflops': 0.0, 'mflops': 0.0}


def compute_rtf(
    model: nn.Module,
    audio_length_seconds: float = 1.0,
    sample_rate: int = 16000,
    n_runs: int = 10,
    device: str = 'cuda',
) -> float:
    """
    Compute Real-Time Factor (RTF).
    
    RTF < 1 means the model can process audio faster than real-time.
    
    Args:
        model: PyTorch model
        audio_length_seconds: Length of audio to process
        sample_rate: Sample rate
        n_runs: Number of runs for averaging
        device: Device to run on
        
    Returns:
        RTF value
    """
    import time
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    n_samples = int(audio_length_seconds * sample_rate)
    n_fft = 512
    hop_length = 160
    n_frames = (n_samples - n_fft) // hop_length + 1
    n_freqs = n_fft // 2 + 1
    
    dummy_input = torch.randn(1, 1, n_freqs, n_frames).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    
    # Measure time
    if device == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start_time = time.time()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    rtf = avg_time / audio_length_seconds
    
    return rtf


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")
    
    # Create dummy audio
    sample_rate = 16000
    duration = 2.0
    n_samples = int(duration * sample_rate)
    
    clean = np.random.randn(n_samples) * 0.1
    noise = np.random.randn(n_samples) * 0.05
    noisy = clean + noise
    enhanced = clean + noise * 0.3  # Simulated enhancement
    
    # Compute metrics
    metrics = compute_all_metrics(clean, enhanced, noisy, sample_rate)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test parameter counting
    print("\n--- Testing parameter counting ---")
    from models import MambaSEUNet
    
    model = MambaSEUNet(num_mamba_blocks=6)
    params = count_parameters(model)
    
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
