"""
Audio Preprocessing for Speech Enhancement

This module implements audio preprocessing including:
- Audio loading and resampling
- STFT/ISTFT computation
- Magnitude and phase extraction
- Normalization
- Speech and noise mixing
"""

import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional, Union
import librosa
import soundfile as sf


class AudioProcessor:
    """
    Audio processor for speech enhancement tasks.
    
    Args:
        sample_rate: Target sample rate (default: 16000)
        n_fft: FFT size (default: 512)
        hop_length: Hop length for STFT (default: 160)
        win_length: Window length for STFT (default: 512)
        window: Window function (default: 'hann')
        center: Whether to center the window (default: True)
        normalized: Whether to normalize the STFT (default: False)
        onesided: Whether to return one-sided STFT (default: True)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 512,
        window: str = 'hann',
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        
        # Create window tensor
        if window == 'hann':
            self.window_tensor = torch.hann_window(win_length)
        elif window == 'hamming':
            self.window_tensor = torch.hamming_window(win_length)
        elif window == 'blackman':
            self.window_tensor = torch.blackman_window(win_length)
        else:
            self.window_tensor = torch.ones(win_length)
    
    def stft(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            audio: Audio waveform tensor of shape (batch, samples) or (samples,)
            
        Returns:
            Complex STFT tensor of shape (batch, freq, time) or (freq, time)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Compute STFT
        stft_result = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window_tensor.to(audio.device),
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )
        
        if squeeze_output:
            stft_result = stft_result.squeeze(0)
        
        return stft_result
    
    def istft(self, stft: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """
        Compute Inverse Short-Time Fourier Transform.
        
        Args:
            stft: Complex STFT tensor of shape (batch, freq, time) or (freq, time)
            length: Target length of the output waveform
            
        Returns:
            Audio waveform tensor of shape (batch, samples) or (samples,)
        """
        if stft.dim() == 2:
            stft = stft.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Compute ISTFT
        audio = torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window_tensor.to(stft.device),
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=length,
            return_complex=False,
        )
        
        if squeeze_output:
            audio = audio.squeeze(0)
        
        return audio
    
    def get_magnitude_phase(self, stft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract magnitude and phase from complex STFT.
        
        Args:
            stft: Complex STFT tensor
            
        Returns:
            Tuple of (magnitude, phase) tensors
        """
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        return magnitude, phase
    
    def reconstruct_from_magnitude_phase(
        self,
        magnitude: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct complex STFT from magnitude and phase.
        
        Args:
            magnitude: Magnitude tensor
            phase: Phase tensor
            
        Returns:
            Complex STFT tensor
        """
        return magnitude * torch.exp(1j * phase)
    
    def normalize_magnitude(
        self,
        magnitude: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize magnitude spectrogram.
        
        Args:
            magnitude: Magnitude tensor
            eps: Small value to avoid division by zero
            
        Returns:
            Tuple of (normalized_magnitude, mean, std)
        """
        mean = magnitude.mean()
        std = magnitude.std()
        normalized = (magnitude - mean) / (std + eps)
        return normalized, mean, std
    
    def denormalize_magnitude(
        self,
        normalized: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denormalize magnitude spectrogram.
        
        Args:
            normalized: Normalized magnitude tensor
            mean: Mean used for normalization
            std: Standard deviation used for normalization
            
        Returns:
            Denormalized magnitude tensor
        """
        return normalized * std + mean


def load_audio(
    path: str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample if necessary.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate
        mono: Whether to convert to mono
        
    Returns:
        Tuple of (audio waveform, sample rate)
    """
    try:
        # Try loading with torchaudio
        audio, sr = torchaudio.load(path)
    except Exception:
        # Fallback to librosa
        audio_np, sr = librosa.load(path, sr=None, mono=False)
        audio = torch.from_numpy(audio_np)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
    
    # Convert to mono if needed
    if mono and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)
    
    return audio.squeeze(0), sample_rate


def save_audio(
    path: str,
    audio: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 16000,
):
    """
    Save audio to file.
    
    Args:
        path: Output file path
        audio: Audio waveform
        sample_rate: Sample rate
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    
    sf.write(path, audio, sample_rate)


def compute_stft(
    audio: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 512,
    window: str = 'hann',
) -> torch.Tensor:
    """
    Compute STFT of audio waveform.
    
    Args:
        audio: Audio waveform tensor
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        window: Window function
        
    Returns:
        Complex STFT tensor
    """
    processor = AudioProcessor(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )
    return processor.stft(audio)


def compute_istft(
    stft: torch.Tensor,
    hop_length: int = 160,
    win_length: int = 512,
    window: str = 'hann',
    length: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute inverse STFT.
    
    Args:
        stft: Complex STFT tensor
        hop_length: Hop length
        win_length: Window length
        window: Window function
        length: Target length
        
    Returns:
        Audio waveform tensor
    """
    processor = AudioProcessor(
        n_fft=stft.shape[-2] * 2 - 2 if stft.is_complex() else stft.shape[-2] - 1,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )
    return processor.istft(stft, length=length)


def mix_speech_noise(
    speech: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """
    Mix speech and noise at a specified SNR.
    
    Args:
        speech: Clean speech waveform
        noise: Noise waveform
        snr_db: Signal-to-Noise Ratio in dB
        
    Returns:
        Noisy speech waveform
    """
    # Ensure same length
    if speech.shape[-1] > noise.shape[-1]:
        # Repeat noise if too short
        num_repeats = (speech.shape[-1] // noise.shape[-1]) + 1
        noise = noise.repeat(num_repeats)
    
    noise = noise[..., :speech.shape[-1]]
    
    # Compute speech and noise power
    speech_power = torch.mean(speech ** 2)
    noise_power = torch.mean(noise ** 2)
    
    # Compute noise scaling factor
    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(speech_power / (snr_linear * noise_power + 1e-8))
    
    # Mix
    noisy = speech + scale * noise
    
    return noisy


def compute_snr(
    clean: torch.Tensor,
    noisy: torch.Tensor,
) -> float:
    """
    Compute Signal-to-Noise Ratio.
    
    Args:
        clean: Clean signal
        noisy: Noisy signal
        
    Returns:
        SNR in dB
    """
    noise = noisy - clean
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean(noise ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    return snr.item()


def normalize_audio(
    audio: torch.Tensor,
    target_level: float = -25.0,
) -> torch.Tensor:
    """
    Normalize audio to target level in dB.
    
    Args:
        audio: Audio waveform
        target_level: Target level in dB
        
    Returns:
        Normalized audio
    """
    rms = torch.sqrt(torch.mean(audio ** 2))
    current_level = 20 * torch.log10(rms + 1e-8)
    gain = 10 ** ((target_level - current_level) / 20)
    return audio * gain


def trim_silence(
    audio: torch.Tensor,
    threshold: float = 0.01,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Trim silence from audio.
    
    Args:
        audio: Audio waveform
        threshold: Energy threshold for silence detection
        frame_length: Frame length for energy computation
        hop_length: Hop length for energy computation
        
    Returns:
        Trimmed audio
    """
    if isinstance(audio, torch.Tensor):
        audio_np = audio.cpu().numpy()
    else:
        audio_np = audio
    
    # Compute energy
    energy = librosa.feature.rms(
        y=audio_np,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    
    # Find non-silent frames
    non_silent = energy > threshold
    
    if non_silent.sum() == 0:
        return audio
    
    # Find start and end indices
    frames = np.where(non_silent)[0]
    start = frames[0] * hop_length
    end = min((frames[-1] + 1) * hop_length, len(audio_np))
    
    if isinstance(audio, torch.Tensor):
        return audio[start:end]
    else:
        return audio_np[start:end]


if __name__ == "__main__":
    # Test audio processing
    print("Testing audio preprocessing...")
    
    # Create processor
    processor = AudioProcessor(sample_rate=16000)
    
    # Create test audio
    duration = 4.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    audio = torch.randn(samples)
    
    print(f"Audio shape: {audio.shape}")
    
    # Compute STFT
    stft = processor.stft(audio)
    print(f"STFT shape: {stft.shape}")
    
    # Extract magnitude and phase
    magnitude, phase = processor.get_magnitude_phase(stft)
    print(f"Magnitude shape: {magnitude.shape}")
    print(f"Phase shape: {phase.shape}")
    
    # Normalize magnitude
    normalized, mean, std = processor.normalize_magnitude(magnitude)
    print(f"Normalized magnitude - mean: {normalized.mean():.4f}, std: {normalized.std():.4f}")
    
    # Reconstruct audio
    reconstructed_stft = processor.reconstruct_from_magnitude_phase(magnitude, phase)
    reconstructed_audio = processor.istft(reconstructed_stft, length=audio.shape[0])
    print(f"Reconstructed audio shape: {reconstructed_audio.shape}")
    
    # Check reconstruction error
    error = torch.mean(torch.abs(audio - reconstructed_audio))
    print(f"Reconstruction error: {error.item():.6f}")
    
    # Test noise mixing
    noise = torch.randn_like(audio)
    snr_db = 10.0
    noisy = mix_speech_noise(audio, noise, snr_db)
    print(f"Noisy audio shape: {noisy.shape}")
    
    # Compute SNR
    computed_snr = compute_snr(audio, noisy)
    print(f"Target SNR: {snr_db:.2f} dB, Computed SNR: {computed_snr:.2f} dB")
