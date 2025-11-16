"""
Audio Augmentation for Speech Enhancement

This module implements data augmentation techniques including:
- Random noise addition
- Time stretching
- Pitch shifting
- SpecAugment for spectrograms
"""

import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple
import random


class AudioAugmentor:
    """
    Audio augmentation for speech enhancement training.
    
    Args:
        noise_prob: Probability of adding noise (default: 0.5)
        time_stretch_prob: Probability of time stretching (default: 0.3)
        pitch_shift_prob: Probability of pitch shifting (default: 0.3)
        spec_augment_prob: Probability of SpecAugment (default: 0.5)
        snr_range: SNR range for noise addition in dB (default: [-5, 20])
        time_stretch_range: Time stretching rate range (default: [0.9, 1.1])
        pitch_shift_range: Pitch shift range in semitones (default: [-2, 2])
        freq_mask_param: Maximum frequency mask size for SpecAugment (default: 15)
        time_mask_param: Maximum time mask size for SpecAugment (default: 25)
        n_freq_masks: Number of frequency masks (default: 2)
        n_time_masks: Number of time masks (default: 2)
    """
    
    def __init__(
        self,
        noise_prob: float = 0.5,
        time_stretch_prob: float = 0.3,
        pitch_shift_prob: float = 0.3,
        spec_augment_prob: float = 0.5,
        snr_range: Tuple[float, float] = (-5, 20),
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        pitch_shift_range: Tuple[int, int] = (-2, 2),
        freq_mask_param: int = 15,
        time_mask_param: int = 25,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        self.noise_prob = noise_prob
        self.time_stretch_prob = time_stretch_prob
        self.pitch_shift_prob = pitch_shift_prob
        self.spec_augment_prob = spec_augment_prob
        self.snr_range = snr_range
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def __call__(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        Apply random augmentations to audio.
        
        Args:
            audio: Audio waveform tensor
            sample_rate: Sample rate
            
        Returns:
            Augmented audio waveform
        """
        # Time stretch
        if random.random() < self.time_stretch_prob:
            audio = self.time_stretch(audio, sample_rate)
        
        # Pitch shift
        if random.random() < self.pitch_shift_prob:
            audio = self.pitch_shift(audio, sample_rate)
        
        return audio
    
    def time_stretch(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        Apply time stretching to audio.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Time-stretched audio
        """
        rate = random.uniform(*self.time_stretch_range)
        
        # Use torchaudio's speed perturbation
        effects = [
            ["speed", str(rate)],
            ["rate", str(sample_rate)],
        ]
        
        try:
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio, sample_rate, effects
            )
            
            if squeeze:
                augmented = augmented.squeeze(0)
            
            return augmented
        except Exception:
            # Fallback to simple linear interpolation
            return self._time_stretch_fallback(audio, rate)
    
    def _time_stretch_fallback(
        self,
        audio: torch.Tensor,
        rate: float,
    ) -> torch.Tensor:
        """Fallback time stretch using linear interpolation."""
        original_length = audio.shape[-1]
        new_length = int(original_length / rate)
        
        # Linear interpolation
        indices = torch.linspace(0, original_length - 1, new_length)
        indices_floor = indices.long()
        indices_ceil = torch.clamp(indices_floor + 1, max=original_length - 1)
        
        weight = indices - indices_floor.float()
        
        stretched = (1 - weight) * audio[indices_floor] + weight * audio[indices_ceil]
        
        return stretched
    
    def pitch_shift(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        Apply pitch shifting to audio.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Pitch-shifted audio
        """
        n_steps = random.randint(*self.pitch_shift_range)
        
        if n_steps == 0:
            return audio
        
        # Use torchaudio's pitch shift
        try:
            effects = [
                ["pitch", str(n_steps * 100)],  # Convert semitones to cents
                ["rate", str(sample_rate)],
            ]
            
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio, sample_rate, effects
            )
            
            if squeeze:
                augmented = augmented.squeeze(0)
            
            return augmented
        except Exception:
            # Fallback: use time stretching approximation
            rate = 2 ** (n_steps / 12)
            return self._time_stretch_fallback(audio, rate)
    
    def add_noise(
        self,
        audio: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        snr_db: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Add noise to audio at specified SNR.
        
        Args:
            audio: Clean audio waveform
            noise: Noise waveform (if None, use Gaussian noise)
            snr_db: SNR in dB (if None, randomly sample from range)
            
        Returns:
            Noisy audio
        """
        if snr_db is None:
            snr_db = random.uniform(*self.snr_range)
        
        if noise is None:
            noise = torch.randn_like(audio)
        
        # Ensure same length
        if audio.shape[-1] > noise.shape[-1]:
            num_repeats = (audio.shape[-1] // noise.shape[-1]) + 1
            noise = noise.repeat(num_repeats)
        
        noise = noise[..., :audio.shape[-1]]
        
        # Compute power
        speech_power = torch.mean(audio ** 2)
        noise_power = torch.mean(noise ** 2)
        
        # Scale noise
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(speech_power / (snr_linear * noise_power + 1e-8))
        
        return audio + scale * noise
    
    def spec_augment(
        self,
        spectrogram: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Spectrogram tensor of shape (..., freq, time)
            
        Returns:
            Augmented spectrogram
        """
        spec = spectrogram.clone()
        
        # Get dimensions
        *batch_dims, freq_size, time_size = spec.shape
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, freq_size - f))
            spec[..., f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(0, time_size - t))
            spec[..., :, t0:t0+t] = 0
        
        return spec


class RandomNoiseAdder:
    """
    Add random noise to audio with various noise types.
    
    Args:
        noise_types: List of noise types to use
        snr_range: SNR range in dB
        prob: Probability of adding noise
    """
    
    def __init__(
        self,
        noise_types: list = ['gaussian', 'uniform'],
        snr_range: Tuple[float, float] = (-5, 20),
        prob: float = 0.5,
    ):
        self.noise_types = noise_types
        self.snr_range = snr_range
        self.prob = prob
    
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Add random noise to audio."""
        if random.random() > self.prob:
            return audio
        
        noise_type = random.choice(self.noise_types)
        snr_db = random.uniform(*self.snr_range)
        
        if noise_type == 'gaussian':
            noise = torch.randn_like(audio)
        elif noise_type == 'uniform':
            noise = torch.rand_like(audio) * 2 - 1
        else:
            noise = torch.randn_like(audio)
        
        # Mix at specified SNR
        speech_power = torch.mean(audio ** 2)
        noise_power = torch.mean(noise ** 2)
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(speech_power / (snr_linear * noise_power + 1e-8))
        
        return audio + scale * noise


class RandomAmplitude:
    """
    Randomly adjust amplitude of audio.
    
    Args:
        gain_range: Range of gain adjustment in dB
        prob: Probability of applying
    """
    
    def __init__(
        self,
        gain_range: Tuple[float, float] = (-6, 6),
        prob: float = 0.5,
    ):
        self.gain_range = gain_range
        self.prob = prob
    
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random amplitude adjustment."""
        if random.random() > self.prob:
            return audio
        
        gain_db = random.uniform(*self.gain_range)
        gain_linear = 10 ** (gain_db / 20)
        
        return audio * gain_linear


class Compose:
    """
    Compose multiple augmentations.
    
    Args:
        transforms: List of augmentation transforms
    """
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            if hasattr(transform, '__call__'):
                # Check if transform requires sample_rate
                import inspect
                sig = inspect.signature(transform.__call__)
                if 'sample_rate' in sig.parameters:
                    audio = transform(audio, sample_rate)
                else:
                    audio = transform(audio)
        return audio


if __name__ == "__main__":
    # Test augmentation
    print("Testing audio augmentation...")
    
    # Create test audio
    duration = 2.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    audio = torch.randn(samples)
    
    print(f"Original audio shape: {audio.shape}")
    
    # Create augmentor
    augmentor = AudioAugmentor(
        noise_prob=0.5,
        time_stretch_prob=0.3,
        pitch_shift_prob=0.3,
        spec_augment_prob=0.5,
    )
    
    # Test time stretch
    print("\n--- Time Stretch Test ---")
    stretched = augmentor.time_stretch(audio, sample_rate)
    print(f"Stretched audio shape: {stretched.shape}")
    
    # Test noise addition
    print("\n--- Noise Addition Test ---")
    noisy = augmentor.add_noise(audio, snr_db=10.0)
    print(f"Noisy audio shape: {noisy.shape}")
    signal_power = torch.mean(audio ** 2)
    noisy_power = torch.mean(noisy ** 2)
    print(f"Signal power: {signal_power:.4f}, Noisy power: {noisy_power:.4f}")
    
    # Test SpecAugment
    print("\n--- SpecAugment Test ---")
    spectrogram = torch.randn(257, 100)  # freq x time
    augmented_spec = augmentor.spec_augment(spectrogram)
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Augmented spectrogram shape: {augmented_spec.shape}")
    print(f"Fraction of zeros (masked): {(augmented_spec == 0).float().mean():.3f}")
    
    # Test composed augmentations
    print("\n--- Composed Augmentations Test ---")
    composed = Compose([
        RandomAmplitude(gain_range=(-3, 3), prob=1.0),
        RandomNoiseAdder(snr_range=[5, 15], prob=1.0),
    ])
    augmented = composed(audio, sample_rate)
    print(f"Composed augmented audio shape: {augmented.shape}")
