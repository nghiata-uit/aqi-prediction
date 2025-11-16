"""
Dataset Loaders for Speech Enhancement

This module implements PyTorch datasets for:
- VCTK+DEMAND dataset
- VoiceBank+DEMAND dataset
- Generic speech enhancement datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Dict
import random
import numpy as np

from .preprocessing import AudioProcessor, load_audio, mix_speech_noise
from .augmentation import AudioAugmentor


class SpeechEnhancementDataset(Dataset):
    """
    Generic speech enhancement dataset.
    
    Args:
        clean_dir: Directory containing clean speech files
        noisy_dir: Directory containing noisy speech files (optional)
        noise_dir: Directory containing noise files for mixing (optional)
        sample_rate: Target sample rate
        segment_length: Length of audio segments in seconds (None for full length)
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        augmentor: Audio augmentor (optional)
        transform: Additional transform function (optional)
        snr_range: SNR range for mixing if noise_dir is provided
        file_ext: Audio file extension
    """
    
    def __init__(
        self,
        clean_dir: str,
        noisy_dir: Optional[str] = None,
        noise_dir: Optional[str] = None,
        sample_rate: int = 16000,
        segment_length: Optional[float] = 4.0,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 512,
        augmentor: Optional[AudioAugmentor] = None,
        transform: Optional[Callable] = None,
        snr_range: Tuple[float, float] = (-5, 20),
        file_ext: str = '.wav',
    ):
        super().__init__()
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir) if noisy_dir else None
        self.noise_dir = Path(noise_dir) if noise_dir else None
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sample_rate) if segment_length else None
        self.augmentor = augmentor
        self.transform = transform
        self.snr_range = snr_range
        self.file_ext = file_ext
        
        # Audio processor
        self.processor = AudioProcessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        
        # Get file lists
        self.clean_files = sorted([
            f for f in self.clean_dir.rglob(f'*{file_ext}')
        ])
        
        if self.noisy_dir:
            self.noisy_files = sorted([
                f for f in self.noisy_dir.rglob(f'*{file_ext}')
            ])
            assert len(self.clean_files) == len(self.noisy_files), \
                "Number of clean and noisy files must match"
        else:
            self.noisy_files = None
        
        if self.noise_dir:
            self.noise_files = sorted([
                f for f in self.noise_dir.rglob(f'*{file_ext}')
            ])
        else:
            self.noise_files = None
        
        if len(self.clean_files) == 0:
            raise ValueError(f"No audio files found in {clean_dir}")
    
    def __len__(self) -> int:
        return len(self.clean_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary with keys:
                - clean_audio: Clean audio waveform
                - noisy_audio: Noisy audio waveform
                - clean_magnitude: Clean magnitude spectrogram
                - noisy_magnitude: Noisy magnitude spectrogram
                - phase: Phase spectrogram
                - file_name: File name
        """
        # Load clean audio
        clean_audio, _ = load_audio(
            str(self.clean_files[idx]),
            sample_rate=self.sample_rate,
        )
        
        # Get or create noisy audio
        if self.noisy_files:
            # Load pre-mixed noisy audio
            noisy_audio, _ = load_audio(
                str(self.noisy_files[idx]),
                sample_rate=self.sample_rate,
            )
        elif self.noise_files:
            # Mix clean audio with random noise
            noise_idx = random.randint(0, len(self.noise_files) - 1)
            noise_audio, _ = load_audio(
                str(self.noise_files[noise_idx]),
                sample_rate=self.sample_rate,
            )
            snr_db = random.uniform(*self.snr_range)
            noisy_audio = mix_speech_noise(clean_audio, noise_audio, snr_db)
        else:
            # Use clean audio as noisy (for testing without noise)
            noisy_audio = clean_audio
        
        # Ensure same length
        min_length = min(clean_audio.shape[0], noisy_audio.shape[0])
        clean_audio = clean_audio[:min_length]
        noisy_audio = noisy_audio[:min_length]
        
        # Segment if needed
        if self.segment_samples and clean_audio.shape[0] > self.segment_samples:
            start = random.randint(0, clean_audio.shape[0] - self.segment_samples)
            clean_audio = clean_audio[start:start + self.segment_samples]
            noisy_audio = noisy_audio[start:start + self.segment_samples]
        elif self.segment_samples and clean_audio.shape[0] < self.segment_samples:
            # Pad if too short
            pad_length = self.segment_samples - clean_audio.shape[0]
            clean_audio = torch.nn.functional.pad(clean_audio, (0, pad_length))
            noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad_length))
        
        # Apply augmentation
        if self.augmentor:
            noisy_audio = self.augmentor(noisy_audio, self.sample_rate)
        
        # Compute STFT
        clean_stft = self.processor.stft(clean_audio)
        noisy_stft = self.processor.stft(noisy_audio)
        
        # Extract magnitude and phase
        clean_magnitude, _ = self.processor.get_magnitude_phase(clean_stft)
        noisy_magnitude, phase = self.processor.get_magnitude_phase(noisy_stft)
        
        # Add channel dimension
        clean_magnitude = clean_magnitude.unsqueeze(0)  # (1, freq, time)
        noisy_magnitude = noisy_magnitude.unsqueeze(0)  # (1, freq, time)
        
        # Apply additional transform
        if self.transform:
            clean_magnitude = self.transform(clean_magnitude)
            noisy_magnitude = self.transform(noisy_magnitude)
        
        return {
            'clean_audio': clean_audio,
            'noisy_audio': noisy_audio,
            'clean_magnitude': clean_magnitude,
            'noisy_magnitude': noisy_magnitude,
            'phase': phase,
            'file_name': self.clean_files[idx].name,
        }


class VCTKDEMANDDataset(SpeechEnhancementDataset):
    """
    VCTK+DEMAND dataset for speech enhancement.
    
    Args:
        root_dir: Root directory containing VCTK+DEMAND data
        split: Dataset split ('train', 'val', 'test')
        **kwargs: Additional arguments for SpeechEnhancementDataset
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        **kwargs
    ):
        root_path = Path(root_dir)
        
        if split == 'train':
            clean_dir = root_path / 'clean_trainset_28spk_wav'
            noisy_dir = root_path / 'noisy_trainset_28spk_wav'
        elif split == 'test':
            clean_dir = root_path / 'clean_testset_wav'
            noisy_dir = root_path / 'noisy_testset_wav'
        elif split == 'val':
            # Use a subset of training data for validation
            clean_dir = root_path / 'clean_trainset_28spk_wav'
            noisy_dir = root_path / 'noisy_trainset_28spk_wav'
        else:
            raise ValueError(f"Invalid split: {split}")
        
        super().__init__(
            clean_dir=str(clean_dir),
            noisy_dir=str(noisy_dir),
            **kwargs
        )
        
        # For validation, use a subset
        if split == 'val':
            val_size = int(0.1 * len(self.clean_files))
            self.clean_files = self.clean_files[:val_size]
            if self.noisy_files:
                self.noisy_files = self.noisy_files[:val_size]


class VoiceBankDEMANDDataset(SpeechEnhancementDataset):
    """
    VoiceBank+DEMAND dataset for speech enhancement.
    
    Args:
        root_dir: Root directory containing VoiceBank+DEMAND data
        split: Dataset split ('train', 'val', 'test')
        **kwargs: Additional arguments for SpeechEnhancementDataset
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        **kwargs
    ):
        root_path = Path(root_dir)
        
        if split == 'train':
            clean_dir = root_path / 'clean_trainset_wav'
            noisy_dir = root_path / 'noisy_trainset_wav'
        elif split == 'test':
            clean_dir = root_path / 'clean_testset_wav'
            noisy_dir = root_path / 'noisy_testset_wav'
        elif split == 'val':
            # Use a subset of training data for validation
            clean_dir = root_path / 'clean_trainset_wav'
            noisy_dir = root_path / 'noisy_trainset_wav'
        else:
            raise ValueError(f"Invalid split: {split}")
        
        super().__init__(
            clean_dir=str(clean_dir),
            noisy_dir=str(noisy_dir),
            **kwargs
        )
        
        # For validation, use a subset
        if split == 'val':
            val_size = int(0.1 * len(self.clean_files))
            self.clean_files = self.clean_files[:val_size]
            if self.noisy_files:
                self.noisy_files = self.noisy_files[:val_size]


def create_dataloaders(
    dataset_name: str,
    root_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    sample_rate: int = 16000,
    segment_length: float = 4.0,
    use_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        dataset_name: Name of dataset ('vctk_demand' or 'voicebank_demand')
        root_dir: Root directory of dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        sample_rate: Target sample rate
        segment_length: Segment length in seconds
        use_augmentation: Whether to use data augmentation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create augmentor for training
    augmentor = AudioAugmentor() if use_augmentation else None
    
    # Select dataset class
    if dataset_name.lower() == 'vctk_demand':
        dataset_class = VCTKDEMANDDataset
    elif dataset_name.lower() == 'voicebank_demand':
        dataset_class = VoiceBankDEMANDDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create datasets
    train_dataset = dataset_class(
        root_dir=root_dir,
        split='train',
        sample_rate=sample_rate,
        segment_length=segment_length,
        augmentor=augmentor,
    )
    
    val_dataset = dataset_class(
        root_dir=root_dir,
        split='val',
        sample_rate=sample_rate,
        segment_length=segment_length,
        augmentor=None,  # No augmentation for validation
    )
    
    test_dataset = dataset_class(
        root_dir=root_dir,
        split='test',
        sample_rate=sample_rate,
        segment_length=None,  # Full length for testing
        augmentor=None,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one file at a time for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing speech enhancement dataset...")
    
    # Create a dummy dataset structure for testing
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    clean_dir = Path(temp_dir) / 'clean'
    noisy_dir = Path(temp_dir) / 'noisy'
    clean_dir.mkdir()
    noisy_dir.mkdir()
    
    # Create dummy audio files
    import soundfile as sf
    
    for i in range(5):
        # Create dummy clean audio
        audio = np.random.randn(16000)  # 1 second at 16kHz
        sf.write(clean_dir / f'audio_{i}.wav', audio, 16000)
        
        # Create dummy noisy audio
        noisy = audio + 0.1 * np.random.randn(16000)
        sf.write(noisy_dir / f'audio_{i}.wav', noisy, 16000)
    
    # Create dataset
    dataset = SpeechEnhancementDataset(
        clean_dir=str(clean_dir),
        noisy_dir=str(noisy_dir),
        segment_length=1.0,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Clean magnitude shape: {sample['clean_magnitude'].shape}")
    print(f"Noisy magnitude shape: {sample['noisy_magnitude'].shape}")
    print(f"Phase shape: {sample['phase'].shape}")
    print(f"File name: {sample['file_name']}")
    
    # Test dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    print(f"\nBatch clean magnitude shape: {batch['clean_magnitude'].shape}")
    print(f"Batch noisy magnitude shape: {batch['noisy_magnitude'].shape}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    print("\nDataset test completed successfully!")
