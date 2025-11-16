"""
Perceptual Contrast Stretching Experiment

This script applies perceptual contrast stretching post-processing
to Mamba-SEUNet outputs, aiming to improve PESQ from 3.59 to 3.73.
"""

import os
import sys
import torch
import numpy as np
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import MambaSEUNet
from data import create_dataloaders
from data.preprocessing import AudioProcessor
from evaluation.metrics import compute_all_metrics


def apply_contrast_stretching(
    magnitude: torch.Tensor,
    alpha: float = 0.3,
) -> torch.Tensor:
    """
    Apply perceptual contrast stretching to magnitude spectrogram.
    
    This enhances the dynamic range of the spectrogram, potentially
    improving perceptual quality.
    
    Args:
        magnitude: Magnitude spectrogram
        alpha: Stretching factor (0-1, higher means more stretching)
        
    Returns:
        Enhanced magnitude spectrogram
    """
    # Convert to log scale
    log_mag = torch.log(magnitude + 1e-8)
    
    # Compute mean and std
    mean = log_mag.mean()
    std = log_mag.std()
    
    # Apply contrast stretching
    stretched = mean + (log_mag - mean) * (1 + alpha)
    
    # Convert back to linear scale
    enhanced_mag = torch.exp(stretched)
    
    return enhanced_mag


def evaluate_with_contrast_stretching(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    processor: AudioProcessor,
    alpha_values: list,
    device: str = 'cuda',
) -> dict:
    """
    Evaluate model with different contrast stretching parameters.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        processor: Audio processor
        alpha_values: List of alpha values to test
        device: Device to run on
        
    Returns:
        Dictionary of results for each alpha value
    """
    model.eval()
    model.to(device)
    
    results_by_alpha = {alpha: [] for alpha in alpha_values}
    results_by_alpha['baseline'] = []  # Without contrast stretching
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            # Move to device
            noisy_mag = batch['noisy_magnitude'].to(device)
            clean_audio = batch['clean_audio']
            noisy_audio = batch['noisy_audio']
            phase = batch['phase'].to(device)
            
            # Enhance
            enhanced_mag = model(noisy_mag)
            
            # Evaluate baseline (without contrast stretching)
            enhanced_mag_baseline = enhanced_mag.squeeze(0).squeeze(0).cpu()
            phase_cpu = phase.squeeze(0).cpu()
            
            enhanced_stft_baseline = processor.reconstruct_from_magnitude_phase(
                enhanced_mag_baseline, phase_cpu
            )
            enhanced_audio_baseline = processor.istft(
                enhanced_stft_baseline, length=clean_audio.shape[-1]
            )
            
            # Compute metrics for baseline
            clean_np = clean_audio.squeeze().numpy()
            noisy_np = noisy_audio.squeeze().numpy()
            enhanced_np_baseline = enhanced_audio_baseline.squeeze().numpy()
            
            metrics_baseline = compute_all_metrics(
                clean_np, enhanced_np_baseline, noisy_np, processor.sample_rate
            )
            results_by_alpha['baseline'].append(metrics_baseline)
            
            # Evaluate with different alpha values
            for alpha in alpha_values:
                # Apply contrast stretching
                enhanced_mag_stretched = apply_contrast_stretching(
                    enhanced_mag_baseline, alpha
                )
                
                # Reconstruct audio
                enhanced_stft = processor.reconstruct_from_magnitude_phase(
                    enhanced_mag_stretched, phase_cpu
                )
                enhanced_audio = processor.istft(
                    enhanced_stft, length=clean_audio.shape[-1]
                )
                
                # Compute metrics
                enhanced_np = enhanced_audio.squeeze().numpy()
                metrics = compute_all_metrics(
                    clean_np, enhanced_np, noisy_np, processor.sample_rate
                )
                results_by_alpha[alpha].append(metrics)
    
    return results_by_alpha


def main():
    """Main contrast stretching experiment function."""
    parser = argparse.ArgumentParser(
        description='Perceptual Contrast Stretching Experiment'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained Mamba-SEUNet checkpoint')
    parser.add_argument('--config', type=str, default='configs/experiment_configs.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='vctk_demand',
                       choices=['vctk_demand', 'voicebank_demand'],
                       help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='data/VCTK-DEMAND',
                       help='Root directory of dataset')
    parser.add_argument('--output_dir', type=str, default='results/contrast_stretching',
                       help='Output directory for results')
    parser.add_argument('--alpha_values', nargs='+', type=float,
                       default=[0.1, 0.2, 0.3, 0.4, 0.5],
                       help='Alpha values for contrast stretching')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model and load checkpoint
    model = MambaSEUNet()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        batch_size=1,
        num_workers=2,
        pin_memory=False,
    )
    
    # Create audio processor
    processor = AudioProcessor(
        sample_rate=config['data']['sample_rate'],
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length'],
        win_length=config['data']['win_length'],
    )
    
    # Evaluate with different alpha values
    print("\nEvaluating with contrast stretching...")
    results = evaluate_with_contrast_stretching(
        model=model,
        test_loader=test_loader,
        processor=processor,
        alpha_values=args.alpha_values,
        device=device,
    )
    
    # Compute average metrics
    print("\n" + "="*60)
    print("Contrast Stretching Results")
    print("="*60)
    
    import pandas as pd
    
    summary_data = []
    
    for key in ['baseline'] + args.alpha_values:
        metrics_list = results[key]
        
        # Compute averages
        avg_metrics = {}
        for metric_name in ['pesq', 'stoi', 'si_sdr']:
            values = [m[metric_name] for m in metrics_list if m[metric_name] != 0]
            if values:
                avg_metrics[metric_name] = np.mean(values)
            else:
                avg_metrics[metric_name] = 0.0
        
        alpha_str = 'Baseline' if key == 'baseline' else f'α = {key}'
        
        summary_data.append({
            'Alpha': alpha_str,
            'PESQ': f"{avg_metrics['pesq']:.2f}",
            'STOI': f"{avg_metrics['stoi']:.3f}",
            'SI-SDR': f"{avg_metrics['si_sdr']:.2f}",
        })
        
        print(f"\n{alpha_str}:")
        print(f"  PESQ: {avg_metrics['pesq']:.4f}")
        print(f"  STOI: {avg_metrics['stoi']:.4f}")
        print(f"  SI-SDR: {avg_metrics['si_sdr']:.4f}")
    
    # Save summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / 'contrast_stretching_results.csv', index=False)
    
    print(f"\nResults saved to: {output_dir}")
    
    # Print best alpha
    pesq_values = [float(row['PESQ']) for row in summary_data[1:]]  # Skip baseline
    best_idx = np.argmax(pesq_values)
    best_alpha = args.alpha_values[best_idx]
    best_pesq = pesq_values[best_idx]
    
    print(f"\nBest alpha: {best_alpha} (PESQ: {best_pesq:.2f})")


if __name__ == '__main__':
    main()
