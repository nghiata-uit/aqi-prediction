"""
Evaluation Script for Speech Enhancement Models

This script evaluates trained models on test sets and generates:
- Enhanced audio files
- Evaluation metrics (PESQ, STOI, SI-SDR)
- Comparison plots
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models import MambaSEUNet, CNNUNet, TransformerUNet, ConformerUNet
from data import create_dataloaders
from data.preprocessing import AudioProcessor, save_audio
from .metrics import compute_all_metrics, count_parameters, compute_rtf


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    processor: AudioProcessor,
    device: str = 'cuda',
    save_audio_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, list]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        processor: Audio processor
        device: Device to run on
        save_audio_dir: Directory to save enhanced audio (optional)
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Dictionary of metric lists for each sample
    """
    model.eval()
    model.to(device)
    
    all_metrics = {
        'pesq': [],
        'stoi': [],
        'si_sdr': [],
        'pesq_improvement': [],
        'stoi_improvement': [],
        'si_sdr_improvement': [],
        'file_name': [],
    }
    
    if save_audio_dir:
        save_audio_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc='Evaluating')):
            if max_samples and idx >= max_samples:
                break
            
            # Move to device
            noisy_mag = batch['noisy_magnitude'].to(device)
            clean_audio = batch['clean_audio']
            noisy_audio = batch['noisy_audio']
            phase = batch['phase'].to(device)
            file_name = batch['file_name'][0]
            
            # Enhance
            enhanced_mag = model(noisy_mag)
            
            # Reconstruct audio
            enhanced_mag = enhanced_mag.squeeze(0).squeeze(0).cpu()  # Remove batch and channel dims
            phase = phase.squeeze(0).cpu()  # Remove batch dim
            
            # Combine magnitude and phase
            enhanced_stft = processor.reconstruct_from_magnitude_phase(enhanced_mag, phase)
            enhanced_audio = processor.istft(enhanced_stft, length=clean_audio.shape[-1])
            
            # Convert to numpy
            clean_np = clean_audio.squeeze().numpy()
            noisy_np = noisy_audio.squeeze().numpy()
            enhanced_np = enhanced_audio.squeeze().numpy()
            
            # Compute metrics
            metrics = compute_all_metrics(
                clean_np,
                enhanced_np,
                noisy_np,
                sample_rate=processor.sample_rate,
            )
            
            # Store metrics
            for key in all_metrics:
                if key == 'file_name':
                    all_metrics[key].append(file_name)
                elif key in metrics:
                    all_metrics[key].append(metrics[key])
            
            # Save audio
            if save_audio_dir:
                save_path = save_audio_dir / f'enhanced_{file_name}'
                save_audio(str(save_path), enhanced_np, processor.sample_rate)
    
    return all_metrics


def generate_comparison_plots(
    results: Dict[str, list],
    output_dir: Path,
):
    """
    Generate comparison plots from evaluation results.
    
    Args:
        results: Dictionary of evaluation results
        output_dir: Output directory for plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot metric distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = ['pesq', 'stoi', 'si_sdr']
    titles = ['PESQ', 'STOI', 'SI-SDR (dB)']
    
    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        if metric in results and len(results[metric]) > 0:
            values = [v for v in results[metric] if v != 0]  # Filter out failed computations
            if values:
                ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
                ax.axvline(np.mean(values), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(values):.3f}')
                ax.set_xlabel(title)
                ax.set_ylabel('Frequency')
                ax.set_title(f'{title} Distribution')
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot improvements
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    improvement_metrics = ['pesq_improvement', 'stoi_improvement', 'si_sdr_improvement']
    improvement_titles = ['PESQ Improvement', 'STOI Improvement', 'SI-SDR Improvement (dB)']
    
    for ax, metric, title in zip(axes, improvement_metrics, improvement_titles):
        if metric in results and len(results[metric]) > 0:
            values = [v for v in results[metric] if v != 0]
            if values:
                ax.hist(values, bins=20, alpha=0.7, edgecolor='black', color='green')
                ax.axvline(np.mean(values), color='red', linestyle='--',
                          label=f'Mean: {np.mean(values):.3f}')
                ax.set_xlabel(title)
                ax.set_ylabel('Frequency')
                ax.set_title(title)
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate speech enhancement model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='mamba_seunet',
                       choices=['mamba_seunet', 'cnn_unet', 'transformer_unet', 'conformer_unet'],
                       help='Model architecture')
    parser.add_argument('--config', type=str, default='configs/experiment_configs.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='vctk_demand',
                       choices=['vctk_demand', 'voicebank_demand'],
                       help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='data/VCTK-DEMAND',
                       help='Root directory of dataset')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory for results')
    parser.add_argument('--save_audio', action='store_true',
                       help='Save enhanced audio files')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders
    _, _, test_loader = create_dataloaders(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        batch_size=1,  # Process one at a time for evaluation
        num_workers=config['evaluation']['num_workers'],
        pin_memory=False,
    )
    
    print(f"Test set: {len(test_loader.dataset)} samples")
    
    # Create model
    if args.model == 'mamba_seunet':
        model = MambaSEUNet()
    elif args.model == 'cnn_unet':
        model = CNNUNet()
    elif args.model == 'transformer_unet':
        model = TransformerUNet()
    elif args.model == 'conformer_unet':
        model = ConformerUNet()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print model info
    params = count_parameters(model)
    print(f"\nModel: {args.model}")
    print(f"Parameters: {params['total']:,}")
    
    # Compute RTF
    try:
        rtf = compute_rtf(model, device=device)
        print(f"Real-Time Factor: {rtf:.4f}")
    except Exception as e:
        print(f"Could not compute RTF: {e}")
    
    # Create audio processor
    data_config = config['data']
    processor = AudioProcessor(
        sample_rate=data_config['sample_rate'],
        n_fft=data_config['n_fft'],
        hop_length=data_config['hop_length'],
        win_length=data_config['win_length'],
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    save_audio_dir = output_dir / 'enhanced_audio' if args.save_audio else None
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        processor=processor,
        device=device,
        save_audio_dir=save_audio_dir,
        max_samples=args.max_samples,
    )
    
    # Compute average metrics
    print("\n=== Evaluation Results ===")
    for metric in ['pesq', 'stoi', 'si_sdr']:
        if metric in results and len(results[metric]) > 0:
            values = [v for v in results[metric] if v != 0]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\n=== Improvements ===")
    for metric in ['pesq_improvement', 'stoi_improvement', 'si_sdr_improvement']:
        if metric in results and len(results[metric]) > 0:
            values = [v for v in results[metric] if v != 0]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'evaluation_results.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'evaluation_results.csv'}")
    
    # Generate plots
    generate_comparison_plots(results, output_dir)
    print(f"Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
