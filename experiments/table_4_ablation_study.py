"""
Table IV: Ablation Study Experiment

This script trains Mamba-SEUNet with varying numbers of TS-Mamba blocks,
reproducing Table IV from the paper.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import Trainer
from evaluation.evaluate import evaluate_model
from evaluation.metrics import count_parameters
from models import MambaSEUNet
from data import create_dataloaders


def train_and_evaluate_ablation(
    num_blocks: int,
    config: dict,
    dataset_name: str,
    data_root: str,
    output_dir: Path,
):
    """
    Train and evaluate Mamba-SEUNet with specific number of blocks.
    
    Args:
        num_blocks: Number of TS-Mamba blocks
        config: Configuration dictionary
        dataset_name: Dataset name
        data_root: Root directory of dataset
        output_dir: Output directory for results
    """
    print(f"\n{'='*60}")
    print(f"Training Mamba-SEUNet with {num_blocks} TS-Mamba blocks")
    print(f"{'='*60}\n")
    
    # Create model
    model = MambaSEUNet(num_mamba_blocks=num_blocks)
    
    # Print model info
    params = count_parameters(model)
    print(f"Number of blocks: {num_blocks}")
    print(f"Total parameters: {params['total']:,}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=dataset_name,
        root_dir=data_root,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
    )
    
    # Create trainer
    model_name = f'mamba_seunet_{num_blocks}blocks'
    checkpoint_dir = output_dir / model_name / 'checkpoints'
    log_dir = output_dir / model_name / 'logs'
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
    )
    
    # Train
    print(f"\nTraining...")
    num_epochs = config['training']['epochs']
    trainer.train(num_epochs)
    
    # Evaluate
    print(f"\nEvaluating...")
    from data.preprocessing import AudioProcessor
    
    processor = AudioProcessor(
        sample_rate=config['data']['sample_rate'],
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length'],
        win_length=config['data']['win_length'],
    )
    
    eval_output_dir = output_dir / model_name / 'evaluation'
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        processor=processor,
        device=trainer.device,
        save_audio_dir=eval_output_dir / 'enhanced_audio',
    )
    
    # Save results
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(results)
    df.to_csv(eval_output_dir / 'evaluation_results.csv', index=False)
    
    # Print summary
    print(f"\n=== Results for {num_blocks} blocks ===")
    for metric in ['pesq', 'stoi', 'si_sdr']:
        if metric in results:
            values = [v for v in results[metric] if v != 0]
            if values:
                print(f"{metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")


def main():
    """Main ablation study function."""
    parser = argparse.ArgumentParser(description='Table IV: Ablation Study Experiment')
    parser.add_argument('--config', type=str, default='configs/experiment_configs.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='vctk_demand',
                       choices=['vctk_demand', 'voicebank_demand'],
                       help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='data/VCTK-DEMAND',
                       help='Root directory of dataset')
    parser.add_argument('--output_dir', type=str, default='results/ablation',
                       help='Output directory for results')
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[2, 4, 6, 8, 10],
                       help='Numbers of Mamba blocks to test')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with experiment-specific settings
    if 'experiments' in config and 'table_4_ablation_study' in config['experiments']:
        exp_config = config['experiments']['table_4_ablation_study']
        config.update(exp_config)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train and evaluate for each number of blocks
    for num_blocks in args.num_blocks:
        try:
            train_and_evaluate_ablation(
                num_blocks=num_blocks,
                config=config,
                dataset_name=args.dataset,
                data_root=args.data_root,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"\nError with {num_blocks} blocks: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate ablation table
    print("\n" + "="*60)
    print("Generating Ablation Study Table")
    print("="*60)
    
    from evaluation.generate_tables import generate_table_4
    
    table = generate_table_4(output_dir)
    print("\n=== Table IV: Ablation Study on Number of TS-Mamba Blocks ===")
    print(table.to_string(index=False))
    
    # Save table
    table_dir = output_dir / 'tables'
    table_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_dir / 'table4_ablation_study.csv', index=False)
    
    print(f"\nAblation study completed! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
