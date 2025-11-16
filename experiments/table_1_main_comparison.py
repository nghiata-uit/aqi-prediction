"""
Table 1: Main Comparison Experiment

This script trains and evaluates all baseline models and Mamba-SEUNet,
reproducing Table 1 from the paper.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import Trainer
from evaluation.evaluate import evaluate_model
from evaluation.metrics import count_parameters, compute_flops
from models import MambaSEUNet, CNNUNet, TransformerUNet, ConformerUNet
from data import create_dataloaders


def train_and_evaluate_model(
    model_name: str,
    config: dict,
    dataset_name: str,
    data_root: str,
    output_dir: Path,
):
    """
    Train and evaluate a single model.
    
    Args:
        model_name: Name of the model
        config: Configuration dictionary
        dataset_name: Dataset name
        data_root: Root directory of dataset
        output_dir: Output directory for results
    """
    print(f"\n{'='*60}")
    print(f"Training and Evaluating: {model_name}")
    print(f"{'='*60}\n")
    
    # Create model
    if model_name == 'mamba_seunet':
        model = MambaSEUNet(
            num_mamba_blocks=config['model']['num_mamba_blocks'],
        )
    elif model_name == 'cnn_unet':
        model = CNNUNet()
    elif model_name == 'transformer_unet':
        model = TransformerUNet()
    elif model_name == 'conformer_unet':
        model = ConformerUNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Print model info
    params = count_parameters(model)
    print(f"Model: {model_name}")
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=dataset_name,
        root_dir=data_root,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
    )
    
    # Create trainer
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
    print(f"\nTraining {model_name}...")
    num_epochs = config['training']['epochs']
    trainer.train(num_epochs)
    
    # Evaluate
    print(f"\nEvaluating {model_name}...")
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
    print(f"\n=== {model_name} Results ===")
    for metric in ['pesq', 'stoi', 'si_sdr']:
        if metric in results:
            values = [v for v in results[metric] if v != 0]
            if values:
                print(f"{metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(description='Table 1: Main Comparison Experiment')
    parser.add_argument('--config', type=str, default='configs/experiment_configs.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='vctk_demand',
                       choices=['vctk_demand', 'voicebank_demand'],
                       help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='data/VCTK-DEMAND',
                       help='Root directory of dataset')
    parser.add_argument('--output_dir', type=str, default='results/table1',
                       help='Output directory for results')
    parser.add_argument('--models', nargs='+', 
                       default=['cnn_unet', 'transformer_unet', 'conformer_unet', 'mamba_seunet'],
                       help='Models to train and evaluate')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with experiment-specific settings
    if 'experiments' in config and 'table_1_main_comparison' in config['experiments']:
        exp_config = config['experiments']['table_1_main_comparison']
        config.update(exp_config)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train and evaluate each model
    for model_name in args.models:
        try:
            train_and_evaluate_model(
                model_name=model_name,
                config=config,
                dataset_name=args.dataset,
                data_root=args.data_root,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"\nError training/evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comparison table
    print("\n" + "="*60)
    print("Generating Comparison Table")
    print("="*60)
    
    from evaluation.generate_tables import generate_table_1
    
    table = generate_table_1(output_dir)
    print("\n=== Table 1: Main Comparison with Baselines ===")
    print(table.to_string(index=False))
    
    # Save table
    table_dir = output_dir / 'tables'
    table_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_dir / 'table1_main_comparison.csv', index=False)
    
    print(f"\nExperiment completed! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
