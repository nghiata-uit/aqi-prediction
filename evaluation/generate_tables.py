"""
Generate Tables from Paper

This script generates tables that reproduce results from the paper:
- Table 1: Main comparison with baseline models
- Table IV: Ablation study on number of TS-Mamba blocks
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse


def generate_table_1(results_dir: Path) -> pd.DataFrame:
    """
    Generate Table 1: Main comparison with baseline models.
    
    Expected columns: Model, PESQ, STOI, SI-SDR, Params (M), FLOPs (G)
    
    Args:
        results_dir: Directory containing evaluation results
        
    Returns:
        DataFrame with comparison results
    """
    models = ['cnn_unet', 'transformer_unet', 'conformer_unet', 'mamba_seunet']
    
    data = []
    
    for model in models:
        result_file = results_dir / model / 'evaluation_results.csv'
        
        if result_file.exists():
            df = pd.read_csv(result_file)
            
            # Compute mean metrics
            pesq_mean = df['pesq'].mean()
            stoi_mean = df['stoi'].mean()
            si_sdr_mean = df['si_sdr'].mean()
            
            # Load parameter count (should be saved separately)
            param_file = results_dir / model / 'model_info.txt'
            params = 0.0  # Placeholder
            flops = 0.0   # Placeholder
            
            data.append({
                'Model': model.replace('_', '-').upper(),
                'PESQ': f'{pesq_mean:.2f}',
                'STOI': f'{stoi_mean:.3f}',
                'SI-SDR': f'{si_sdr_mean:.2f}',
                'Params (M)': f'{params:.1f}',
                'FLOPs (G)': f'{flops:.1f}',
            })
    
    df = pd.DataFrame(data)
    return df


def generate_table_4(results_dir: Path) -> pd.DataFrame:
    """
    Generate Table IV: Ablation study on number of TS-Mamba blocks.
    
    Expected columns: # Blocks, PESQ, STOI, SI-SDR, Params (M), FLOPs (G)
    
    Args:
        results_dir: Directory containing ablation study results
        
    Returns:
        DataFrame with ablation results
    """
    num_blocks_list = [2, 4, 6, 8, 10]
    
    data = []
    
    for num_blocks in num_blocks_list:
        result_file = results_dir / f'mamba_seunet_{num_blocks}blocks' / 'evaluation_results.csv'
        
        if result_file.exists():
            df = pd.read_csv(result_file)
            
            # Compute mean metrics
            pesq_mean = df['pesq'].mean()
            stoi_mean = df['stoi'].mean()
            si_sdr_mean = df['si_sdr'].mean()
            
            # Placeholders for params and FLOPs
            params = 0.0
            flops = 0.0
            
            data.append({
                '# Mamba Blocks': num_blocks,
                'PESQ': f'{pesq_mean:.2f}',
                'STOI': f'{stoi_mean:.3f}',
                'SI-SDR': f'{si_sdr_mean:.2f}',
                'Params (M)': f'{params:.1f}',
                'FLOPs (G)': f'{flops:.1f}',
            })
    
    df = pd.DataFrame(data)
    return df


def to_latex(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to LaTeX table format.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        LaTeX table string
    """
    return df.to_latex(index=False, escape=False)


def to_markdown(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to Markdown table format.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        Markdown table string
    """
    return df.to_markdown(index=False)


def main():
    """Main table generation function."""
    parser = argparse.ArgumentParser(description='Generate result tables from paper')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, default='results/tables',
                       help='Output directory for tables')
    parser.add_argument('--format', type=str, default='all',
                       choices=['csv', 'latex', 'markdown', 'all'],
                       help='Output format')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate Table 1
    print("Generating Table 1: Main Comparison...")
    table1 = generate_table_1(results_dir)
    
    print("\n=== Table 1: Main Comparison with Baselines ===")
    print(table1.to_string(index=False))
    
    # Save Table 1
    if args.format in ['csv', 'all']:
        table1.to_csv(output_dir / 'table1_main_comparison.csv', index=False)
    
    if args.format in ['latex', 'all']:
        with open(output_dir / 'table1_main_comparison.tex', 'w') as f:
            f.write(to_latex(table1))
    
    if args.format in ['markdown', 'all']:
        with open(output_dir / 'table1_main_comparison.md', 'w') as f:
            f.write(to_markdown(table1))
    
    # Generate Table 4
    print("\nGenerating Table IV: Ablation Study...")
    table4 = generate_table_4(results_dir / 'ablation')
    
    print("\n=== Table IV: Ablation Study on Number of TS-Mamba Blocks ===")
    print(table4.to_string(index=False))
    
    # Save Table 4
    if args.format in ['csv', 'all']:
        table4.to_csv(output_dir / 'table4_ablation_study.csv', index=False)
    
    if args.format in ['latex', 'all']:
        with open(output_dir / 'table4_ablation_study.tex', 'w') as f:
            f.write(to_latex(table4))
    
    if args.format in ['markdown', 'all']:
        with open(output_dir / 'table4_ablation_study.md', 'w') as f:
            f.write(to_markdown(table4))
    
    print(f"\nTables saved to: {output_dir}")


if __name__ == '__main__':
    main()
