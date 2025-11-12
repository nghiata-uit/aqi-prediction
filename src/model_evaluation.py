"""
Module đánh giá và so sánh models
"""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Tính toán các metrics đánh giá model
    
    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        
    Returns:
        Dictionary chứa các metrics: MAE, RMSE, R²
    """
    try:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R²': round(r2, 4)
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Error calculating metrics: {str(e)}")
        raise


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                     title: str, save_path: str) -> None:
    """
    Vẽ biểu đồ actual vs predicted
    
    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn lưu file
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, color='#3498db')
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 
                     'r--', lw=2, label='Perfect prediction')
        axes[0].set_xlabel('Actual AQI', fontsize=12)
        axes[0].set_ylabel('Predicted AQI', fontsize=12)
        axes[0].set_title(f'{title} - Scatter Plot', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Time series comparison
        indices = range(min(len(y_true), 200))  # Show first 200 points
        axes[1].plot(indices, y_true[:200], label='Actual', linewidth=2, alpha=0.7)
        axes[1].plot(indices, y_pred[:200], label='Predicted', linewidth=2, alpha=0.7)
        axes[1].set_xlabel('Sample Index', fontsize=12)
        axes[1].set_ylabel('AQI', fontsize=12)
        axes[1].set_title(f'{title} - Time Series', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Create directory if not exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Prediction plot saved to {save_path}")
        
    except Exception as e:
        logger.error(f"❌ Error plotting predictions: {str(e)}")
        raise


def plot_feature_importance(importance_df: pd.DataFrame, save_path: str, top_n: int = 20) -> None:
    """
    Vẽ biểu đồ feature importance
    
    Args:
        importance_df: DataFrame với columns ['feature', 'importance']
        save_path: Đường dẫn lưu file
        top_n: Số lượng features quan trọng nhất để hiển thị
    """
    try:
        # Sort by importance and get top N
        importance_df_sorted = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        colors = sns.color_palette('husl', n_colors=len(importance_df_sorted))
        
        bars = plt.barh(range(len(importance_df_sorted)), 
                       importance_df_sorted['importance'].values,
                       color=colors)
        
        plt.yticks(range(len(importance_df_sorted)), 
                  importance_df_sorted['feature'].values)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Create directory if not exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Feature importance plot saved to {save_path}")
        
    except Exception as e:
        logger.error(f"❌ Error plotting feature importance: {str(e)}")
        raise


def compare_models(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    So sánh kết quả các models
    
    Args:
        results_dict: Dictionary với format {model_name: {metric: value}}
        
    Returns:
        DataFrame so sánh các models
    """
    try:
        df_comparison = pd.DataFrame(results_dict).T
        
        # Sort by R² score (descending)
        if 'R²' in df_comparison.columns:
            df_comparison = df_comparison.sort_values('R²', ascending=False)
        
        logger.info("\n" + "="*60)
        logger.info("📊 MODEL COMPARISON")
        logger.info("="*60)
        logger.info("\n" + df_comparison.to_string())
        logger.info("="*60 + "\n")
        
        return df_comparison
        
    except Exception as e:
        logger.error(f"❌ Error comparing models: {str(e)}")
        raise


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str) -> None:
    """
    Vẽ biểu đồ so sánh các models
    
    Args:
        comparison_df: DataFrame kết quả so sánh
        save_path: Đường dẫn lưu file
    """
    try:
        metrics = ['MAE', 'RMSE', 'R²']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                ax = axes[idx]
                colors = sns.color_palette('Set2', n_colors=len(comparison_df))
                
                bars = ax.bar(range(len(comparison_df)), 
                            comparison_df[metric].values,
                            color=colors)
                
                ax.set_xticks(range(len(comparison_df)))
                ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
                ax.set_ylabel(metric, fontsize=12)
                ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}',
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Create directory if not exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Model comparison plot saved to {save_path}")
        
    except Exception as e:
        logger.error(f"❌ Error plotting model comparison: {str(e)}")
        raise
