"""
Training Script for Mamba-SEUNet

This module implements the main training loop with:
- Model checkpointing
- TensorBoard logging
- Validation
- Early stopping
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml
from tqdm import tqdm
import argparse

from models import MambaSEUNet, CNNUNet, TransformerUNet, ConformerUNet
from data import create_dataloaders
from .losses import create_loss_function
from .optimizer import create_optimizer, create_scheduler, GradientClipper


class Trainer:
    """
    Trainer for speech enhancement models.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'runs',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create loss function
        loss_config = config.get('loss', {'type': 'combined'})
        self.criterion = create_loss_function(loss_config).to(device)
        
        # Create optimizer
        optimizer_config = config.get('training', {}).get('optimizer', {})
        self.optimizer = create_optimizer(model, optimizer_config)
        
        # Create scheduler
        scheduler_config = config.get('training', {}).get('scheduler', {})
        self.scheduler = create_scheduler(self.optimizer, scheduler_config)
        
        # Gradient clipping
        gradient_clip = config.get('training', {}).get('gradient_clip', 5.0)
        self.gradient_clipper = GradientClipper(max_norm=gradient_clip)
        
        # Mixed precision training
        self.use_amp = config.get('training', {}).get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # TensorBoard
        use_tensorboard = config.get('logging', {}).get('use_tensorboard', True)
        self.writer = SummaryWriter(log_dir) if use_tensorboard else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Early stopping
        early_stopping_config = config.get('training', {}).get('early_stopping', {})
        self.early_stopping_enabled = early_stopping_config.get('enabled', True)
        self.early_stopping_patience = early_stopping_config.get('patience', 15)
        self.early_stopping_min_delta = early_stopping_config.get('min_delta', 0.001)
        self.epochs_without_improvement = 0
        
        # Logging
        self.log_every_n_steps = config.get('logging', {}).get('log_every_n_steps', 100)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        epoch_losses = {}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            noisy_mag = batch['noisy_magnitude'].to(self.device)
            clean_mag = batch['clean_magnitude'].to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                predicted_mag = self.model(noisy_mag)
                
                # Compute loss
                if isinstance(self.criterion, nn.ModuleDict) or hasattr(self.criterion, 'loss_fns'):
                    losses = self.criterion(predicted_mag, clean_mag)
                    loss = losses['total']
                else:
                    loss = self.criterion(predicted_mag, clean_mag)
                    losses = {'total': loss}
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = self.gradient_clipper(self.model)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = self.gradient_clipper(self.model)
                self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value.item()
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            if self.writer and self.global_step % self.log_every_n_steps == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(f'train/{key}', value.item(), self.global_step)
                self.writer.add_scalar('train/grad_norm', grad_norm, self.global_step)
                self.writer.add_scalar('train/learning_rate', 
                                     self.optimizer.param_groups[0]['lr'], 
                                     self.global_step)
            
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        val_losses = {}
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            # Move data to device
            noisy_mag = batch['noisy_magnitude'].to(self.device)
            clean_mag = batch['clean_magnitude'].to(self.device)
            
            # Forward pass
            predicted_mag = self.model(noisy_mag)
            
            # Compute loss
            if isinstance(self.criterion, nn.ModuleDict) or hasattr(self.criterion, 'loss_fns'):
                losses = self.criterion(predicted_mag, clean_mag)
            else:
                loss = self.criterion(predicted_mag, clean_mag)
                losses = {'total': loss}
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in val_losses:
                    val_losses[key] = 0.0
                val_losses[key] += value.item()
            
            num_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        if hasattr(self.scheduler, 'state_dict'):
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    def train(self, num_epochs: int):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_losses['total']:.6f}")
            
            # Validate
            validate_every_n = self.config.get('training', {}).get('validate_every_n_epochs', 1)
            if (epoch + 1) % validate_every_n == 0:
                val_losses = self.validate()
                print(f"  Val Loss: {val_losses['total']:.6f}")
                
                # Log validation metrics
                if self.writer:
                    for key, value in val_losses.items():
                        self.writer.add_scalar(f'val/{key}', value, epoch)
                
                # Check for improvement
                val_loss = val_losses['total']
                if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(f'epoch_{epoch}_best.pt', is_best=True)
                    print(f"  New best model! Val Loss: {val_loss:.6f}")
                else:
                    self.epochs_without_improvement += 1
            
            # Update learning rate
            if hasattr(self.scheduler, 'step'):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            save_every_n = self.config.get('training', {}).get('save_every_n_epochs', 5)
            if (epoch + 1) % save_every_n == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')
            
            # Early stopping
            if self.early_stopping_enabled:
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        print("\nTraining completed!")
        
        if self.writer:
            self.writer.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Mamba-SEUNet')
    parser.add_argument('--config', type=str, default='configs/experiment_configs.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='mamba_seunet',
                       choices=['mamba_seunet', 'cnn_unet', 'transformer_unet', 'conformer_unet'],
                       help='Model architecture to train')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--dataset', type=str, default='vctk_demand',
                       choices=['vctk_demand', 'voicebank_demand'],
                       help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='data/VCTK-DEMAND',
                       help='Root directory of dataset')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
    )
    
    print(f"Train set: {len(train_loader.dataset)} samples")
    print(f"Val set: {len(val_loader.dataset)} samples")
    print(f"Test set: {len(test_loader.dataset)} samples")
    
    # Create model
    model_config = config['model']
    if args.model == 'mamba_seunet':
        model = MambaSEUNet(
            num_mamba_blocks=model_config['num_mamba_blocks'],
            encoder_channels=[64, 128, 256, 512],
        )
    elif args.model == 'cnn_unet':
        model = CNNUNet()
    elif args.model == 'transformer_unet':
        model = TransformerUNet()
    elif args.model == 'conformer_unet':
        model = ConformerUNet()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print(f"\nModel: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=f'checkpoints/{args.model}',
        log_dir=f'runs/{args.model}',
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    num_epochs = config['training']['epochs']
    trainer.train(num_epochs)


if __name__ == '__main__':
    main()
