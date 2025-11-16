"""
Optimizer and Learning Rate Scheduler Configuration

This module provides utilities for creating optimizers and schedulers.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    ExponentialLR,
)
from typing import Dict, Optional
import math


class WarmupScheduler:
    """
    Learning rate scheduler with warmup phase.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        base_scheduler: Base scheduler to use after warmup
        warmup_start_lr: Starting learning rate for warmup
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        base_scheduler: Optional[object] = None,
        warmup_start_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.warmup_start_lr = warmup_start_lr
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None, metrics: Optional[float] = None):
        """Step the scheduler."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.warmup_start_lr + (self.base_lrs[i] - self.warmup_start_lr) * \
                     (self.current_epoch / self.warmup_epochs)
                param_group['lr'] = lr
        else:
            # After warmup, use base scheduler
            if self.base_scheduler is not None:
                if isinstance(self.base_scheduler, ReduceLROnPlateau):
                    if metrics is not None:
                        self.base_scheduler.step(metrics)
                else:
                    self.base_scheduler.step()
    
    def get_last_lr(self):
        """Get last learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]


def create_optimizer(
    model: torch.nn.Module,
    config: Dict,
) -> optim.Optimizer:
    """
    Create optimizer from configuration.
    
    Args:
        model: PyTorch model
        config: Optimizer configuration dictionary
        
    Returns:
        PyTorch optimizer
    """
    optimizer_type = config.get('type', 'adam').lower()
    learning_rate = config.get('learning_rate', 0.0001)
    weight_decay = config.get('weight_decay', 0.0001)
    
    if optimizer_type == 'adam':
        betas = config.get('betas', [0.9, 0.999])
        eps = config.get('eps', 1e-8)
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    
    elif optimizer_type == 'adamw':
        betas = config.get('betas', [0.9, 0.999])
        eps = config.get('eps', 1e-8)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    
    elif optimizer_type == 'sgd':
        momentum = config.get('momentum', 0.9)
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    
    elif optimizer_type == 'rmsprop':
        alpha = config.get('alpha', 0.99)
        eps = config.get('eps', 1e-8)
        momentum = config.get('momentum', 0.0)
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            alpha=alpha,
            eps=eps,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    config: Dict,
) -> object:
    """
    Create learning rate scheduler from configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Scheduler configuration dictionary
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = config.get('type', 'cosine_annealing').lower()
    
    if scheduler_type == 'cosine_annealing':
        T_max = config.get('T_max', 100)
        eta_min = config.get('eta_min', 1e-6)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
        )
    
    elif scheduler_type == 'reduce_on_plateau':
        mode = config.get('mode', 'min')
        factor = config.get('factor', 0.5)
        patience = config.get('patience', 10)
        min_lr = config.get('min_lr', 1e-6)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
    
    elif scheduler_type == 'step':
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    
    elif scheduler_type == 'exponential':
        gamma = config.get('gamma', 0.95)
        scheduler = ExponentialLR(
            optimizer,
            gamma=gamma,
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Add warmup if specified
    warmup_epochs = config.get('warmup_epochs', 0)
    if warmup_epochs > 0:
        warmup_start_lr = config.get('warmup_start_lr', 1e-6)
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            base_scheduler=scheduler,
            warmup_start_lr=warmup_start_lr,
        )
    
    return scheduler


class GradientClipper:
    """
    Gradient clipping utility.
    
    Args:
        max_norm: Maximum gradient norm
        norm_type: Type of norm to use (default: 2)
    """
    
    def __init__(self, max_norm: float = 5.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, model: torch.nn.Module) -> float:
        """
        Clip gradients.
        
        Args:
            model: PyTorch model
            
        Returns:
            Total gradient norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_norm,
            norm_type=self.norm_type,
        )


if __name__ == "__main__":
    # Test optimizer and scheduler creation
    print("Testing optimizer and scheduler creation...")
    
    # Create a dummy model
    model = torch.nn.Linear(10, 10)
    
    # Test optimizer creation
    optimizer_config = {
        'type': 'adam',
        'learning_rate': 0.001,
        'betas': [0.9, 0.999],
        'weight_decay': 0.0001,
    }
    
    optimizer = create_optimizer(model, optimizer_config)
    print(f"Created optimizer: {type(optimizer).__name__}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    
    # Test scheduler creation
    scheduler_config = {
        'type': 'cosine_annealing',
        'T_max': 100,
        'eta_min': 1e-6,
        'warmup_epochs': 5,
        'warmup_start_lr': 1e-6,
    }
    
    scheduler = create_scheduler(optimizer, scheduler_config)
    print(f"\nCreated scheduler: {type(scheduler).__name__}")
    
    # Test warmup
    print("\n--- Testing warmup phase ---")
    for epoch in range(10):
        scheduler.step(epoch)
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: lr = {lr:.6f}")
    
    # Test gradient clipping
    print("\n--- Testing gradient clipping ---")
    clipper = GradientClipper(max_norm=5.0)
    
    # Create dummy gradients
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 10
    
    grad_norm = clipper(model)
    print(f"Gradient norm before clipping: {grad_norm:.4f}")
    
    # Check norm after clipping
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Gradient norm after clipping: {total_norm:.4f}")
