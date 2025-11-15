"""
Ensemble model implementation for AQI prediction
"""
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple models
    """
    
    def __init__(self, models: Dict[str, object], weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble model
        
        Args:
            models: Dictionary of {model_name: model_object}
            weights: Optional dictionary of {model_name: weight}
        """
        self.models = models
        
        if weights is None:
            # Equal weights
            self.weights = {name: 1.0 / len(models) for name in models.keys()}
        else:
            # Normalize weights
            total_weight = sum(weights.values())
            self.weights = {name: w / total_weight for name, w in weights.items()}
        
        logger.info(f"Ensemble model initialized with {len(models)} models")
        logger.info(f"Weights: {self.weights}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using ensemble of models
        
        Args:
            X: Input features
            
        Returns:
            Weighted average predictions
        """
        predictions = []
        
        for name, model in self.models.items():
            try:
                # Handle different model types
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    
                    # Flatten if needed
                    if len(pred.shape) > 1:
                        pred = pred.flatten()
                    
                    # Apply weight
                    weighted_pred = pred * self.weights[name]
                    predictions.append(weighted_pred)
                    
            except Exception as e:
                logger.warning(f"Error getting prediction from {name}: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        # Sum weighted predictions
        ensemble_prediction = np.sum(predictions, axis=0)
        
        return ensemble_prediction
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score
        
        Args:
            X: Input features
            y: True targets
            
        Returns:
            R² score
        """
        from sklearn.metrics import r2_score
        
        predictions = self.predict(X)
        return r2_score(y, predictions)


def train_ensemble_model(
    models_dict: Dict[str, object],
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_val_seq: Optional[np.ndarray] = None,
    y_val_seq: Optional[np.ndarray] = None
) -> Optional[EnsembleModel]:
    """
    Train ensemble model by combining multiple models
    
    Args:
        models_dict: Dictionary of trained models {name: model}
        X_val: Validation features (2D)
        y_val: Validation targets
        X_val_seq: Optional validation sequences for sequence models
        y_val_seq: Optional validation targets for sequence models
        
    Returns:
        Trained ensemble model
    """
    try:
        logger.info("🎯 Training Ensemble model...")
        
        # Calculate performance-based weights
        weights = {}
        
        for name, model in models_dict.items():
            try:
                if name in ['LSTM', 'GRU'] and X_val_seq is not None and y_val_seq is not None:
                    # Use sequence data for LSTM/GRU
                    pred = model.predict(X_val_seq, verbose=0)
                    if len(pred.shape) > 1:
                        pred = pred.flatten()
                    
                    from sklearn.metrics import mean_absolute_error
                    mae = mean_absolute_error(y_val_seq, pred)
                else:
                    # Use regular data for other models
                    pred = model.predict(X_val)
                    if len(pred.shape) > 1:
                        pred = pred.flatten()
                    
                    from sklearn.metrics import mean_absolute_error
                    mae = mean_absolute_error(y_val, pred)
                
                # Weight is inverse of MAE (lower MAE = higher weight)
                weights[name] = 1.0 / (mae + 0.001)  # Add small epsilon to avoid division by zero
                
                logger.info(f"   {name} - MAE: {mae:.4f}, Weight: {weights[name]:.4f}")
                
            except Exception as e:
                logger.warning(f"   Could not evaluate {name}: {str(e)}")
                continue
        
        if not weights:
            logger.error("No models could be evaluated for ensemble")
            return None
        
        # Create ensemble model (only include models that work with 2D input)
        ensemble_models = {}
        for name, model in models_dict.items():
            if name not in ['LSTM', 'GRU']:  # Exclude sequence models from ensemble
                ensemble_models[name] = model
        
        if not ensemble_models:
            logger.warning("No compatible models for ensemble")
            return None
        
        # Filter weights to only include ensemble models
        ensemble_weights = {name: weights[name] for name in ensemble_models.keys() if name in weights}
        
        ensemble = EnsembleModel(ensemble_models, ensemble_weights)
        
        # Evaluate ensemble
        ensemble_score = ensemble.score(X_val, y_val)
        logger.info(f"   Ensemble R² score: {ensemble_score:.4f}")
        logger.info("✅ Ensemble training completed")
        
        return ensemble
        
    except Exception as e:
        logger.error(f"❌ Error training Ensemble: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
