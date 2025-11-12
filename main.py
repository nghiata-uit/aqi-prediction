"""
Script chính để train và evaluate models
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.model_training import AQIModelTrainer
from src.model_evaluation import (calculate_metrics, compare_models, 
                                   plot_predictions, plot_feature_importance,
                                   plot_model_comparison)
from src.prediction import predict_next_24h, visualize_predictions

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_sequences(X, y, seq_length=24):
    """
    Prepare sequences for LSTM
    
    Args:
        X: Feature array
        y: Target array
        seq_length: Length of sequences
        
    Returns:
        X_seq, y_seq: Sequenced data
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    
    return np.array(X_seq), np.array(y_seq)


def main():
    """
    Main pipeline:
    1. Load và preprocess data
    2. Feature engineering
    3. Split train/val/test
    4. Train 3 models (RF, XGBoost, LSTM)
    5. Evaluate và compare
    6. Predict 24h ahead
    7. Save results
    """
    print("\n" + "="*70)
    print("🚀 STARTING AQI PREDICTION PIPELINE")
    print("="*70 + "\n")
    
    # Define paths
    base_dir = Path(__file__).parent
    data_path = base_dir / 'data' / 'sample_data.csv'
    model_dir = base_dir / 'models'
    results_dir = base_dir / 'results'
    
    # Create directories
    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    try:
        # ===== STEP 1: Load and Preprocess Data =====
        logger.info("\n📥 STEP 1: Loading and Preprocessing Data")
        logger.info("-" * 70)
        df = preprocess_data(str(data_path))
        
        # ===== STEP 2: Feature Engineering =====
        logger.info("\n🔧 STEP 2: Feature Engineering")
        logger.info("-" * 70)
        df_featured = engineer_features(df)
        
        # ===== STEP 3: Prepare Data for Training =====
        logger.info("\n📊 STEP 3: Preparing Data for Training")
        logger.info("-" * 70)
        
        # Define features and target
        exclude_cols = ['datetime', 'aqi', 'lon', 'lat']
        feature_cols = [col for col in df_featured.columns if col not in exclude_cols]
        
        X = df_featured[feature_cols].values
        y = df_featured['aqi'].values
        
        logger.info(f"   Feature shape: {X.shape}")
        logger.info(f"   Target shape: {y.shape}")
        logger.info(f"   Number of features: {len(feature_cols)}")
        
        # Time-based split (70/15/15)
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        logger.info(f"   Train set: {X_train.shape[0]} samples")
        logger.info(f"   Validation set: {X_val.shape[0]} samples")
        logger.info(f"   Test set: {X_test.shape[0]} samples")
        
        # Scale features
        logger.info("\n   Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        scaler_path = model_dir / 'scaler.pkl'
        joblib.dump(scaler, scaler_path)
        logger.info(f"   ✅ Scaler saved to {scaler_path}")
        
        # ===== STEP 4: Train Models =====
        logger.info("\n🤖 STEP 4: Training Models")
        logger.info("-" * 70)
        
        trainer = AQIModelTrainer()
        results = {}
        
        # Train Random Forest
        logger.info("\n[1/3] Training Random Forest...")
        rf_model = trainer.train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)
        trainer.save_model(rf_model, 'random_forest', str(model_dir))
        
        # Train XGBoost
        logger.info("\n[2/3] Training XGBoost...")
        xgb_model = trainer.train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
        trainer.save_model(xgb_model, 'xgboost', str(model_dir))
        
        # Train LSTM (if TensorFlow available)
        logger.info("\n[3/3] Training LSTM...")
        try:
            # Prepare sequences for LSTM
            seq_length = 24
            X_train_seq, y_train_seq = prepare_sequences(X_train_scaled, y_train, seq_length)
            X_val_seq, y_val_seq = prepare_sequences(X_val_scaled, y_val, seq_length)
            X_test_seq, y_test_seq = prepare_sequences(X_test_scaled, y_test, seq_length)
            
            logger.info(f"   LSTM sequence shape: {X_train_seq.shape}")
            
            lstm_model = trainer.train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            if lstm_model is not None:
                trainer.save_model(lstm_model, 'lstm', str(model_dir))
        except Exception as e:
            logger.warning(f"⚠️  LSTM training skipped: {str(e)}")
            lstm_model = None
        
        # ===== STEP 5: Evaluate Models =====
        logger.info("\n📈 STEP 5: Evaluating Models")
        logger.info("-" * 70)
        
        # Evaluate Random Forest
        logger.info("\n[1/3] Evaluating Random Forest...")
        rf_pred_test = rf_model.predict(X_test_scaled)
        rf_metrics = calculate_metrics(y_test, rf_pred_test)
        results['Random Forest'] = rf_metrics
        
        plot_predictions(y_test, rf_pred_test, 
                        'Random Forest Model',
                        str(results_dir / 'rf_predictions.png'))
        
        # Evaluate XGBoost
        logger.info("\n[2/3] Evaluating XGBoost...")
        xgb_pred_test = xgb_model.predict(X_test_scaled)
        xgb_metrics = calculate_metrics(y_test, xgb_pred_test)
        results['XGBoost'] = xgb_metrics
        
        plot_predictions(y_test, xgb_pred_test,
                        'XGBoost Model',
                        str(results_dir / 'xgb_predictions.png'))
        
        # Evaluate LSTM
        if lstm_model is not None:
            logger.info("\n[3/3] Evaluating LSTM...")
            lstm_pred_test = lstm_model.predict(X_test_seq, verbose=0)
            lstm_metrics = calculate_metrics(y_test_seq, lstm_pred_test.flatten())
            results['LSTM'] = lstm_metrics
            
            plot_predictions(y_test_seq, lstm_pred_test.flatten(),
                            'LSTM Model',
                            str(results_dir / 'lstm_predictions.png'))
        
        # ===== STEP 6: Compare Models =====
        logger.info("\n🏆 STEP 6: Comparing Models")
        logger.info("-" * 70)
        
        comparison_df = compare_models(results)
        
        # Save comparison
        comparison_df.to_csv(results_dir / 'model_comparison.csv')
        logger.info(f"✅ Model comparison saved to {results_dir / 'model_comparison.csv'}")
        
        # Plot comparison
        plot_model_comparison(comparison_df, str(results_dir / 'model_comparison.png'))
        
        # ===== STEP 7: Feature Importance =====
        logger.info("\n🔍 STEP 7: Analyzing Feature Importance")
        logger.info("-" * 70)
        
        # Get feature importance from XGBoost (usually best model)
        importance_dict = trainer.get_feature_importance(xgb_model, feature_cols)
        
        if importance_dict:
            importance_df = pd.DataFrame([
                {'feature': k, 'importance': v} 
                for k, v in importance_dict.items()
            ])
            
            importance_df.to_csv(results_dir / 'feature_importance.csv', index=False)
            logger.info(f"✅ Feature importance saved to {results_dir / 'feature_importance.csv'}")
            
            plot_feature_importance(importance_df, str(results_dir / 'feature_importance.png'))
        
        # ===== STEP 8: 24h Prediction =====
        logger.info("\n🔮 STEP 8: Generating 24-Hour Predictions")
        logger.info("-" * 70)
        
        # Use best model (usually XGBoost)
        best_model = xgb_model
        
        # Get last available data with features
        last_data = df_featured.tail(100)  # Use last 100 rows for context
        
        # Generate predictions
        predictions_24h = predict_next_24h(best_model, last_data, scaler, feature_cols)
        
        # Save predictions
        predictions_24h.to_csv(results_dir / '24h_predictions.csv', index=False)
        logger.info(f"✅ 24-hour predictions saved to {results_dir / '24h_predictions.csv'}")
        
        # Visualize predictions
        visualize_predictions(predictions_24h, str(results_dir / '24h_forecast.png'))
        
        # Display sample predictions
        logger.info("\n📋 Sample 24-Hour Forecast:")
        logger.info("-" * 70)
        logger.info("\n" + predictions_24h.head(10).to_string(index=False))
        logger.info("...")
        
        # ===== FINAL SUMMARY =====
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info("\n📁 Generated Files:")
        logger.info(f"   Models: {model_dir}/")
        logger.info(f"   Results: {results_dir}/")
        logger.info(f"   - Evaluation plots for each model")
        logger.info(f"   - Model comparison table and plot")
        logger.info(f"   - Feature importance analysis")
        logger.info(f"   - 24-hour AQI forecast")
        logger.info("\n🎯 Best Model Performance:")
        best_model_name = comparison_df.index[0]
        best_metrics = comparison_df.iloc[0]
        logger.info(f"   Model: {best_model_name}")
        for metric, value in best_metrics.items():
            logger.info(f"   {metric}: {value}")
        logger.info("\n" + "="*70 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
