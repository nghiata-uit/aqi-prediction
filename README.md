# 🌍 AQI Prediction System - 24h Forecast

Complete Air Quality Index (AQI) prediction system that trains multiple models per location and provides real-time predictions via FastAPI.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red)](https://xgboost.readthedocs.io/)
[![Prophet](https://img.shields.io/badge/Prophet-1.1-blue)](https://facebook.github.io/prophet/)

## 📊 Overview

This system predicts Air Quality Index (AQI) 24 hours ahead using ensemble machine learning models. It features:

- **5 Model Types**: LSTM, GRU, Prophet, XGBoost, Random Forest, and Ensemble
- **Location-Based Training**: Trains separate models for each unique location (lat, lon)
- **Automatic Model Selection**: Selects the best performing model per location
- **REST API**: FastAPI endpoints for predictions and model management
- **Production Ready**: Docker support, comprehensive error handling, logging

### Model Performance:

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Random Forest | 0.0032 | 0.0219 | 0.9994 |
| XGBoost | 0.1335 | 0.1782 | 0.9598 |
| Prophet | 0.2800 | 0.3255 | 0.8656 |
| LSTM | 0.6647 | 0.8124 | 0.2277 |
| GRU | 0.7093 | 0.8745 | 0.1052 |
| Ensemble | 0.0043 | 0.0223 | 0.9994 |

## 📁 Project Structure

```
aqi-prediction/
├── api/                         # FastAPI application
│   ├── __init__.py
│   ├── main.py                 # FastAPI app with endpoints
│   ├── predict.py              # Prediction logic
│   ├── schemas.py              # Pydantic models
│   └── utils.py                # Utility functions
├── src/
│   ├── models/                 # Model implementations
│   │   ├── lstm_model.py
│   │   ├── gru_model.py
│   │   ├── prophet_model.py
│   │   ├── xgboost_model.py
│   │   └── ensemble_model.py
│   ├── data_preprocessing.py   # Data preprocessing
│   ├── feature_engineering.py  # Feature creation
│   ├── model_training.py       # Legacy training code
│   ├── model_evaluation.py     # Evaluation metrics
│   └── prediction.py           # Prediction utilities
├── data/
│   └── sample_data.csv         # Training data
├── models/                     # Saved models directory
│   ├── lat_X_lon_Y_best.pkl    # Best model per location
│   ├── lat_X_lon_Y_metrics.json # Performance metrics
│   └── scaler.pkl              # Feature scaler
├── notebooks/
│   └── AQI_Prediction_Analysis.ipynb
├── results/                    # Evaluation results
├── train_models.py             # Main training script
├── main.py                     # Legacy training script
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## 🚀 Installation

### 1. Clone repository

```bash
git clone https://github.com/nghiata-uit/aqi-prediction.git
cd aqi-prediction
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env with your settings
```

## 💻 Usage

### Training Models

#### Option 1: Using the Shell Script (Recommended)

The shell script can train models for all CSV files in the data directory or specific files:

```bash
# Train ALL CSV files in data/ directory (default behavior)
./train_all_locations.sh

# Train all CSV files explicitly
./train_all_locations.sh --all

# Train a specific CSV file
./train_all_locations.sh data/my_data.csv

# Specify custom data directory containing multiple CSV files
./train_all_locations.sh -d my_data_folder/

# Specify output directory
./train_all_locations.sh -o my_models/

# See all options
./train_all_locations.sh --help
```

The script will:
- **Automatically find all CSV files** in the data directory
- Check Python dependencies
- Validate data files
- Train models for each file with progress logging
- Save separate log files for each dataset
- Show comprehensive summary of all trained models

**Key Features:**
- ✅ Processes multiple CSV files automatically
- ✅ Individual log files per dataset
- ✅ Progress tracking across all files
- ✅ Summary of successful/failed trainings
- ✅ Total model count across all datasets

#### Option 2: Using Python Directly

```bash
python train_models.py --data data/sample_data.csv --output models/
```

Both methods will:
1. Load and preprocess data
2. Group data by unique locations (lat, lon)
3. For each location:
   - Train 6 models: LSTM, GRU, Prophet, XGBoost, Random Forest, Ensemble
   - Evaluate all models
   - Select and save the best model
   - Save performance metrics

### Running the API

Start the FastAPI server:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Or use Docker:

```bash
docker build -t aqi-prediction .
docker run -p 8000:8000 -v $(pwd)/models:/app/models aqi-prediction
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 📡 API Endpoints

### 1. Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": 1
}
```

### 2. Predict AQI (24h ahead)

```bash
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "lat": 10.7828,
  "lon": 106.6953,
  "current_data": {
    "co": 1842.5,
    "no": 34.87,
    "no2": 54.84,
    "o3": 0,
    "so2": 60.56,
    "pm2_5": 67.8,
    "pm10": 81.92,
    "nh3": 10.77
  },
  "datetime": "2022-10-12 20:00:00"
}
```

**Response:**
```json
{
  "location": {
    "lat": 106.7075,
    "lon": 10.804
  },
  "predicted_aqi": 5.0,
  "prediction_for": "2022-10-13 20:00:00",
  "model_used": "RandomForest",
  "confidence_score": 1.0,
  "metrics": {
    "mae": 0.0032,
    "rmse": 0.0219,
    "r2": 0.9994
  }
}
```

### 3. List Available Models

```bash
GET /models
```

**Response:**
```json
{
  "total_locations": 1,
  "models": [
    {
      "location": {
        "lat": 106.7075,
        "lon": 10.804
      },
      "model_type": "RandomForest",
      "metrics": {
        "mae": 0.0032,
        "rmse": 0.0219,
        "r2": 0.9994
      },
      "trained_date": null
    }
  ]
}
```

### 4. Model Comparison

```bash
GET /model-comparison/{lat}/{lon}
```

**Example:**
```bash
GET /model-comparison/106.7075/10.804
```

**Response:**
```json
{
  "location": {
    "lat": 106.7075,
    "lon": 10.804
  },
  "models": {
    "RandomForest": {
      "mae": 0.0032,
      "rmse": 0.0219,
      "r2": 0.9994
    },
    "XGBoost": {
      "mae": 0.1335,
      "rmse": 0.1782,
      "r2": 0.9598
    },
    "Prophet": {
      "mae": 0.28,
      "rmse": 0.3255,
      "r2": 0.8656
    },
    "LSTM": {
      "mae": 0.6647,
      "rmse": 0.8124,
      "r2": 0.2277
    },
    "GRU": {
      "mae": 0.7093,
      "rmse": 0.8745,
      "r2": 0.1052
    },
    "Ensemble": {
      "mae": 0.0043,
      "rmse": 0.0223,
      "r2": 0.9994
    }
  },
  "best_model": "RandomForest"
}
```

## 🔧 API Usage Examples

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 10.7828,
    "lon": 106.6953,
    "current_data": {
      "co": 1842.5,
      "no": 34.87,
      "no2": 54.84,
      "o3": 0,
      "so2": 60.56,
      "pm2_5": 67.8,
      "pm10": 81.92,
      "nh3": 10.77
    }
  }'

# List models
curl http://localhost:8000/models

# Get model comparison
curl http://localhost:8000/model-comparison/106.7075/10.804
```

### Python Examples

```python
import requests

# API endpoint
API_URL = "http://localhost:8000"

# Make prediction
response = requests.post(
    f"{API_URL}/predict",
    json={
        "lat": 10.7828,
        "lon": 106.6953,
        "current_data": {
            "co": 1842.5,
            "no": 34.87,
            "no2": 54.84,
            "o3": 0,
            "so2": 60.56,
            "pm2_5": 67.8,
            "pm10": 81.92,
            "nh3": 10.77
        }
    }
)

prediction = response.json()
print(f"Predicted AQI: {prediction['predicted_aqi']}")
print(f"Model used: {prediction['model_used']}")
print(f"Confidence: {prediction['confidence_score']}")

# List available models
response = requests.get(f"{API_URL}/models")
models = response.json()
print(f"Total locations: {models['total_locations']}")

# Get model comparison
response = requests.get(f"{API_URL}/model-comparison/106.7075/10.804")
comparison = response.json()
print(f"Best model: {comparison['best_model']}")
```

## 📈 Models

### 1. Random Forest Regressor

- **Architecture**: 200 estimators, max depth 15
- **Advantages**: Robust, interpretable, feature importance
- **Performance**: R² = 0.9994, MAE = 0.0032

### 2. XGBoost Regressor

- **Architecture**: 300 estimators, learning rate 0.05
- **Advantages**: Excellent performance, handles missing values
- **Performance**: R² = 0.9598, MAE = 0.1335

### 3. LSTM (Long Short-Term Memory)

- **Architecture**: 2 LSTM layers (128, 64 units) + Dense layers
- **Advantages**: Captures temporal dependencies
- **Performance**: R² = 0.2277, MAE = 0.6647

### 4. GRU (Gated Recurrent Unit)

- **Architecture**: 2 GRU layers (128, 64 units) + Dense layers
- **Advantages**: Faster than LSTM, good for time series
- **Performance**: R² = 0.1052, MAE = 0.7093

### 5. Prophet

- **Architecture**: Facebook's time series forecasting
- **Advantages**: Handles seasonality automatically
- **Performance**: R² = 0.8656, MAE = 0.2800

### 6. Ensemble Model

- **Architecture**: Weighted combination of RF and XGBoost
- **Advantages**: Best of multiple models
- **Performance**: R² = 0.9994, MAE = 0.0043

## 🔍 Feature Engineering

The system automatically creates **165 features** from the raw data:

### Time Features (9 features)
- `hour`, `day_of_week`, `day`, `month`, `is_weekend`
- Cyclical encoding: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`

### Lag Features (48 features)
- Historical values of 8 pollutants at different time points
- Lags: 1h, 2h, 3h, 6h, 12h, 24h

### Rolling Statistics (96 features)
- Mean, Std, Min, Max for 8 pollutants
- Windows: 6h, 12h, 24h

### Original Features (8 features)
- `co`, `no`, `no2`, `o3`, `so2`, `pm2_5`, `pm10`, `nh3`

### Additional Features (4 features)
- Location: `lat`, `lon`
- Target: `aqi`
- Timestamp: `datetime`

## 📊 Data Format

### Input Data Structure

The training data should be a CSV file with the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `datetime` | datetime | Timestamp | 2022-10-12 20:00:00 |
| `lon` | float | Longitude | 106.6953 |
| `lat` | float | Latitude | 10.7828 |
| `co` | float | Carbon Monoxide (μg/m³) | 1842.5 |
| `no` | float | Nitrogen Monoxide (μg/m³) | 34.87 |
| `no2` | float | Nitrogen Dioxide (μg/m³) | 54.84 |
| `o3` | float | Ozone (μg/m³) | 0 |
| `so2` | float | Sulfur Dioxide (μg/m³) | 60.56 |
| `pm2_5` | float | PM2.5 (μg/m³) | 67.8 |
| `pm10` | float | PM10 (μg/m³) | 81.92 |
| `nh3` | float | Ammonia (μg/m³) | 10.77 |
| `aqi` | int | Air Quality Index (1-5) | 5 |

### AQI Levels

| AQI | Level | Description | Color |
|-----|-------|-------------|-------|
| 1-2 | Good | Air quality is satisfactory | 🟢 Green |
| 2-3 | Fair | Air quality is acceptable | 🟡 Yellow |
| 3-4 | Moderate | Air quality is unhealthy for sensitive groups | 🟠 Orange |
| 4-5 | Poor | Air quality is unhealthy | 🔴 Red |

## 🛠️ Development

### Adding New Models

1. Create model file in `src/models/`:

```python
# src/models/my_model.py
def train_my_model(X_train, y_train, X_val, y_val):
    model = MyModel(...)
    model.fit(X_train, y_train)
    return model
```

2. Import in `src/models/__init__.py`

3. Add to training pipeline in `train_models.py`

### Adding New Features

Implement in `src/feature_engineering.py`:

```python
def create_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    # Your feature engineering logic
    df['my_feature'] = ...
    return df
```

### Testing

```bash
# Test data preprocessing
python -c "from src.data_preprocessing import preprocess_data; \
           df = preprocess_data('data/sample_data.csv'); \
           print(df.shape)"

# Test feature engineering
python -c "from src.feature_engineering import engineer_features; \
           from src.data_preprocessing import preprocess_data; \
           df = preprocess_data('data/sample_data.csv'); \
           df_feat = engineer_features(df); \
           print(df_feat.shape)"

# Test API
pytest tests/ -v  # if tests are available
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t aqi-prediction:latest .
```

### Run Container

```bash
# Run with volume mount for models
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name aqi-api \
  aqi-prediction:latest
```

### Check Logs

```bash
docker logs -f aqi-api
```

### Stop Container

```bash
docker stop aqi-api
docker rm aqi-api
```

## 📝 Configuration

Environment variables can be set in `.env` file:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Models Directory
MODELS_DIR=models/

# Logging
LOG_LEVEL=INFO

# CORS Settings
CORS_ORIGINS=*

# Model Training
TRAIN_TEST_SPLIT=0.7,0.15,0.15
RANDOM_SEED=42
```

## 🔒 Security Considerations

- **Input Validation**: All API inputs are validated using Pydantic
- **CORS**: Configure `CORS_ORIGINS` appropriately for production
- **Rate Limiting**: Consider adding rate limiting for production use
- **Authentication**: Add API key authentication if needed
- **Data Privacy**: Ensure location data is handled according to privacy regulations

## 📝 Notes

### Best Practices

- ✅ Use **time-based split** for time series data (no shuffling)
- ✅ **Lag features** are most important for time series
- ✅ **Random Forest** usually gives best results for this dataset
- ✅ LSTM/GRU need **sequence preparation** (3D input)
- ✅ Use **StandardScaler** for numerical features
- ✅ Save models and scaler for **reuse**

### Common Issues

**Q: LSTM/GRU training fails?**
A: Ensure TensorFlow is installed: `pip install tensorflow>=2.10.0`

**Q: Prophet training is slow?**
A: Prophet uses MCMC sampling which can be slow. Consider using fewer data points or adjusting Prophet parameters.

**Q: Results differ between runs?**
A: Random seeds are set for reproducibility, but small variations can occur with neural networks.

**Q: Want to use real-time data?**
A: Integrate with air quality APIs (e.g., OpenWeatherMap, IQAir) to fetch real-time pollutant data.

**Q: Model not found for location?**
A: The system finds the closest trained model. Ensure models are trained for locations near your query point.

## 🎯 Future Improvements

- [ ] Add more sophisticated models (LightGBM, CatBoost, Transformers)
- [ ] Implement hyperparameter tuning with Optuna/Ray Tune
- [ ] Add real-time data ingestion pipeline
- [ ] Implement model versioning and A/B testing
- [ ] Add comprehensive unit and integration tests
- [ ] Integrate with real AQI data sources
- [ ] Add data validation with Great Expectations
- [ ] Implement MLOps pipeline (MLflow, Kubeflow)
- [ ] Add monitoring and alerting (Prometheus, Grafana)
- [ ] Deploy to cloud (AWS, GCP, Azure)
- [ ] Add frontend dashboard for visualizations
- [ ] Implement batch prediction endpoints
- [ ] Add model explainability (SHAP, LIME)
- [ ] Support for multiple prediction horizons (1h, 6h, 12h, 24h)

## 📚 References

- [WHO Air Quality Guidelines](https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health)
- [EPA AQI Basics](https://www.airnow.gov/aqi/aqi-basics/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Time Series](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 👨‍💻 Author

**nghiata-uit**

- GitHub: [@nghiata-uit](https://github.com/nghiata-uit)
- Repository: [aqi-prediction](https://github.com/nghiata-uit/aqi-prediction)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

⭐ If this project helped you, please star the repository!
