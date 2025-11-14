# 🌍 AQI Prediction System

Hệ thống dự đoán chỉ số AQI (Air Quality Index) cho 24 giờ tiếp theo sử dụng Machine Learning.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red)](https://xgboost.readthedocs.io/)

## 📊 Giới thiệu

Dự án này xây dựng các models Machine Learning để dự đoán chỉ số chất lượng không khí (AQI) dựa trên dữ liệu lịch sử về các chất ô nhiễm như CO, NO, NO2, O3, SO2, PM2.5, PM10, NH3.

### Features chính:

- ✅ **3 Machine Learning models**: Random Forest, XGBoost, LSTM
- ✅ **Feature engineering tự động** với lag và rolling statistics
- ✅ **Dự đoán 24h trước** với độ chính xác cao
- ✅ **Jupyter Notebook** để phân tích và visualization
- ✅ **Visualizations đẹp mắt** với matplotlib và seaborn
- ✅ **Code production-ready** với logging, error handling, type hints

### Kết quả Performance:

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Random Forest | 0.0032 | 0.0219 | 0.9994 |
| XGBoost | 0.1335 | 0.1782 | 0.9598 |

## 📁 Cấu trúc Project

```
aqi-prediction/
├── data/
│   ├── .gitkeep
│   └── sample_data.csv          # Dữ liệu mẫu (588 rows, hourly data)
├── notebooks/
│   └── AQI_Prediction_Analysis.ipynb  # Jupyter notebook phân tích
├── models/
│   ├── .gitkeep
│   ├── random_forest.pkl        # Random Forest model
│   ├── xgboost.pkl              # XGBoost model
│   ├── scaler.pkl               # StandardScaler
│   ├── xgboost_global.pkl       # Global XGBoost với spatial features
│   ├── feature_columns_global.pkl  # Feature names cho global model
│   └── spatial_scaler.pkl       # Spatial scaler cho lat/lon
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Xử lý và làm sạch dữ liệu
│   ├── feature_engineering.py   # Tạo features (bao gồm spatial features)
│   ├── model_training.py        # Train models
│   ├── model_evaluation.py      # Đánh giá models
│   └── prediction.py            # Dự đoán 24h
├── scripts/
│   └── train_global_model.py    # Train global spatial model
├── api/
│   ├── app.py                   # FastAPI application
│   └── dependencies.py          # Load và quản lý artifacts
├── results/
│   ├── .gitkeep
│   ├── rf_predictions.png       # Random Forest evaluation
│   ├── xgb_predictions.png      # XGBoost evaluation
│   ├── model_comparison.png     # So sánh models
│   ├── feature_importance.png   # Feature importance
│   └── 24h_forecast.png         # Dự đoán 24h
├── requirements.txt             # Dependencies
├── .gitignore
├── README.md
└── main.py                      # Script chính
```

## 🚀 Installation

### 1. Clone repository

```bash
git clone https://github.com/nghiata-uit/aqi-prediction.git
cd aqi-prediction
```

### 2. Tạo virtual environment (khuyến nghị)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Quick Start - Chạy toàn bộ pipeline

```bash
python main.py
```

Pipeline sẽ tự động:
1. Load và preprocess dữ liệu
2. Tạo features (time, lag, rolling)
3. Split train/validation/test sets
4. Train Random Forest và XGBoost models
5. Evaluate và compare models
6. Generate 24-hour predictions
7. Save models và visualizations

### Output

Sau khi chạy, bạn sẽ có:

```
models/
├── random_forest.pkl    # Trained Random Forest
├── xgboost.pkl          # Trained XGBoost
└── scaler.pkl           # Feature scaler

results/
├── rf_predictions.png       # RF evaluation plots
├── xgb_predictions.png      # XGBoost evaluation plots
├── model_comparison.png     # Model comparison
├── model_comparison.csv     # Metrics table
├── feature_importance.png   # Top features
├── feature_importance.csv   # All features
├── 24h_forecast.png         # 24h prediction visualization
└── 24h_predictions.csv      # 24h predictions data
```

### Sử dụng trong code

```python
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.prediction import predict_next_24h
import joblib

# Load data và preprocess
df = preprocess_data('data/sample_data.csv')

# Feature engineering
df_featured = engineer_features(df)

# Load trained model
model = joblib.load('models/xgboost.pkl')
scaler = joblib.load('models/scaler.pkl')

# Dự đoán 24h
predictions = predict_next_24h(model, df_featured.tail(100), scaler, feature_cols)
print(predictions)
```

## 📈 Models

### 1. Random Forest Regressor

- **n_estimators**: 200
- **max_depth**: 15
- **Ưu điểm**: Dễ interpret, robust, feature importance
- **Kết quả**: R² = 0.9994

### 2. XGBoost Regressor

- **n_estimators**: 300
- **max_depth**: 7
- **learning_rate**: 0.05
- **Ưu điểm**: Best performance, production-ready, handle missing values
- **Kết quả**: R² = 0.9598

### 3. LSTM (Optional)

- **Architecture**: LSTM(128) → Dropout → LSTM(64) → Dense
- **Ưu điểm**: Deep learning cho time series
- **Note**: Cần cài đặt TensorFlow

## 🔍 Feature Engineering

Hệ thống tự động tạo 161 features từ dữ liệu gốc:

### Time Features (9 features)
- hour, day_of_week, day, month, is_weekend
- Cyclical encoding: hour_sin, hour_cos, dow_sin, dow_cos

### Lag Features (48 features)
- Giá trị của 8 pollutants ở các thời điểm trước đó
- Lags: 1h, 2h, 3h, 6h, 12h, 24h

### Rolling Statistics (96 features)
- Mean, Std, Min, Max cho 8 pollutants
- Windows: 6h, 12h, 24h

### Original Features (8 features)
- co, no, no2, o3, so2, pm2_5, pm10, nh3

## 🔮 24h Prediction Example

```python
from src.prediction import load_model_and_predict

predictions = load_model_and_predict(
    model_path='models/xgboost.pkl',
    scaler_path='models/scaler.pkl',
    data_path='data/sample_data.csv'
)

print(predictions.head())
```

Output:
```
 hour           timestamp  predicted_aqi
    1 2020-11-25 12:00:00           4.65
    2 2020-11-25 13:00:00           4.65
    3 2020-11-25 14:00:00           4.65
    4 2020-11-25 15:00:00           4.65
    5 2020-11-25 16:00:00           4.65
```

## 📊 Data

### Sample Data

Dữ liệu mẫu bao gồm:
- **Thời gian**: 2020-11-01 00:00:00 đến 2020-11-25 11:00:00 (588 hours)
- **Location**: lon=10.804, lat=106.7075 (khu vực TP.HCM)
- **Target**: AQI (2-5: Good to Poor)
- **Features**: 8 pollutants

### Pollutants

| Pollutant | Range | Unit | Description |
|-----------|-------|------|-------------|
| CO | 400-1200 | μg/m³ | Carbon Monoxide |
| NO | 1-25 | μg/m³ | Nitrogen Monoxide |
| NO2 | 10-35 | μg/m³ | Nitrogen Dioxide |
| O3 | 1-100 | μg/m³ | Ozone |
| SO2 | 15-40 | μg/m³ | Sulfur Dioxide |
| PM2.5 | 15-50 | μg/m³ | Fine Particulate Matter |
| PM10 | 30-80 | μg/m³ | Coarse Particulate Matter |
| NH3 | 3-15 | μg/m³ | Ammonia |

### AQI Levels

| AQI | Level | Description |
|-----|-------|-------------|
| 1-2 | Good | Chất lượng không khí tốt |
| 2-3 | Fair | Chất lượng không khí trung bình |
| 3-4 | Moderate | Chất lượng không khí kém |
| 4-5 | Poor | Chất lượng không khí xấu |

## 🛠️ Development

### Thêm model mới

1. Implement trong `src/model_training.py`:

```python
def train_new_model(self, X_train, y_train, X_val, y_val):
    model = YourModel(...)
    model.fit(X_train, y_train)
    return model
```

2. Thêm vào pipeline trong `main.py`
3. Update evaluation và comparison

### Thêm features mới

Implement trong `src/feature_engineering.py`:

```python
def create_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    # Your feature engineering logic
    return df
```

### Testing

```bash
# Test data preprocessing
python -c "from src.data_preprocessing import preprocess_data; df = preprocess_data('data/sample_data.csv'); print(df.shape)"

# Test feature engineering
python -c "from src.feature_engineering import engineer_features; from src.data_preprocessing import preprocess_data; df = preprocess_data('data/sample_data.csv'); df_feat = engineer_features(df); print(df_feat.shape)"
```

## 📝 Notes

### Best Practices

- ✅ Time series dùng **time-based split** (không shuffle)
- ✅ Lag features là **quan trọng nhất** cho time series
- ✅ XGBoost thường cho **kết quả tốt nhất**
- ✅ LSTM cần **sequence preparation** (3D input)
- ✅ Dùng **StandardScaler** cho numerical features
- ✅ Save models và scaler để **reuse**

### Common Issues

**Q: LSTM không train được?**
A: Cần cài đặt TensorFlow: `pip install tensorflow>=2.10.0`

**Q: Kết quả khác nhau mỗi lần chạy?**
A: Set random seed trong code (đã implement)

**Q: Muốn dùng dữ liệu thực?**
A: Replace `data/sample_data.csv` với data của bạn (cùng format)

## 🌍 Global Spatial Model (Method A)

### Giới thiệu

Project hiện đã implement **Method A**: một global XGBoost model với spatial features (lat, lon). Model này có thể dự đoán AQI cho bất kỳ vị trí địa lý nào, không chỉ giới hạn ở một địa điểm cụ thể.

### Training Global Model

Để train global model với spatial features:

```bash
python scripts/train_global_model.py
```

Script này sẽ:
- Load dữ liệu từ `data/sample_data.csv`
- Tạo time features, lag features, rolling features và **spatial features** (lat, lon chuẩn hóa)
- Split dữ liệu theo thời gian (70% train, 30% test)
- Train XGBoost model
- Save artifacts vào `models/`:
  - `xgboost_global.pkl` - Model đã train
  - `feature_columns_global.pkl` - Danh sách feature names
  - `spatial_scaler.pkl` - StandardScaler cho lat/lon

### Model Artifacts

Sau khi train, các artifacts được lưu trong `models/`:

```
models/
├── xgboost_global.pkl           # Trained XGBoost model
├── feature_columns_global.pkl   # Feature names (163 features)
└── spatial_scaler.pkl           # Spatial scaler cho lat/lon
```

### Sử dụng API

API cung cấp endpoints để predict AQI dựa trên pollutants và vị trí địa lý.

#### Khởi động API:

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Hoặc:

```bash
python api/app.py
```

API sẽ tự động load các artifacts khi startup.

#### API Endpoints:

- **GET /** - Root endpoint với thông tin API
- **GET /health** - Health check, kiểm tra artifacts đã load
- **GET /model-info** - Thông tin chi tiết về model và features
- **POST /predict** - Predict AQI từ input data

#### Example Request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 106.7075,
    "lon": 10.804,
    "co": 704.51,
    "no": 8.31,
    "no2": 21.89,
    "o3": 63.35,
    "so2": 21.33,
    "pm2_5": 25.13,
    "pm10": 63.95,
    "nh3": 9.5
  }'
```

#### Example Response:

```json
{
  "predicted_aqi": 3.05,
  "lat_scaled": 0.0,
  "lon_scaled": 0.0,
  "model_name": "xgboost_global"
}
```

### Kiến trúc

```
┌─────────────────┐
│   Input Data    │
│ (lat, lon, CO,  │
│  NO, NO2, ...)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Spatial Scaler  │
│ (lat/lon → std) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Feature Engineer │
│ (time, lag,     │
│  rolling feat.) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ XGBoost Global  │
│     Model       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Predicted AQI  │
└─────────────────┘
```

### Features

Global model sử dụng **163 features**, bao gồm:
- **2 Spatial features**: lat_scaled, lon_scaled
- **9 Time features**: hour, day_of_week, day, month, is_weekend, cyclical encodings
- **48 Lag features**: 8 pollutants × 6 lags (1h, 2h, 3h, 6h, 12h, 24h)
- **96 Rolling features**: 8 pollutants × 3 windows (6h, 12h, 24h) × 4 stats (mean, std, min, max)
- **8 Original features**: co, no, no2, o3, so2, pm2_5, pm10, nh3

## 🎯 Future Improvements

- [ ] Add more models (LightGBM, CatBoost)
- [ ] Implement hyperparameter tuning
- [ ] Add real-time prediction API
- [ ] Deploy with Docker
- [ ] Add unit tests
- [ ] Integrate with real AQI APIs
- [ ] Add data validation với Great Expectations
- [ ] Implement MLOps pipeline

## 📚 References

- [WHO Air Quality Guidelines](https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health)
- [EPA AQI Basics](https://www.airnow.gov/aqi/aqi-basics/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Time Series](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html)

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

## 👨‍💻 Author

**nghiata-uit**

- GitHub: [@nghiata-uit](https://github.com/nghiata-uit)
- Repository: [aqi-prediction](https://github.com/nghiata-uit/aqi-prediction)

---

⭐ Nếu project hữu ích, hãy star repository này!
