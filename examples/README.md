# Global Spatial Model - Examples

Thư mục này chứa các ví dụ minh họa cách sử dụng Global Spatial Model để dự đoán AQI.

## 📁 Files

- **global_spatial_example.py**: Script Python chứa 4 ví dụ đầy đủ về training, prediction, và API usage

## 🚀 Quick Start

### Chạy tất cả examples:

```bash
python examples/global_spatial_example.py
```

### Chạy từng example riêng lẻ:

```bash
# Example 1: Training model
python examples/global_spatial_example.py --example 1

# Example 2: Load và predict
python examples/global_spatial_example.py --example 2

# Example 3: API usage guide
python examples/global_spatial_example.py --example 3

# Example 4: Batch predictions
python examples/global_spatial_example.py --example 4
```

## 📚 Examples Overview

### Example 1: Training Global Spatial Model

Minh họa toàn bộ quy trình training:
- Load và preprocess data
- Engineer features với spatial features (lat/lon)
- Train XGBoost model
- Evaluate performance
- Save artifacts

**Output:**
```
Performance Metrics:
MAE:  0.0042
RMSE: 0.0357
R²:   0.9984
```

### Example 2: Loading Model và Predicting

Cho thấy cách:
- Load trained model và artifacts
- Prepare input data cho location mới
- Combine với historical data cho lag/rolling features
- Predict AQI và interpret kết quả

**Sample Input:**
```python
{
    'datetime': '2020-11-25 12:00:00',
    'lat': 106.7075,      # Ho Chi Minh City
    'lon': 10.804,
    'co': 700.0,
    'no': 8.0,
    'no2': 22.0,
    'o3': 60.0,
    'so2': 20.0,
    'pm2_5': 25.0,
    'pm10': 60.0,
    'nh3': 9.0
}
```

**Output:**
```
🎯 Predicted AQI: 3.02
📊 AQI Level: Moderate (Kém)
```

### Example 3: Using API for Prediction

Hướng dẫn đầy đủ về cách sử dụng FastAPI endpoint:
- Start server
- Health check
- Make prediction requests với curl
- Sử dụng Python requests library

**API Request:**
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

**API Response:**
```json
{
  "predicted_aqi": 3.02,
  "model_name": "xgboost_global"
}
```

### Example 4: Batch Predictions for Multiple Locations

Minh họa cách predict cho nhiều locations cùng lúc:
- Ho Chi Minh City
- Hanoi
- Da Nang

**Output:**
```
Results:
------------------------------------------------------------
Ho Chi Minh City     | AQI: 3.02
Hanoi                | AQI: 2.98
Da Nang              | AQI: 2.85
------------------------------------------------------------
```

## 📋 Prerequisites

Trước khi chạy examples, cần:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train model (nếu chưa có artifacts):**
   ```bash
   python scripts/train_global_model.py
   ```

3. **Verify artifacts exist:**
   ```bash
   ls -la models/
   # Should see:
   # - xgboost_global.pkl
   # - feature_columns_global.pkl
   # - spatial_scaler.pkl
   ```

## 🔍 Understanding the Code

### Spatial Features

Model sử dụng lat/lon được standardized:

```python
# During training
spatial_scaler = StandardScaler()
df[['lat_scaled', 'lon_scaled']] = spatial_scaler.fit_transform(df[['lat', 'lon']])

# During inference
df[['lat_scaled', 'lon_scaled']] = spatial_scaler.transform(df[['lat', 'lon']])
```

### Feature Engineering Pipeline

1. **Spatial features**: lat/lon scaled
2. **Time features**: hour, day_of_week, month, cyclical encodings
3. **Lag features**: 1h, 2h, 3h, 6h, 12h, 24h
4. **Rolling features**: mean, std, min, max for 6h, 12h, 24h windows

Total: **163 features**

### Prediction Flow

```
Input (lat, lon, pollutants)
    ↓
Combine with historical data
    ↓
Engineer features (spatial scaling + time + lag + rolling)
    ↓
Select features in training order
    ↓
Predict with XGBoost model
    ↓
Output: AQI value
```

## 💡 Tips

1. **Historical data is required**: Model cần historical data để tính lag và rolling features. Nếu không có, các features này sẽ là NaN.

2. **Feature order matters**: Phải sử dụng features theo đúng thứ tự đã train (lưu trong feature_columns_global.pkl).

3. **Spatial scaler**: Phải dùng cùng spatial_scaler đã fit khi training để transform lat/lon.

4. **Time-based data**: Input datetime nên nằm sau historical data để tính lag features chính xác.

## 🐛 Troubleshooting

**Error: "Artifacts not found"**
- Solution: Chạy `python scripts/train_global_model.py` để train model

**Error: "Historical data not found"**
- Solution: Đảm bảo file `data/sample_data.csv` tồn tại

**Prediction returns NaN**
- Cause: Thiếu historical data cho lag/rolling features
- Solution: Cung cấp ít nhất 24 giờ historical data

## 📚 Additional Resources

- [README.md](../README.md) - Project documentation
- [API Documentation](../api/README.md) - API endpoints detail
- [Training Script](../scripts/train_global_model.py) - Training implementation

---

🌍 Global Spatial Model cho phép predict AQI cho bất kỳ location nào với lat/lon!
