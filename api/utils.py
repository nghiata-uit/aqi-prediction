"""
Utility functions cho API
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict


def get_aqi_category(aqi: float) -> str:
    """Chuyển AQI number thành category"""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def calculate_confidence(predictions: List[float]) -> List[float]:
    """Tính confidence score cho predictions"""
    confidences = []
    for i, pred in enumerate(predictions):
        if i == 0:
            confidences.append(0.95)
        else:
            base_conf = 0.95 - (i * 0.01)
            variance_penalty = min(0.2, abs(pred - predictions[i - 1]) / 100)
            confidences.append(max(0.5, base_conf - variance_penalty))
    return confidences


def create_prediction_dataframe(predictions: List[float], start_time: datetime, hours_ahead: int) -> List[Dict]:
    """Tạo structured predictions"""
    confidences = calculate_confidence(predictions)

    results = []
    for i, (pred, conf) in enumerate(zip(predictions[:hours_ahead], confidences[:hours_ahead])):
        timestamp = start_time + timedelta(hours=i + 1)
        results.append({
            "timestamp": timestamp,
            "hour_ahead": i + 1,
            "predicted_aqi": round(float(pred), 2),
            "aqi_category": get_aqi_category(pred),
            "confidence": round(conf, 2)
        })

    return results