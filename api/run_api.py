"""Script để run API"""
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting AQI Prediction API...")
    print("📖 Docs: http://localhost:8000/docs")

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)