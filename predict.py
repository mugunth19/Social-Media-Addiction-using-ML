from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the trained model, scaler, and vectorizer
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
dv = joblib.load("dict_vectorizer.pkl")

class PredictionInput(BaseModel):
    age: int
    gender: str
    academic_level: str
    avg_daily_usage_hours: float
    most_used_platform: str
    sleep_hours_per_night: int
    mental_health_score: int
    conflicts_over_social_media: int
    affects_academic_performance: str
    relationship_status: str

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    addiction_status: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionInput):
    """Predict social media addiction status"""
    try:
        input_dict = data.dict()
        X_features = dv.transform([input_dict])
        X_scaled = scaler.transform(X_features)
        
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        addiction_status = "Addicted" if prediction == 1 else "Not Addicted"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            addiction_status=addiction_status
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
