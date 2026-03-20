from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Fake Job Detection API")

# load model
model = pickle.load(open("artifacts/model.pkl", "rb"))
vectorizer = pickle.load(open("artifacts/vectorizer.pkl", "rb"))

# request schema
class JobRequest(BaseModel):
    description: str = Field(min_length=10, max_length=2000)

# response schema
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def home():
    return {"status": "API is running 🚀"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: JobRequest):
    try:
        text = request.description

        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1]

        result = "Fake Job" if pred == 1 else "Real Job"

        return PredictionResponse(
            prediction=result,
            confidence=round(float(prob), 3)
        )

    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail="Prediction error")