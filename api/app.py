from contextlib import asynccontextmanager
import pickle
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import CONFIG

logging.basicConfig(level=logging.INFO)

# Global holders for model artifacts
model = None
vectorizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup, release on shutdown."""
    global model, vectorizer
    cfg = CONFIG
    try:
        with open(cfg["model"]["model_path"], "rb") as f:
            model = pickle.load(f)
        with open(cfg["transformation"]["vectorizer_path"], "rb") as f:
            vectorizer = pickle.load(f)
        logging.info("Model and vectorizer loaded successfully")
    except FileNotFoundError as e:
        logging.error(f"Artifact not found: {e}. Run main.py to train first.")
        raise
    yield
    # Cleanup on shutdown (nothing needed for pickle objects)
    model = None
    vectorizer = None


app = FastAPI(title="Fake Job Detection API", lifespan=lifespan)


# Request schema
class JobRequest(BaseModel):
    description: str = Field(
        min_length=CONFIG["api"]["min_description_length"],
        max_length=CONFIG["api"]["max_description_length"]
    )


# Response schema
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