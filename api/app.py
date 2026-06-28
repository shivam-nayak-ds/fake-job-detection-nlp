from contextlib import asynccontextmanager
import pickle
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import CONFIG
from src.explainability import SHAPExplainer

logging.basicConfig(level=logging.INFO)

# Global holders for model artifacts
model      = None
vectorizer = None
explainer  = None        # SHAP TreeExplainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup, release on shutdown."""
    global model, vectorizer, explainer
    cfg = CONFIG
    try:
        with open(cfg["model"]["model_path"], "rb") as f:
            model = pickle.load(f)
        with open(cfg["transformation"]["vectorizer_path"], "rb") as f:
            vectorizer = pickle.load(f)

        # Build SHAP explainer once at startup (expensive to create each request)
        explainer = SHAPExplainer(model, vectorizer)

        logging.info("Model, vectorizer, and SHAP explainer loaded successfully")
    except FileNotFoundError as e:
        logging.error(f"Artifact not found: {e}. Run main.py to train first.")
        raise
    yield
    model = None
    vectorizer = None
    explainer  = None


app = FastAPI(
    title       = "Fake Job Detection API",
    description = "Detects fraudulent job postings using XGBoost + TF-IDF + SHAP explainability",
    version     = "2.0.0",
    lifespan    = lifespan
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class JobRequest(BaseModel):
    description: str = Field(
        min_length = CONFIG["api"]["min_description_length"],
        max_length  = CONFIG["api"]["max_description_length"]
    )


class PredictionResponse(BaseModel):
    prediction : str
    confidence : float


class FeatureImportance(BaseModel):
    word       : str
    shap_value : float
    direction  : str   # "fake" | "real"


class ExplainResponse(BaseModel):
    prediction   : str
    confidence   : float
    top_features : list[FeatureImportance]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "API is running", "version": "2.0.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: JobRequest):
    """Predict whether a job posting is real or fake."""
    try:
        text = request.description
        vec  = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1]

        return PredictionResponse(
            prediction = "Fake Job" if pred == 1 else "Real Job",
            confidence = round(float(prob), 3)
        )
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail="Prediction error")


@app.post("/explain", response_model=ExplainResponse)
def explain(request: JobRequest):
    """
    Predict + explain: returns the top words that influenced the decision.

    SHAP values:
    - Positive → word pushes prediction towards 'Fake Job'
    - Negative → word pushes prediction towards 'Real Job'
    """
    try:
        text = request.description
        vec  = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1]

        top_features = explainer.explain(text, top_n=10)

        return ExplainResponse(
            prediction   = "Fake Job" if pred == 1 else "Real Job",
            confidence   = round(float(prob), 3),
            top_features = top_features
        )
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")