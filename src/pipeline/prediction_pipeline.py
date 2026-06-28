import pickle
from src.config import CONFIG


class PredictionPipeline:
    def __init__(self):
        cfg = CONFIG
        # Load once at init — not on every prediction call
        with open(cfg["model"]["model_path"], "rb") as f:
            self.model = pickle.load(f)
        with open(cfg["transformation"]["vectorizer_path"], "rb") as f:
            self.vectorizer = pickle.load(f)

    # 🔹 Simple job check
    def is_job_text(self, text):
        keywords = ["job", "hiring", "experience", "developer", "engineer", "analyst"]
        return any(word in text.lower() for word in keywords)

    def predict(self, text):

        # ❌ Invalid input
        if len(text.strip()) < 5:
            return "Invalid Input ❌", 0.0, "Too short input"

        # ❌ Not job related
        if not self.is_job_text(text):
            return "Not a Job Description ❌", 0.0, "No job-related keywords found"

        # Use cached model and vectorizer
        transformed = self.vectorizer.transform([text])
        prob = self.model.predict_proba(transformed)[0][1]

        if prob > 0.6:
            label = "Fake Job"
            reason = "Suspicious wording detected"
        else:
            label = "Real Job"
            reason = "Professional job-related content"

        confidence = prob if label == "Fake Job" else (1 - prob)

        return label, confidence, reason