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

    def predict(self, text: str):
        """
        Predict whether a job description is Real or Fake.

        Returns
        -------
        label      : "Fake Job" | "Real Job"
        confidence : float (0-1), probability of the predicted class
        reason     : short human-readable explanation
        """
        if len(text.strip()) < 10:
            return "Invalid Input", 0.0, "Description too short (min 10 characters)"

        transformed = self.vectorizer.transform([text])
        prob = self.model.predict_proba(transformed)[0][1]  # P(Fake)

        if prob > 0.5:
            label      = "Fake Job"
            reason     = "Suspicious patterns detected in the job description"
            confidence = prob
        else:
            label      = "Real Job"
            reason     = "Professional job-related content detected"
            confidence = 1 - prob

        return label, round(confidence, 3), reason