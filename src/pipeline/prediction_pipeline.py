import pickle


class PredictionPipeline:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.vectorizer_path = "artifacts/vectorizer.pkl"

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

        # Load model
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)

        with open(self.vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        transformed = vectorizer.transform([text])

        prob = model.predict_proba(transformed)[0][1]

        if prob > 0.6:
            label = "Fake Job"
            reason = "Suspicious wording detected"
        else:
            label = "Real Job"
            reason = "Professional job-related content"

        confidence = prob if label == "Fake Job" else (1 - prob)

        return label, confidence, reason