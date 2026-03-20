from src.pipeline.prediction_pipeline import PredictionPipeline

pipeline = PredictionPipeline()

text = input("Enter job description: ")

label, confidence, reason = pipeline.predict(text)

print("Prediction:", label)
print("Confidence:", round(confidence * 100, 2), "%")
print("Reason:", reason)