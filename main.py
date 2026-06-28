import os
import json
from src.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    print(">>>>> Training Pipeline Started <<<<<")

    pipeline = TrainingPipeline()
    acc = pipeline.run_pipeline()

    # ─────────────────────────────────────────────
    # Write metrics for DVC to track
    # Run: dvc metrics show   → to view
    #      dvc metrics diff   → to compare with previous run
    # ─────────────────────────────────────────────
    os.makedirs("dvclive", exist_ok=True)
    metrics = {"accuracy": round(float(acc), 4)}
    with open("dvclive/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f">>>>> Training Pipeline Completed | Accuracy: {acc:.4f} <<<<<")
