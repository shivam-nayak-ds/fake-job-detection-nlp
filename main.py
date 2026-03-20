from src.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    print(">>>>> Training Pipeline Started <<<<<")

    pipeline = TrainingPipeline()
    pipeline.run_pipeline()

    print(">>>>> Training Pipeline Completed <<<<<")
