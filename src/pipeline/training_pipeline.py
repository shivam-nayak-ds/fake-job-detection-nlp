import sys
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            print("🚀 Training pipeline started")

            # =========================
            # Step 1: Data Ingestion
            # =========================
            print("📥 Data Ingestion Started")

            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            print(f"Train Path: {train_path}")
            print(f"Test Path: {test_path}")
            print("✅ Data Ingestion Completed")

            # =========================
            # Step 2: Data Transformation
            # =========================
            print("🔄 Data Transformation Started")

            transformation = DataTransformation()
            X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(
                train_path, test_path
            )

            print("X_train shape:", getattr(X_train, "shape", "N/A"))
            print("X_test shape:", getattr(X_test, "shape", "N/A"))
            print("✅ Data Transformation Completed")

            # =========================
            # Step 3: Model Training
            # =========================
            print("🤖 Model Training Started")

            trainer = ModelTrainer()
            acc = trainer.initiate_model_training(
                X_train, X_test, y_train, y_test
            )

            print(f"🎯 Accuracy: {acc}")
            print("✅ Model Training Completed")

            print("🎉 Pipeline completed successfully")

        except Exception as e:
            raise CustomException(e, sys)