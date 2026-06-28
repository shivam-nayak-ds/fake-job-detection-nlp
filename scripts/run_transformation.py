"""
DVC Stage 2: Data Transformation
Run via: python scripts/run_transformation.py
Or via:  dvc repro data_transformation
"""
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

ingestion = DataIngestion()
train_path, test_path = ingestion.initiate_data_ingestion()

transformation = DataTransformation()
X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(
    train_path, test_path
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
