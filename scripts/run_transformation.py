"""
DVC Stage 2: Data Transformation
Run via: python scripts/run_transformation.py
Or via:  dvc repro data_transformation

Expects artifacts/train.csv and artifacts/test.csv to exist
(produced by Stage 1: data_ingestion).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.data_transformation import DataTransformation
from src.config import CONFIG

# Read paths from config — Stage 1 already produced these files
train_path = CONFIG["data"]["train_data_path"]
test_path  = CONFIG["data"]["test_data_path"]

transformation = DataTransformation()
X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(
    train_path, test_path
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print("Vectorizer saved to artifacts/vectorizer.pkl")
