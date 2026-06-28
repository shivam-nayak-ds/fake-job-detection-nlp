"""
DVC Stage 1: Data Ingestion
Run via: python scripts/run_ingestion.py
Or via:  dvc repro data_ingestion
"""
from src.components.data_ingestion import DataIngestion

ingestion = DataIngestion()
train_path, test_path = ingestion.initiate_data_ingestion()

print(f"Train: {train_path}")
print(f"Test:  {test_path}")
