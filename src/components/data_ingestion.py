import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


class DataIngestionConfig:
    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "data.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data ingestion started")

            # Read dataset
            df = pd.read_csv(r"C:\fake-job-detection-nlp\data\raw_data\fake_job_postings.csv")
            logging.info("Dataset loaded successfully")

            # Create artifacts directory
            os.makedirs("artifacts", exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved in artifacts folder")

            # Train-test split
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42
            )

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Train-test split completed")
            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)