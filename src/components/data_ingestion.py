import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.config import CONFIG


class DataIngestionConfig:
    def __init__(self):
        cfg = CONFIG["data"]
        self.raw_data_path = cfg["raw_data_artifact"]
        self.train_data_path = cfg["train_data_path"]
        self.test_data_path = cfg["test_data_path"]
        self.test_size = cfg["test_size"]
        self.random_state = cfg["random_state"]


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data ingestion started")

            # Read dataset — path resolved relative to project root via config.yaml
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            raw_data_path = os.path.join(project_root, CONFIG["data"]["raw_data_path"])
            df = pd.read_csv(raw_data_path)
            logging.info("Dataset loaded successfully")

            # Create artifacts directory
            os.makedirs("artifacts", exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved in artifacts folder")

            # Train-test split — values from config.yaml
            train_set, test_set = train_test_split(
                df,
                test_size=self.ingestion_config.test_size,
                random_state=self.ingestion_config.random_state
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