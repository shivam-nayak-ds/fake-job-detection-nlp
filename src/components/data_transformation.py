import sys
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from src.exception import CustomException
from src.logger import logging
from src.config import CONFIG


class DataTransformationConfig:
    def __init__(self):
        cfg = CONFIG["transformation"]
        self.preprocessor_path = cfg["vectorizer_path"]
        self.max_features = cfg["max_features"]
        self.text_columns = cfg["text_columns"]
        self.target_column = cfg["target_column"]


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.vectorizer = TfidfVectorizer(max_features=self.config.max_features)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data transformation started")

            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded")

            # Handle missing values
            train_df = train_df.fillna("")
            test_df = test_df.fillna("")

            # Combine text columns — columns driven by config.yaml
            cols = self.config.text_columns
            train_df["text"] = train_df[cols].fillna("").agg(" ".join, axis=1)
            test_df["text"]  = test_df[cols].fillna("").agg(" ".join, axis=1)

            # Input-output split — target column from config.yaml
            target = self.config.target_column
            X_train = train_df["text"]
            y_train = train_df[target]

            X_test = test_df["text"]
            y_test = test_df[target]

            # Vectorization
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)

            logging.info("Text vectorization completed")

            # Save vectorizer
            os.makedirs("artifacts", exist_ok=True)

            with open(self.config.preprocessor_path, "wb") as f:
                pickle.dump(self.vectorizer, f)

            logging.info("Vectorizer saved successfully")

            return X_train_vec, X_test_vec, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)