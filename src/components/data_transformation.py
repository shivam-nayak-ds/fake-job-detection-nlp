import sys
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from src.exception import CustomException
from src.logger import logging


class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "vectorizer.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.vectorizer = TfidfVectorizer(max_features=5000)

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

            # Combine text columns
            train_df["text"] = (
                train_df["title"] + " " +
                train_df["description"] + " " +
                train_df["requirements"] + " " +
                train_df["benefits"] + " " +
                train_df["company_profile"]
            )

            test_df["text"] = (
                test_df["title"] + " " +
                test_df["description"] + " " +
                test_df["requirements"] + " " +
                test_df["benefits"] + " " +
                test_df["company_profile"]
            )

            # Input-output split
            X_train = train_df["text"]
            y_train = train_df["fraudulent"]

            X_test = test_df["text"]
            y_test = test_df["fraudulent"]

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