import sys
import os
import pickle

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.exception import CustomException


class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            print("🤖 Model training started")

            # =========================
            # Debug: Check class distribution
            # =========================
            import pandas as pd
            print("\n📊 y_train distribution:\n", pd.Series(y_train).value_counts())
            print("\n📊 y_test distribution:\n", pd.Series(y_test).value_counts())

            # =========================
            # Handle imbalance (IMPORTANT)
            # =========================
            neg = sum(y_train == 0)
            pos = sum(y_train == 1)

            scale_pos_weight = neg / pos if pos != 0 else 1

            print(f"\n⚖️ scale_pos_weight: {scale_pos_weight:.2f}")

            # =========================
            # Model (IMPROVED)
            # =========================
            model = XGBClassifier(
                learning_rate=0.08,
                max_depth=6,
                n_estimators=200,
                scale_pos_weight=scale_pos_weight,  # 🔥 dynamic
                eval_metric='logloss',
                use_label_encoder=False
            )

            # =========================
            # Train
            # =========================
            model.fit(X_train, y_train)

            # =========================
            # Predict
            # =========================
            y_pred = model.predict(X_test)

            print("\n🔍 Unique predictions:", set(y_pred))

            # =========================
            # Accuracy
            # =========================
            acc = accuracy_score(y_test, y_pred)
            print(f"\n🎯 Accuracy: {acc}")

            # =========================
            # Classification Report
            # =========================
            report = classification_report(y_test, y_pred)
            print("\n📊 Classification Report:\n", report)

            # =========================
            # Save model
            # =========================
            os.makedirs("artifacts", exist_ok=True)

            with open(self.model_path, "wb") as f:
                pickle.dump(model, f)

            print("\n✅ Model saved successfully")

            return acc

        except Exception as e:
            raise CustomException(e, sys)