import os
import pickle


def save_object(file_path, obj):
    """
    Save any object (model, vectorizer, etc.)
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_object(file_path):
    """
    Load any saved object
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)