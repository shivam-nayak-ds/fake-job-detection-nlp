"""
SHAP Explainability Module — Trust-Hire
========================================
Uses SHAP TreeExplainer on the XGBoost model to identify
which words in a job description pushed the prediction
towards "Fake" or "Real".

Usage:
    from src.explainability import SHAPExplainer
    explainer = SHAPExplainer(model, vectorizer)
    top_features = explainer.explain(text, top_n=10)
"""

import numpy as np
import shap
from typing import List, Dict


class SHAPExplainer:
    """
    Wraps SHAP TreeExplainer for XGBoost + TF-IDF models.
    Instantiate once, call explain() per prediction.
    """

    def __init__(self, model, vectorizer):
        """
        Parameters
        ----------
        model      : trained XGBClassifier
        vectorizer : fitted TfidfVectorizer
        """
        self.model      = model
        self.vectorizer = vectorizer
        self.feature_names = np.array(vectorizer.get_feature_names_out())

        # TreeExplainer is fast and exact for XGBoost
        self.explainer = shap.TreeExplainer(model)

    def explain(self, text: str, top_n: int = 10) -> List[Dict]:
        """
        Compute SHAP values for a single job description.

        Returns
        -------
        List of dicts sorted by |shap_value| descending:
        [
          {"word": "urgent", "shap_value": 0.42, "direction": "fake"},
          {"word": "engineer", "shap_value": -0.31, "direction": "real"},
          ...
        ]
        Positive shap_value  → pushes towards Fake
        Negative shap_value  → pushes towards Real
        """
        # Vectorize the input text
        vec = self.vectorizer.transform([text])          # (1, n_features) sparse

        # SHAP values — shape (1, n_features) for binary class (class=1 = Fake)
        shap_values = self.explainer.shap_values(vec)

        # shap_values can be a list [class0, class1] or single array
        if isinstance(shap_values, list):
            values = shap_values[1][0]   # class 1 (Fake) SHAP values for sample 0
        else:
            values = shap_values[0]      # single output (log-odds)

        # Only keep non-zero features (words actually present in this text)
        nonzero_idx = vec.nonzero()[1]
        nonzero_vals = values[nonzero_idx]
        nonzero_words = self.feature_names[nonzero_idx]

        # Sort by absolute SHAP value — most impactful words first
        order = np.argsort(np.abs(nonzero_vals))[::-1]
        top_idx = order[:top_n]

        results = []
        for i in top_idx:
            sv = float(nonzero_vals[i])
            results.append({
                "word"       : str(nonzero_words[i]),
                "shap_value" : round(sv, 4),
                "direction"  : "fake" if sv > 0 else "real",
            })

        return results

    def summary(self, text: str, top_n: int = 10) -> str:
        """Human-readable summary string for quick debugging."""
        features = self.explain(text, top_n)
        lines = [f"  SHAP Top-{top_n} features:"]
        for f in features:
            arrow = "-> FAKE" if f["direction"] == "fake" else "-> REAL"
            lines.append(f"    [{f['shap_value']:+.4f}] '{f['word']}' {arrow}")
        return "\n".join(lines)
