# models/tfidf_baseline.py
"""
TF-IDF + Logistic Regression baseline.

Provides a scikit-learn pipeline that:
  1. Vectorises text with TF-IDF (character + word n-grams optional)
  2. Trains a Logistic Regression classifier

Results are serialisable with joblib for later comparison.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
from typing import List, Tuple, Dict, Optional

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

from data.preprocessing import clean_text


class TFIDFBaseline:
    """
    Scikit-learn TF-IDF + Logistic Regression pipeline.

    Args:
        max_features : Maximum number of TF-IDF features.
        ngram_range  : Tuple (min_n, max_n) for n-gram extraction.
        sublinear_tf : Apply sublinear TF scaling (log(1 + tf)).
        C            : Inverse regularisation strength for LR.
        max_iter     : Maximum LR solver iterations.
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        C: float = 5.0,
        max_iter: int = 1000,
    ) -> None:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                preprocessor=clean_text,
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=sublinear_tf,
                min_df=2,
                strip_accents="unicode",
            )),
            ("clf", LogisticRegression(
                C=C,
                max_iter=max_iter,
                solver="lbfgs",
                multi_class="multinomial",
                n_jobs=-1,
                random_state=42,
            )),
        ])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, texts: List[str], labels: List[int]) -> "TFIDFBaseline":
        """
        Fit the TF-IDF vectoriser and the classifier.

        Args:
            texts  : Raw text strings.
            labels : Integer class labels.

        Returns:
            self
        """
        self.pipeline.fit(texts, labels)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        texts: List[str],
        labels: List[int],
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute and print a full classification report.

        Returns:
            Dict with 'accuracy' and 'macro_f1' keys.
        """
        preds = self.predict(texts)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")

        print("\n" + "=" * 60)
        print("TF-IDF Baseline Evaluation")
        print("=" * 60)
        print(classification_report(
            labels, preds,
            target_names=class_names,
            digits=4,
        ))
        print(f"Accuracy   : {acc:.4f}")
        print(f"Macro F1   : {f1:.4f}")
        print("=" * 60)

        return {"accuracy": acc, "macro_f1": f1}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the fitted pipeline to disk with joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"[TFIDFBaseline] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TFIDFBaseline":
        """Load a previously saved pipeline."""
        baseline = cls.__new__(cls)
        baseline.pipeline = joblib.load(path)
        print(f"[TFIDFBaseline] Loaded from {path}")
        return baseline


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_tfidf_baseline(cfg: dict) -> TFIDFBaseline:
    """
    Build TFIDFBaseline from the 'tfidf' section of config.

    Args:
        cfg: Full config dict.

    Returns:
        Uninitialised TFIDFBaseline (call .fit() to train).
    """
    tcfg = cfg["tfidf"]
    return TFIDFBaseline(
        max_features=tcfg["max_features"],
        ngram_range=tuple(tcfg["ngram_range"]),
        sublinear_tf=tcfg["sublinear_tf"],
        C=tcfg["lr_C"],
        max_iter=tcfg["lr_max_iter"],
    )
