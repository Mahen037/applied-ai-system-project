# ml_experiments.py
"""
Enhanced ML experiments for the Mood Machine.

Uses TF-IDF features (instead of raw counts) and includes:
  - Confidence scoring via predicted probabilities
  - Cross-validation evaluation
  - Comparison with rule-based model
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

from dataset import SAMPLE_POSTS, TRUE_LABELS

logger = logging.getLogger(__name__)


class MLMoodClassifier:
    """
    A machine learning mood classifier using TF-IDF + Logistic Regression
    with confidence scoring.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=500,
            stop_words=None,     # Keep all words since sentiment words matter
        )
        self.model = LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
        )
        self._is_trained = False
        self._classes = []

    def train(self, texts: List[str], labels: List[str]) -> float:
        """
        Train the classifier on labeled texts.

        Returns training accuracy.
        """
        if len(texts) != len(labels):
            raise ValueError(
                f"texts ({len(texts)}) and labels ({len(labels)}) must be the same length."
            )
        if not texts:
            raise ValueError("No training data provided.")

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self._is_trained = True
        self._classes = list(self.model.classes_)

        preds = self.model.predict(X)
        accuracy = accuracy_score(labels, preds)

        logger.info("Trained ML model: accuracy=%.2f on %d examples", accuracy, len(texts))
        return accuracy

    def predict(self, text: str) -> str:
        """Predict the mood label for a single text."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

    def predict_with_confidence(self, text: str) -> Tuple[str, float, dict]:
        """
        Predict mood label with confidence score.

        Returns:
            (label, confidence, details) where confidence is the
            probability of the predicted class.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X = self.vectorizer.transform([text])
        label = self.model.predict(X)[0]
        probas = self.model.predict_proba(X)[0]

        class_probs = dict(zip(self._classes, probas))
        confidence = max(probas)

        details = {
            "class_probabilities": class_probs,
            "predicted_class": label,
        }

        logger.info("ML prediction: label=%s, confidence=%.2f", label, confidence)
        return label, confidence, details

    def evaluate(self, texts: List[str], labels: List[str]) -> dict:
        """
        Evaluate the model and return a results dictionary.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X = self.vectorizer.transform(texts)
        preds = self.model.predict(X)
        accuracy = accuracy_score(labels, preds)

        results = {
            "accuracy": accuracy,
            "predictions": list(preds),
            "true_labels": labels,
            "correct": sum(p == t for p, t in zip(preds, labels)),
            "total": len(labels),
        }

        logger.info("ML evaluation: accuracy=%.2f (%d/%d)",
                     accuracy, results["correct"], results["total"])
        return results

    def cross_validate(self, texts: List[str], labels: List[str],
                       cv: int = 3) -> dict:
        """
        Run cross-validation and return scores.
        """
        X = self.vectorizer.fit_transform(texts)
        # For very small datasets, reduce cv
        actual_cv = min(cv, len(set(labels)), len(labels))
        if actual_cv < 2:
            logger.warning("Dataset too small for cross-validation")
            return {"mean_accuracy": 0.0, "std_accuracy": 0.0, "scores": []}

        scores = cross_val_score(self.model, X, labels, cv=actual_cv)
        result = {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
            "scores": list(scores),
        }
        logger.info("Cross-validation: mean=%.2f, std=%.2f", result["mean_accuracy"], result["std_accuracy"])
        return result


def train_ml_model(
    texts: List[str],
    labels: List[str],
) -> MLMoodClassifier:
    """
    Train and return an MLMoodClassifier.
    """
    classifier = MLMoodClassifier()
    classifier.train(texts, labels)
    return classifier


def run_evaluation() -> None:
    """Run full evaluation and print results."""
    print("Training an ML model on SAMPLE_POSTS and TRUE_LABELS from dataset.py...")
    print(f"Dataset size: {len(SAMPLE_POSTS)} posts\n")

    classifier = train_ml_model(SAMPLE_POSTS, TRUE_LABELS)

    # Training evaluation
    results = classifier.evaluate(SAMPLE_POSTS, TRUE_LABELS)
    print("=== ML Model Evaluation (Training Data) ===")
    for text, true_label, pred_label in zip(
        SAMPLE_POSTS, TRUE_LABELS, results["predictions"]
    ):
        marker = "✓" if pred_label == true_label else "✗"
        print(f'  {marker} "{text}" -> predicted={pred_label}, true={true_label}')
    print(f"\nAccuracy: {results['accuracy']:.2f} ({results['correct']}/{results['total']})")

    # Confidence examples
    print("\n=== Confidence Scores for Sample Predictions ===")
    for text in SAMPLE_POSTS[:5]:
        label, conf, details = classifier.predict_with_confidence(text)
        probs = details["class_probabilities"]
        probs_str = ", ".join(f"{k}: {v:.2f}" for k, v in sorted(probs.items()))
        print(f'  "{text}" -> {label} (conf={conf:.2f}, probs=[{probs_str}])')

    return classifier


if __name__ == "__main__":
    classifier = run_evaluation()

    # Interactive loop
    print("\n=== Interactive Mood Machine (ML model) ===")
    print("Type a sentence to analyze its mood.")
    print("Type 'quit' or press Enter on an empty line to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input == "" or user_input.lower() == "quit":
            print("Goodbye from the ML Mood Machine.")
            break

        label, conf, details = classifier.predict_with_confidence(user_input)
        probs = details["class_probabilities"]
        probs_str = ", ".join(f"{k}: {v:.2f}" for k, v in sorted(probs.items()))
        print(f"ML model: {label} (confidence={conf:.2f}, probabilities=[{probs_str}])")
