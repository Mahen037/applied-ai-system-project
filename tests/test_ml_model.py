# tests/test_ml_model.py
"""
Unit tests for the ML mood classifier.

Tests cover:
  - Training and basic prediction
  - Confidence scoring
  - Evaluation metrics
  - Error handling
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_experiments import MLMoodClassifier
from dataset import SAMPLE_POSTS, TRUE_LABELS


class TestMLClassifier(unittest.TestCase):
    """Tests for the ML mood classifier."""

    @classmethod
    def setUpClass(cls):
        """Train the model once for all tests."""
        cls.classifier = MLMoodClassifier()
        cls.classifier.train(SAMPLE_POSTS, TRUE_LABELS)

    def test_model_trains_successfully(self):
        """Model should train without errors."""
        self.assertTrue(self.classifier._is_trained)

    def test_predict_returns_valid_label(self):
        """Predictions should be one of the known labels."""
        valid_labels = {"positive", "negative", "neutral", "mixed"}
        label = self.classifier.predict("I love this")
        self.assertIn(label, valid_labels)

    def test_predict_with_confidence(self):
        """Prediction with confidence should return a tuple."""
        label, conf, details = self.classifier.predict_with_confidence("I love this")
        self.assertIsInstance(label, str)
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        self.assertIn("class_probabilities", details)

    def test_probabilities_sum_to_one(self):
        """Class probabilities should sum to approximately 1."""
        _, _, details = self.classifier.predict_with_confidence("I feel okay today")
        probs = details["class_probabilities"]
        total = sum(probs.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_training_accuracy_above_50(self):
        """Training accuracy should be above 50% (better than random for 4 classes)."""
        results = self.classifier.evaluate(SAMPLE_POSTS, TRUE_LABELS)
        self.assertGreater(results["accuracy"], 0.5)

    def test_evaluate_returns_metrics(self):
        """evaluate() should return expected keys."""
        results = self.classifier.evaluate(SAMPLE_POSTS, TRUE_LABELS)
        self.assertIn("accuracy", results)
        self.assertIn("predictions", results)
        self.assertIn("correct", results)
        self.assertIn("total", results)


class TestMLErrorHandling(unittest.TestCase):
    """Tests for ML classifier error handling."""

    def test_mismatched_lengths_raises(self):
        """Training with mismatched lengths should raise ValueError."""
        clf = MLMoodClassifier()
        with self.assertRaises(ValueError):
            clf.train(["hello"], ["positive", "negative"])

    def test_empty_data_raises(self):
        """Training with no data should raise ValueError."""
        clf = MLMoodClassifier()
        with self.assertRaises(ValueError):
            clf.train([], [])

    def test_predict_before_training_raises(self):
        """Predicting before training should raise RuntimeError."""
        clf = MLMoodClassifier()
        with self.assertRaises(RuntimeError):
            clf.predict("hello")


if __name__ == "__main__":
    unittest.main()
