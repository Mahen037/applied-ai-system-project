# tests/test_pipeline.py
"""
Integration tests for the full Mood Machine pipeline.

Tests cover:
  - Pipeline initialization
  - Full analysis flow (rule-based + ML + RAG)
  - Ensemble logic
  - Batch processing
  - Evaluation metrics
  - Error handling and edge cases
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import MoodPipeline
from dataset import SAMPLE_POSTS, TRUE_LABELS


class TestPipelineInit(unittest.TestCase):
    """Tests for pipeline initialization."""

    def test_full_pipeline_init(self):
        """Pipeline should initialize with all components."""
        pipeline = MoodPipeline(use_rag=True, use_ml=True)
        self.assertIsNotNone(pipeline.rule_analyzer)
        self.assertIsNotNone(pipeline.ml_classifier)
        self.assertIsNotNone(pipeline.rag_explainer)

    def test_pipeline_without_rag(self):
        """Pipeline should work without RAG."""
        pipeline = MoodPipeline(use_rag=False, use_ml=True)
        self.assertIsNone(pipeline.rag_explainer)

    def test_pipeline_without_ml(self):
        """Pipeline should work without ML."""
        pipeline = MoodPipeline(use_rag=True, use_ml=False)
        self.assertIsNone(pipeline.ml_classifier)

    def test_minimal_pipeline(self):
        """Pipeline should work with only rule-based."""
        pipeline = MoodPipeline(use_rag=False, use_ml=False)
        self.assertIsNotNone(pipeline.rule_analyzer)


class TestPipelineAnalysis(unittest.TestCase):
    """Tests for the analysis pipeline."""

    @classmethod
    def setUpClass(cls):
        """Initialize pipeline once."""
        cls.pipeline = MoodPipeline(use_rag=True, use_ml=True)

    def test_analyze_returns_dict(self):
        """analyze() should return a dictionary."""
        result = self.pipeline.analyze("I love this")
        self.assertIsInstance(result, dict)

    def test_analyze_has_required_keys(self):
        """Result should have all required keys."""
        result = self.pipeline.analyze("I love this")
        required_keys = ["text", "label", "confidence", "rule_based",
                         "ml", "explanation", "retrieved_examples", "timestamp"]
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_analyze_label_is_valid(self):
        """Predicted label should be a valid mood."""
        valid = {"positive", "negative", "neutral", "mixed"}
        result = self.pipeline.analyze("I love this")
        self.assertIn(result["label"], valid)

    def test_analyze_confidence_range(self):
        """Confidence should be between 0 and 1."""
        result = self.pipeline.analyze("I love this so much")
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_analyze_includes_explanation(self):
        """Result should include a non-empty explanation."""
        result = self.pipeline.analyze("I am very happy today")
        self.assertIsInstance(result["explanation"], str)
        self.assertGreater(len(result["explanation"]), 0)

    def test_analyze_includes_retrieved_examples(self):
        """Result should include retrieved examples."""
        result = self.pipeline.analyze("I am very happy today")
        self.assertIsInstance(result["retrieved_examples"], list)
        self.assertGreater(len(result["retrieved_examples"]), 0)

    def test_positive_text(self):
        """Clearly positive text should be classified as positive."""
        result = self.pipeline.analyze("I absolutely love this amazing wonderful day!")
        self.assertEqual(result["label"], "positive")

    def test_negative_text(self):
        """Clearly negative text should be classified as negative."""
        result = self.pipeline.analyze("This is terrible, awful, and horrible")
        self.assertEqual(result["label"], "negative")


class TestBatchProcessing(unittest.TestCase):
    """Tests for batch analysis."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = MoodPipeline(use_rag=True, use_ml=True)

    def test_batch_analyze(self):
        """Batch analysis should process all texts."""
        texts = ["I love this", "I hate this", "okay"]
        results = self.pipeline.batch_analyze(texts)
        self.assertEqual(len(results), 3)

    def test_batch_results_have_labels(self):
        """Each batch result should have a label."""
        texts = ["happy day", "sad day"]
        results = self.pipeline.batch_analyze(texts)
        for r in results:
            self.assertIn("label", r)


class TestEvaluation(unittest.TestCase):
    """Tests for pipeline evaluation."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = MoodPipeline(use_rag=True, use_ml=True)

    def test_evaluate_returns_metrics(self):
        """evaluate() should return accuracy metrics."""
        evaluation = self.pipeline.evaluate(SAMPLE_POSTS, TRUE_LABELS)
        self.assertIn("accuracy", evaluation)
        self.assertIn("correct", evaluation)
        self.assertIn("total", evaluation)
        self.assertIn("average_confidence", evaluation)

    def test_accuracy_above_30_percent(self):
        """Pipeline accuracy should be above 30% (better than random for 4 classes)."""
        evaluation = self.pipeline.evaluate(SAMPLE_POSTS, TRUE_LABELS)
        self.assertGreater(evaluation["accuracy"], 0.30)

    def test_confidence_in_valid_range(self):
        """Average confidence should be in valid range."""
        evaluation = self.pipeline.evaluate(SAMPLE_POSTS, TRUE_LABELS)
        self.assertGreaterEqual(evaluation["average_confidence"], 0.0)
        self.assertLessEqual(evaluation["average_confidence"], 1.0)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = MoodPipeline(use_rag=True, use_ml=True)

    def test_empty_text(self):
        """Empty text should not crash."""
        result = self.pipeline.analyze("")
        self.assertIn("label", result)

    def test_single_word(self):
        """Single word should work."""
        result = self.pipeline.analyze("happy")
        self.assertIn("label", result)

    def test_very_long_text(self):
        """Very long text should not crash."""
        long_text = "I am happy " * 100
        result = self.pipeline.analyze(long_text)
        self.assertIn("label", result)

    def test_special_characters(self):
        """Text with special characters should not crash."""
        result = self.pipeline.analyze("@#$%^&*() ???!!!")
        self.assertIn("label", result)

    def test_only_emojis(self):
        """Text with only emojis should not crash."""
        result = self.pipeline.analyze("😊😊😊")
        self.assertIn("label", result)


if __name__ == "__main__":
    unittest.main()
