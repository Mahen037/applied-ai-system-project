# tests/test_mood_analyzer.py
"""
Unit tests for the rule-based MoodAnalyzer.

Tests cover:
  - Preprocessing (tokenization, emoji extraction, normalization)
  - Scoring logic (positive/negative words, negation, weights, emojis)
  - Label prediction (thresholds, mixed mood detection)
  - Confidence scoring
  - Edge cases (empty input, all neutral, nonsense text)
"""

import sys
import os
import unittest

# Add parent directory to path so we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mood_analyzer import MoodAnalyzer


class TestPreprocessing(unittest.TestCase):
    """Tests for the preprocessor."""

    def setUp(self):
        self.analyzer = MoodAnalyzer()

    def test_basic_tokenization(self):
        """Text should be lowercased and split into tokens."""
        tokens, emojis = self.analyzer.preprocess("Hello World")
        self.assertEqual(tokens, ["hello", "world"])

    def test_strips_whitespace(self):
        """Leading/trailing whitespace should be removed."""
        tokens, _ = self.analyzer.preprocess("  hello  ")
        self.assertEqual(tokens, ["hello"])

    def test_removes_punctuation(self):
        """Punctuation should be removed from tokens."""
        tokens, _ = self.analyzer.preprocess("Hello, world! How are you?")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        # Punctuation chars should not appear as standalone tokens
        self.assertNotIn(",", tokens)
        self.assertNotIn("!", tokens)

    def test_normalizes_repeated_chars(self):
        """Repeated characters (3+) should be reduced to 2."""
        tokens, _ = self.analyzer.preprocess("soooo gooood")
        # 'soooo' -> 'soo', 'gooood' -> 'good' (after normalization: goood -> good)
        for token in tokens:
            # No token should have 3+ of the same char in a row
            import re
            self.assertIsNone(
                re.search(r"(.)\1{2,}", token),
                f"Token '{token}' still has 3+ repeated chars"
            )

    def test_emoji_extraction(self):
        """Emojis should be extracted and returned separately."""
        tokens, emojis = self.analyzer.preprocess("I am happy 😊")
        self.assertIn("😊", emojis)

    def test_preserves_contractions(self):
        """Apostrophes in contractions should be preserved."""
        tokens, _ = self.analyzer.preprocess("I don't like this")
        self.assertIn("don't", tokens)

    def test_empty_input(self):
        """Empty string should return empty lists."""
        tokens, emojis = self.analyzer.preprocess("")
        self.assertEqual(tokens, [])
        self.assertEqual(emojis, [])


class TestScoring(unittest.TestCase):
    """Tests for the scoring logic."""

    def setUp(self):
        self.analyzer = MoodAnalyzer()

    def test_positive_words_increase_score(self):
        """Positive words should increase the score."""
        score, _, _ = self.analyzer.score_text("I am happy and great")
        self.assertGreater(score, 0)

    def test_negative_words_decrease_score(self):
        """Negative words should decrease the score."""
        score, _, _ = self.analyzer.score_text("This is terrible and awful")
        self.assertLess(score, 0)

    def test_neutral_text_score_zero(self):
        """Text with no sentiment words should score around zero."""
        score, _, _ = self.analyzer.score_text("the weather is okay")
        self.assertEqual(score, 0)

    def test_negation_flips_positive(self):
        """'not happy' should have a negative score."""
        score, _, _ = self.analyzer.score_text("I am not happy")
        self.assertLess(score, 0)

    def test_negation_flips_negative(self):
        """'not bad' should have a positive score."""
        score, _, _ = self.analyzer.score_text("This is not bad")
        self.assertGreater(score, 0)

    def test_weighted_words(self):
        """Strongly weighted words should have bigger impact."""
        score_love, _, _ = self.analyzer.score_text("I love this")
        score_good, _, _ = self.analyzer.score_text("I think this is good")
        # 'love' has weight 2, 'good' has weight 1
        self.assertGreater(score_love, score_good)

    def test_emoji_positive_contribution(self):
        """Positive emojis should increase the score."""
        score_no_emoji, _, _ = self.analyzer.score_text("I feel fine")
        score_with_emoji, _, _ = self.analyzer.score_text("I feel fine 😊")
        self.assertGreater(score_with_emoji, score_no_emoji)

    def test_emoji_negative_contribution(self):
        """Negative emojis should decrease the score."""
        score_no_emoji, _, _ = self.analyzer.score_text("That happened")
        score_with_emoji, _, _ = self.analyzer.score_text("That happened 😢")
        self.assertLess(score_with_emoji, score_no_emoji)

    def test_score_returns_tuple(self):
        """score_text should return (score, confidence, details)."""
        result = self.analyzer.score_text("I am happy")
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], int)      # score
        self.assertIsInstance(result[1], float)    # confidence
        self.assertIsInstance(result[2], dict)     # details


class TestPrediction(unittest.TestCase):
    """Tests for label prediction."""

    def setUp(self):
        self.analyzer = MoodAnalyzer()

    def test_positive_prediction(self):
        """Clearly positive text should be labeled positive."""
        label = self.analyzer.predict_label("I love this amazing day")
        self.assertEqual(label, "positive")

    def test_negative_prediction(self):
        """Clearly negative text should be labeled negative."""
        label = self.analyzer.predict_label("This is terrible and horrible")
        self.assertEqual(label, "negative")

    def test_neutral_prediction(self):
        """Text with no sentiment should be labeled neutral."""
        label = self.analyzer.predict_label("The meeting is at three")
        self.assertEqual(label, "neutral")

    def test_valid_labels_only(self):
        """Predictions should only return valid labels."""
        valid = {"positive", "negative", "neutral", "mixed"}
        for text in ["I love it", "I hate it", "ok", "good and bad"]:
            label = self.analyzer.predict_label(text)
            self.assertIn(label, valid, f"Invalid label '{label}' for '{text}'")


class TestConfidence(unittest.TestCase):
    """Tests for confidence scoring."""

    def setUp(self):
        self.analyzer = MoodAnalyzer()

    def test_confidence_range(self):
        """Confidence should be between 0 and 1."""
        _, confidence, _ = self.analyzer.predict_with_confidence("I love this")
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_strong_signals_higher_confidence(self):
        """Text with many sentiment signals should have higher confidence."""
        _, conf_strong, _ = self.analyzer.predict_with_confidence(
            "I love this amazing wonderful brilliant day"
        )
        _, conf_weak, _ = self.analyzer.predict_with_confidence(
            "the weather might be okay"
        )
        self.assertGreaterEqual(conf_strong, conf_weak)

    def test_no_signals_low_confidence(self):
        """Text with no sentiment words should have low confidence."""
        _, confidence, _ = self.analyzer.predict_with_confidence(
            "the cat sat on the mat"
        )
        self.assertLess(confidence, 0.5)


class TestExplain(unittest.TestCase):
    """Tests for the explain method."""

    def setUp(self):
        self.analyzer = MoodAnalyzer()

    def test_explain_returns_string(self):
        """explain() should return a non-empty string."""
        explanation = self.analyzer.explain("I love this")
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)

    def test_explain_includes_label(self):
        """Explanation should mention the predicted label."""
        explanation = self.analyzer.explain("I love this")
        self.assertIn("positive", explanation.lower())

    def test_explain_includes_score(self):
        """Explanation should include the numeric score."""
        explanation = self.analyzer.explain("I love this")
        self.assertIn("Score", explanation)


if __name__ == "__main__":
    unittest.main()
