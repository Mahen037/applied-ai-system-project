# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class implements a complete text classification pipeline:
  - Preprocess the text (tokenize, normalize, extract emojis)
  - Look for positive and negative words with negation handling
  - Compute a numeric score with word weights and emoji sentiment
  - Convert that score into a mood label with confidence scoring
"""

import re
import logging
from typing import List, Dict, Tuple, Optional

from dataset import (
    POSITIVE_WORDS, NEGATIVE_WORDS,
    EMOJI_SENTIMENT, NEGATION_WORDS, WORD_WEIGHTS,
)

logger = logging.getLogger(__name__)


class MoodAnalyzer:
    """
    A rule-based mood classifier with negation handling, weighted words,
    emoji sentiment, and confidence scoring.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)
        self.negation_words = set(w.lower() for w in NEGATION_WORDS)
        self.word_weights = {k.lower(): v for k, v in WORD_WEIGHTS.items()}
        self.emoji_sentiment = EMOJI_SENTIMENT

    # -----------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------

    def _extract_emojis(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract known emojis from text and return cleaned text + emoji list.
        """
        found_emojis = []
        # Check for multi-character emojis first (longer patterns first)
        for emoji in sorted(self.emoji_sentiment.keys(), key=len, reverse=True):
            while emoji in text:
                found_emojis.append(emoji)
                text = text.replace(emoji, " ", 1)
        return text, found_emojis

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps:
          1. Strip leading and trailing whitespace
          2. Extract emojis before lowering (preserves emoji case like :D)
          3. Convert to lowercase
          4. Remove punctuation (except apostrophes for contractions)
          5. Normalize repeated characters (soooo -> soo)
          6. Split on whitespace
          7. Filter out empty tokens
        """
        cleaned = text.strip()

        # Extract emojis before lowering case
        cleaned, emojis = self._extract_emojis(cleaned)

        cleaned = cleaned.lower()

        # Remove punctuation except apostrophes (for contractions like don't)
        cleaned = re.sub(r"[^\w\s']", " ", cleaned)

        # Normalize repeated characters: 3+ of the same char -> 2
        cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)

        tokens = cleaned.split()
        tokens = [t for t in tokens if t]  # Remove empty strings

        logger.debug("Preprocessed '%s' -> tokens=%s, emojis=%s", text, tokens, emojis)
        return tokens, emojis

    # -----------------------------------------------------------------
    # Scoring logic
    # -----------------------------------------------------------------

    def score_text(self, text: str) -> Tuple[int, float, Dict]:
        """
        Compute a numeric "mood score" for the given text.

        Features:
          - Positive words increase score, negative words decrease it
          - Word weights give some words stronger impact
          - Negation handling: "not happy" flips the sentiment
          - Emoji sentiment adds/subtracts from the score
          - Returns (score, confidence, details) tuple

        Returns:
            Tuple of (score, confidence, details_dict)
        """
        tokens, emojis = self.preprocess(text)

        score = 0
        positive_hits = []
        negative_hits = []
        negated_words = []
        emoji_contributions = []

        # --- Word scoring with negation ---
        is_negated = False
        for i, token in enumerate(tokens):
            # Check if this token is a negation word
            if token in self.negation_words:
                is_negated = True
                continue

            # Check weighted words first
            if token in self.word_weights:
                weight = self.word_weights[token]
                if is_negated:
                    weight = -weight
                    negated_words.append(token)
                score += weight
                if weight > 0:
                    positive_hits.append(token)
                else:
                    negative_hits.append(token)
                is_negated = False
                continue

            # Check positive words
            if token in self.positive_words:
                if is_negated:
                    score -= 1
                    negated_words.append(token)
                    negative_hits.append(f"NOT {token}")
                else:
                    score += 1
                    positive_hits.append(token)
                is_negated = False
                continue

            # Check negative words
            if token in self.negative_words:
                if is_negated:
                    score += 1
                    negated_words.append(token)
                    positive_hits.append(f"NOT {token}")
                else:
                    score -= 1
                    negative_hits.append(token)
                is_negated = False
                continue

            # Reset negation if no sentiment word follows within 1 token
            is_negated = False

        # --- Emoji scoring ---
        for emoji in emojis:
            emoji_score = self.emoji_sentiment.get(emoji, 0)
            score += emoji_score
            emoji_contributions.append((emoji, emoji_score))

        # --- Confidence calculation ---
        # Confidence is based on the magnitude of the score relative
        # to the number of sentiment signals found
        total_signals = len(positive_hits) + len(negative_hits) + len(emoji_contributions)
        if total_signals == 0:
            confidence = 0.3  # Low confidence when no signals found
        else:
            # Higher absolute score + more signals = higher confidence
            raw_confidence = min(abs(score) / max(total_signals, 1), 1.0)
            confidence = 0.3 + 0.7 * raw_confidence  # Scale to [0.3, 1.0]

        details = {
            "positive_words": positive_hits,
            "negative_words": negative_hits,
            "negated_words": negated_words,
            "emoji_contributions": emoji_contributions,
            "total_signals": total_signals,
        }

        logger.info(
            "Scored text: score=%d, confidence=%.2f, details=%s",
            score, confidence, details
        )
        return score, confidence, details

    # -----------------------------------------------------------------
    # Label prediction
    # -----------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score into a mood label.

        Label mapping with thresholds:
          - score >= 2   -> "positive"
          - score == 1   -> "positive" (mild)
          - score == 0   -> "neutral"
          - score == -1  -> "negative" (mild)
          - score <= -2  -> "negative"

        Special case: if both positive and negative signals exist
        and the score is close to zero (-1 to 1), return "mixed".
        """
        score, confidence, details = self.score_text(text)

        has_positive = len(details["positive_words"]) > 0 or any(
            s > 0 for _, s in details["emoji_contributions"]
        )
        has_negative = len(details["negative_words"]) > 0 or any(
            s < 0 for _, s in details["emoji_contributions"]
        )

        # Mixed: both positive and negative signals, score near zero
        if has_positive and has_negative and -1 <= score <= 1:
            label = "mixed"
        elif score > 0:
            label = "positive"
        elif score < 0:
            label = "negative"
        else:
            label = "neutral"

        logger.info("Predicted '%s' for text: '%s' (score=%d, conf=%.2f)",
                     label, text[:50], score, confidence)
        return label

    def predict_with_confidence(self, text: str) -> Tuple[str, float, Dict]:
        """
        Return the predicted label along with confidence score and details.
        """
        score, confidence, details = self.score_text(text)

        has_positive = len(details["positive_words"]) > 0 or any(
            s > 0 for _, s in details["emoji_contributions"]
        )
        has_negative = len(details["negative_words"]) > 0 or any(
            s < 0 for _, s in details["emoji_contributions"]
        )

        if has_positive and has_negative and -1 <= score <= 1:
            label = "mixed"
        elif score > 0:
            label = "positive"
        elif score < 0:
            label = "negative"
        else:
            label = "neutral"

        details["score"] = score
        return label, confidence, details

    # -----------------------------------------------------------------
    # Explanations
    # -----------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.
        """
        label, confidence, details = self.predict_with_confidence(text)

        parts = [f"Score = {details['score']}"]
        if details["positive_words"]:
            parts.append(f"positive: {details['positive_words']}")
        if details["negative_words"]:
            parts.append(f"negative: {details['negative_words']}")
        if details["negated_words"]:
            parts.append(f"negated: {details['negated_words']}")
        if details["emoji_contributions"]:
            emoji_str = ", ".join(f"{e}({s:+d})" for e, s in details["emoji_contributions"])
            parts.append(f"emojis: [{emoji_str}]")
        parts.append(f"confidence: {confidence:.2f}")

        return f"Label={label} | " + " | ".join(parts)
