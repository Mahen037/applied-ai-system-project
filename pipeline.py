# pipeline.py
"""
Unified Mood Machine Pipeline.

Orchestrates the full AI system:
  1. Rule-based MoodAnalyzer for scoring and initial prediction
  2. ML classifier (TF-IDF + Logistic Regression) for a second opinion
  3. RAG Explainer for context-aware explanations
  4. Ensemble logic to combine signals with confidence scoring
  5. Comprehensive logging throughout
"""

import logging
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from mood_analyzer import MoodAnalyzer
from ml_experiments import MLMoodClassifier
from rag_explainer import RAGExplainer
from dataset import SAMPLE_POSTS, TRUE_LABELS

logger = logging.getLogger(__name__)


class MoodPipeline:
    """
    The unified pipeline that combines rule-based analysis, ML classification,
    and RAG-powered explanations into a single prediction system.
    """

    def __init__(self, use_rag: bool = True, use_ml: bool = True) -> None:
        """
        Initialize the pipeline.

        Args:
            use_rag: Whether to enable RAG explanations
            use_ml: Whether to enable the ML classifier
        """
        self.use_rag = use_rag
        self.use_ml = use_ml

        # Initialize components
        logger.info("Initializing MoodPipeline (rag=%s, ml=%s)", use_rag, use_ml)

        self.rule_analyzer = MoodAnalyzer()
        logger.info("Rule-based analyzer initialized")

        self.ml_classifier: Optional[MLMoodClassifier] = None
        if use_ml:
            self.ml_classifier = MLMoodClassifier()
            self.ml_classifier.train(SAMPLE_POSTS, TRUE_LABELS)
            logger.info("ML classifier trained on %d examples", len(SAMPLE_POSTS))

        self.rag_explainer: Optional[RAGExplainer] = None
        if use_rag:
            self.rag_explainer = RAGExplainer(top_k=3)
            self.rag_explainer.build_index(SAMPLE_POSTS, TRUE_LABELS)
            logger.info("RAG index built with %d examples", len(SAMPLE_POSTS))

        logger.info("MoodPipeline ready")

    def analyze(self, text: str) -> Dict:
        """
        Run the full analysis pipeline on a text input.

        Returns a dictionary with:
          - text: the original input
          - label: final predicted mood label
          - confidence: overall confidence score
          - rule_based: rule-based prediction details
          - ml: ML prediction details (if enabled)
          - explanation: RAG-powered explanation (if enabled)
          - retrieved_examples: similar examples from dataset (if RAG enabled)
          - timestamp: when the analysis was done
        """
        logger.info("Analyzing: '%s'", text[:80])
        result = {
            "text": text,
            "timestamp": datetime.now().isoformat(),
        }

        # --- Step 1: Rule-based analysis ---
        rule_label, rule_conf, rule_details = self.rule_analyzer.predict_with_confidence(text)
        result["rule_based"] = {
            "label": rule_label,
            "confidence": rule_conf,
            "details": rule_details,
        }
        logger.info("Rule-based: label=%s, confidence=%.2f", rule_label, rule_conf)

        # --- Step 2: ML classification (if enabled) ---
        if self.ml_classifier is not None:
            ml_label, ml_conf, ml_details = self.ml_classifier.predict_with_confidence(text)
            result["ml"] = {
                "label": ml_label,
                "confidence": ml_conf,
                "details": ml_details,
            }
            logger.info("ML: label=%s, confidence=%.2f", ml_label, ml_conf)
        else:
            ml_label = None
            ml_conf = 0.0
            result["ml"] = None

        # --- Step 3: Ensemble decision ---
        final_label, final_conf = self._ensemble(
            rule_label, rule_conf, ml_label, ml_conf
        )
        result["label"] = final_label
        result["confidence"] = final_conf
        logger.info("Ensemble: label=%s, confidence=%.2f", final_label, final_conf)

        # --- Step 4: RAG explanation (if enabled) ---
        if self.rag_explainer is not None:
            retrieved = self.rag_explainer.retrieve(text)
            result["retrieved_examples"] = [
                {
                    "text": ex["text"],
                    "label": ex["metadata"]["label"],
                    "similarity": ex["similarity"],
                }
                for ex in retrieved
            ]

            # Generate RAG-augmented explanation
            explanation = self.rag_explainer.generate_explanation(
                text=text,
                predicted_label=final_label,
                confidence=final_conf,
                score_details=rule_details,
                retrieved_examples=retrieved,
            )
            result["explanation"] = explanation
        else:
            result["explanation"] = self.rule_analyzer.explain(text)
            result["retrieved_examples"] = []

        return result

    def _ensemble(
        self,
        rule_label: str,
        rule_conf: float,
        ml_label: Optional[str],
        ml_conf: float,
    ) -> Tuple[str, float]:
        """
        Combine rule-based and ML predictions into a final decision.

        Strategy:
          - If both agree, use the agreed label with boosted confidence
          - If they disagree, use the one with higher confidence
          - If only rule-based is available, use it directly
        """
        if ml_label is None:
            return rule_label, rule_conf

        if rule_label == ml_label:
            # Agreement: boost confidence
            combined_conf = min(1.0, (rule_conf + ml_conf) / 2 + 0.1)
            logger.info("Ensemble agreement: both say '%s' (conf=%.2f)", rule_label, combined_conf)
            return rule_label, combined_conf
        else:
            # Disagreement: pick higher confidence, reduce overall confidence
            if rule_conf >= ml_conf:
                chosen = rule_label
                chosen_conf = rule_conf * 0.8  # Reduce since there's disagreement
            else:
                chosen = ml_label
                chosen_conf = ml_conf * 0.8

            logger.info(
                "Ensemble disagreement: rule=%s(%.2f) vs ml=%s(%.2f) -> chose %s(%.2f)",
                rule_label, rule_conf, ml_label, ml_conf, chosen, chosen_conf,
            )
            return chosen, chosen_conf

    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """Analyze a batch of texts."""
        return [self.analyze(text) for text in texts]

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict:
        """
        Evaluate the pipeline against labeled data.

        Returns accuracy metrics and per-example results.
        """
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")

        results = []
        correct = 0
        confidences = []

        for text, true_label in zip(texts, labels):
            analysis = self.analyze(text)
            predicted = analysis["label"]
            is_correct = predicted == true_label

            if is_correct:
                correct += 1

            confidences.append(analysis["confidence"])
            results.append({
                "text": text,
                "true_label": true_label,
                "predicted_label": predicted,
                "correct": is_correct,
                "confidence": analysis["confidence"],
            })

        accuracy = correct / len(labels) if labels else 0.0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        evaluation = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(labels),
            "average_confidence": avg_confidence,
            "results": results,
        }

        logger.info(
            "Pipeline evaluation: accuracy=%.2f (%d/%d), avg_confidence=%.2f",
            accuracy, correct, len(labels), avg_confidence,
        )
        return evaluation
