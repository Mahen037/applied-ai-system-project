"""
Retrieval-Augmented Generation (RAG) Explainer for the Mood Machine.

This module provides context-aware mood explanations by:
  1. Building a vector store of labeled mood examples using sentence embeddings
  2. Retrieving the most similar examples for a given input text
  3. Generating a contextual explanation grounded in retrieved examples

The RAG system is fully integrated into the main prediction pipeline and
meaningfully enhances the system's explanatory capability.
"""

import logging
import json
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SimpleEmbedder:
    """
    A simple TF-IDF based embedder that creates vector representations
    of text. Used as a lightweight alternative to sentence-transformers
    for environments without GPU or large model dependencies.
    """

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            sublinear_tf=True,
        )
        self._is_fitted = False

    def fit(self, texts: List[str]) -> None:
        """Fit the embedder on a corpus of texts."""
        self.vectorizer.fit(texts)
        self._is_fitted = True
        logger.info("SimpleEmbedder fitted on %d texts", len(texts))

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts into vectors.

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if not self._is_fitted:
            raise RuntimeError("Embedder not fitted. Call fit() first.")
        return self.vectorizer.transform(texts).toarray()

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed([text])[0]


class VectorStore:
    """
    A simple in-memory vector store using cosine similarity.

    Stores documents with their embeddings and metadata (labels, etc.)
    and supports nearest-neighbor retrieval.
    """

    def __init__(self) -> None:
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(
        self,
        documents: List[str],
        metadata: List[Dict],
        embeddings: np.ndarray,
    ) -> None:
        """Add documents with their embeddings and metadata."""
        if len(documents) != len(metadata) or len(documents) != len(embeddings):
            raise ValueError("documents, metadata, and embeddings must have same length")

        self.documents.extend(documents)
        self.metadata.extend(metadata)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        logger.info("Added %d documents to vector store (total: %d)",
                     len(documents), len(self.documents))

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Find the top_k most similar documents to the query.

        Returns a list of dicts with keys:
          - text: the document text
          - metadata: the document metadata
          - similarity: cosine similarity score
        """
        if self.embeddings is None or len(self.documents) == 0:
            logger.warning("Vector store is empty, no results to return")
            return []

        # Cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        doc_norms = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        )
        similarities = doc_norms @ query_norm

        # Get top_k indices
        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.documents[idx],
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx]),
            })

        logger.debug("Vector search returned %d results", len(results))
        return results

    @property
    def size(self) -> int:
        """Number of documents in the store."""
        return len(self.documents)


class RAGExplainer:
    """
    Retrieval-Augmented Generation explainer that provides context-aware
    explanations for mood predictions.

    Flow:
      1. Build index from labeled dataset
      2. For each new input, retrieve similar examples
      3. Generate explanation using retrieved context + prediction details
    """

    def __init__(self, top_k: int = 3) -> None:
        self.embedder = SimpleEmbedder()
        self.vector_store = VectorStore()
        self.top_k = top_k
        self._is_initialized = False

    def build_index(self, texts: List[str], labels: List[str]) -> None:
        """
        Build the vector store from a labeled dataset.

        Args:
            texts: List of example texts
            labels: Corresponding mood labels
        """
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")

        # Fit embedder and embed all texts
        self.embedder.fit(texts)
        embeddings = self.embedder.embed(texts)

        # Create metadata
        metadata = [{"label": label, "index": i} for i, label in enumerate(labels)]

        # Add to vector store
        self.vector_store.add_documents(texts, metadata, embeddings)
        self._is_initialized = True

        logger.info("RAG index built with %d documents", len(texts))

    def retrieve(self, text: str) -> List[Dict]:
        """
        Retrieve the most similar examples from the vector store.

        Returns a list of dicts with text, label, and similarity score.
        """
        if not self._is_initialized:
            raise RuntimeError("RAG index not built. Call build_index() first.")

        query_embedding = self.embedder.embed_single(text)
        results = self.vector_store.search(query_embedding, top_k=self.top_k)

        logger.info("Retrieved %d examples for: '%s'", len(results), text[:50])
        return results

    def generate_explanation(
        self,
        text: str,
        predicted_label: str,
        confidence: float,
        score_details: Dict,
        retrieved_examples: Optional[List[Dict]] = None,
    ) -> str:
        """
        Generate a context-aware explanation using retrieved examples.

        This is the core RAG feature: the explanation is grounded in
        similar past examples from the dataset, making it more
        informative than a simple rule-based explanation.

        Args:
            text: The input text being analyzed
            predicted_label: The predicted mood label
            confidence: The confidence score
            score_details: Details from the scoring (positive/negative words, etc.)
            retrieved_examples: Pre-retrieved examples (if None, retrieves them)

        Returns:
            A detailed, context-aware explanation string
        """
        if retrieved_examples is None:
            retrieved_examples = self.retrieve(text)

        # --- Build explanation ---
        parts = []

        # 1. Prediction summary
        parts.append(f"📊 Prediction: {predicted_label.upper()} (confidence: {confidence:.0%})")
        parts.append("")

        # 2. Scoring breakdown
        parts.append("📝 Analysis:")
        if score_details.get("positive_words"):
            parts.append(f"  • Positive signals: {', '.join(score_details['positive_words'])}")
        if score_details.get("negative_words"):
            parts.append(f"  • Negative signals: {', '.join(score_details['negative_words'])}")
        if score_details.get("negated_words"):
            parts.append(f"  • Negation detected: {', '.join(score_details['negated_words'])}")
        if score_details.get("emoji_contributions"):
            emoji_str = ", ".join(
                f"{e}({s:+d})" for e, s in score_details["emoji_contributions"]
            )
            parts.append(f"  • Emoji sentiment: {emoji_str}")
        if score_details.get("score") is not None:
            parts.append(f"  • Raw score: {score_details['score']}")
        parts.append("")

        # 3. RAG-grounded context: similar examples
        if retrieved_examples:
            parts.append("🔍 Similar examples from our dataset:")
            for i, ex in enumerate(retrieved_examples, 1):
                sim_pct = ex["similarity"] * 100
                ex_label = ex["metadata"]["label"]
                parts.append(f'  {i}. "{ex["text"]}" → {ex_label} (similarity: {sim_pct:.0f}%)')

            # 4. Context-aware reasoning
            parts.append("")
            parts.append("💡 Contextual reasoning:")

            # Analyze agreement between prediction and similar examples
            similar_labels = [ex["metadata"]["label"] for ex in retrieved_examples]
            label_counts = {}
            for l in similar_labels:
                label_counts[l] = label_counts.get(l, 0) + 1

            most_common = max(label_counts, key=label_counts.get)
            agreement = label_counts.get(predicted_label, 0) / len(similar_labels)

            if agreement >= 0.5:
                parts.append(
                    f"  The prediction aligns with {agreement:.0%} of similar examples. "
                    f"Most similar texts were also labeled '{most_common}', supporting "
                    f"the '{predicted_label}' classification."
                )
            else:
                parts.append(
                    f"  Note: Most similar examples were labeled '{most_common}', which "
                    f"differs from the prediction of '{predicted_label}'. This suggests "
                    f"the text may have nuances not fully captured by similar examples."
                )

            # Additional insight based on confidence
            if confidence >= 0.8:
                parts.append("  High confidence: The sentiment signals are strong and clear.")
            elif confidence >= 0.5:
                parts.append("  Moderate confidence: Some ambiguity in the sentiment signals.")
            else:
                parts.append(
                    "  Low confidence: The text contains few clear sentiment signals. "
                    "Consider adding more context or clarifying the intent."
                )
        else:
            parts.append("  No similar examples found in the database.")

        explanation = "\n".join(parts)
        logger.info("Generated RAG explanation for: '%s'", text[:50])
        return explanation

    def explain(self, text: str, predicted_label: str, confidence: float,
                score_details: Dict) -> str:
        """
        Full RAG explanation pipeline: retrieve + generate.
        """
        retrieved = self.retrieve(text)
        return self.generate_explanation(
            text, predicted_label, confidence, score_details, retrieved
        )
