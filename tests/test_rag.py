# tests/test_rag.py
"""
Unit tests for the RAG (Retrieval-Augmented Generation) explainer.

Tests cover:
  - Vector store operations (add, search)
  - Embedder functionality
  - RAG index building and retrieval
  - Explanation generation quality
  - Integration of RAG with prediction details
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_explainer import SimpleEmbedder, VectorStore, RAGExplainer
from dataset import SAMPLE_POSTS, TRUE_LABELS


class TestSimpleEmbedder(unittest.TestCase):
    """Tests for the TF-IDF based embedder."""

    def setUp(self):
        self.embedder = SimpleEmbedder()
        self.texts = ["I am happy", "I am sad", "The weather is nice"]
        self.embedder.fit(self.texts)

    def test_fit_and_embed(self):
        """Embedder should produce vectors after fitting."""
        embeddings = self.embedder.embed(self.texts)
        self.assertEqual(len(embeddings), 3)
        self.assertGreater(embeddings.shape[1], 0)

    def test_embed_single(self):
        """Single embed should return a 1D vector."""
        vec = self.embedder.embed_single("I am happy")
        self.assertEqual(len(vec.shape), 1)

    def test_embed_before_fit_raises(self):
        """Embedding before fitting should raise RuntimeError."""
        fresh_embedder = SimpleEmbedder()
        with self.assertRaises(RuntimeError):
            fresh_embedder.embed(["hello"])


class TestVectorStore(unittest.TestCase):
    """Tests for the in-memory vector store."""

    def setUp(self):
        self.store = VectorStore()
        self.embedder = SimpleEmbedder()
        self.texts = ["I love this", "I hate this", "The cat sat"]
        self.embedder.fit(self.texts)
        embeddings = self.embedder.embed(self.texts)
        metadata = [{"label": "positive"}, {"label": "negative"}, {"label": "neutral"}]
        self.store.add_documents(self.texts, metadata, embeddings)

    def test_store_size(self):
        """Store should contain the correct number of documents."""
        self.assertEqual(self.store.size, 3)

    def test_search_returns_results(self):
        """Search should return results."""
        query = self.embedder.embed_single("I love it")
        results = self.store.search(query, top_k=2)
        self.assertEqual(len(results), 2)

    def test_search_has_similarity(self):
        """Search results should include similarity scores."""
        query = self.embedder.embed_single("I love it")
        results = self.store.search(query, top_k=1)
        self.assertIn("similarity", results[0])
        self.assertIsInstance(results[0]["similarity"], float)

    def test_similar_text_ranks_higher(self):
        """More similar text should rank higher in results."""
        query = self.embedder.embed_single("I love everything")
        results = self.store.search(query, top_k=3)
        # "I love this" should be most similar to "I love everything"
        self.assertEqual(results[0]["text"], "I love this")

    def test_empty_store_returns_empty(self):
        """Searching an empty store should return empty list."""
        empty_store = VectorStore()
        import numpy as np
        results = empty_store.search(np.zeros(10), top_k=3)
        self.assertEqual(len(results), 0)


class TestRAGExplainer(unittest.TestCase):
    """Tests for the RAG explainer."""

    @classmethod
    def setUpClass(cls):
        """Build the RAG index once."""
        cls.rag = RAGExplainer(top_k=3)
        cls.rag.build_index(SAMPLE_POSTS, TRUE_LABELS)

    def test_index_built(self):
        """RAG index should be successfully built."""
        self.assertTrue(self.rag._is_initialized)
        self.assertEqual(self.rag.vector_store.size, len(SAMPLE_POSTS))

    def test_retrieve_returns_examples(self):
        """Retrieve should return similar examples."""
        results = self.rag.retrieve("I am feeling very happy today")
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)

    def test_retrieved_examples_have_labels(self):
        """Retrieved examples should include label metadata."""
        results = self.rag.retrieve("This is the worst day ever")
        for r in results:
            self.assertIn("metadata", r)
            self.assertIn("label", r["metadata"])

    def test_explanation_is_string(self):
        """Generated explanation should be a non-empty string."""
        explanation = self.rag.generate_explanation(
            text="I love this",
            predicted_label="positive",
            confidence=0.85,
            score_details={
                "positive_words": ["love"],
                "negative_words": [],
                "negated_words": [],
                "emoji_contributions": [],
                "score": 2,
            },
        )
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 50)

    def test_explanation_includes_prediction(self):
        """Explanation should mention the predicted label."""
        explanation = self.rag.generate_explanation(
            text="I hate this",
            predicted_label="negative",
            confidence=0.9,
            score_details={
                "positive_words": [],
                "negative_words": ["hate"],
                "negated_words": [],
                "emoji_contributions": [],
                "score": -3,
            },
        )
        self.assertIn("NEGATIVE", explanation)

    def test_explanation_includes_similar_examples(self):
        """Explanation should include similar examples section."""
        explanation = self.rag.explain(
            text="I am excited",
            predicted_label="positive",
            confidence=0.8,
            score_details={
                "positive_words": ["excited"],
                "negative_words": [],
                "negated_words": [],
                "emoji_contributions": [],
                "score": 1,
            },
        )
        self.assertIn("Similar examples", explanation)

    def test_retrieve_before_build_raises(self):
        """Retrieving before building index should raise RuntimeError."""
        fresh_rag = RAGExplainer()
        with self.assertRaises(RuntimeError):
            fresh_rag.retrieve("hello")


if __name__ == "__main__":
    unittest.main()
