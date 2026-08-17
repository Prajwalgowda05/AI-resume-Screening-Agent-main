"""Embedding services for semantic similarity calculations."""

from abc import ABC, abstractmethod
import re
from typing import List
import numpy as np


class BaseEmbeddingService(ABC):
    """Abstract base class for text embedding and semantic similarity services."""

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Generate a dense vector embedding for the input text.

        Args:
            text: Input text string.

        Returns:
            List of floats representing the embedding vector.
        """
        pass

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two text strings.

        Args:
            text_a: First text string.
            text_b: Second text string.

        Returns:
            Cosine similarity score between -1.0 and 1.0 (typically 0.0 to 1.0).
        """
        if not text_a or not text_a.strip() or not text_b or not text_b.strip():
            return 0.0

        vec_a = np.array(self.get_embedding(text_a), dtype=np.float32)
        vec_b = np.array(self.get_embedding(text_b), dtype=np.float32)

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        cosine_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        return max(-1.0, min(1.0, cosine_sim))


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """SentenceTransformers embedding service using local open-source models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with specified HuggingFace/SentenceTransformers model name."""
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy load the sentence transformer model upon first call."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is not installed. Please install it using "
                    "'pip install sentence-transformers' or use MockEmbeddingService."
                ) from e
        return self._model

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector using sentence-transformers."""
        if not text or not text.strip():
            return [0.0] * 384
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class MockEmbeddingService(BaseEmbeddingService):
    """Deterministic token/n-gram based mock embedding service for lightweight testing."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def get_embedding(self, text: str) -> List[float]:
        """Generate a deterministic frequency-based vector representation."""
        if not text or not text.strip():
            return [0.0] * self.dim

        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"\b[A-Za-z0-9\+\#\.\-]+\b", text.lower())

        for token in tokens:
            # Deterministic hash to bucket
            bucket = hash(token) % self.dim
            vector[bucket] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()


def get_default_embedding_service(
    use_mock: bool = False, model_name: str = "all-MiniLM-L6-v2"
) -> BaseEmbeddingService:
    """Retrieve the active embedding service instance."""
    if use_mock:
        return MockEmbeddingService()
    try:
        import sentence_transformers

        return SentenceTransformerEmbeddingService(model_name=model_name)
    except ImportError:
        return MockEmbeddingService()
