"""Text encoders behind one interface.

Production uses a sentence-transformers bi-encoder on the best available
device (MPS on Apple silicon, CUDA, else CPU). Tests and offline environments
use a deterministic hashing encoder so nothing downloads model weights.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

DEFAULT_BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class TextEncoder(Protocol):
    """Maps texts to L2-normalized embedding rows."""

    @property
    def name(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def pick_device() -> str:
    """Best available torch device: mps (Apple silicon) > cuda > cpu."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentenceTransformerEncoder:
    """Bi-encoder wrapper; loads the model lazily on first encode."""

    def __init__(
        self,
        model_name: str = DEFAULT_BI_ENCODER,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._device = device
        self._model = None

    @property
    def name(self) -> str:
        return self.model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name, device=self._device or pick_device()
            )
        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


class HashingEncoder:
    """Deterministic bag-of-tokens embedding via signed feature hashing.

    No weights, no downloads: token identity is hashed to a dimension and a
    sign, so texts sharing vocabulary land near each other in cosine space.
    Good enough to exercise dense retrieval and fusion logic in tests.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    @property
    def name(self) -> str:
        return f"hashing-{self.dim}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        from .text import token_sequence

        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in token_sequence(text):
                bucket = hash_bucket(token, self.dim)
                matrix[row, bucket.index] += bucket.sign
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.divide(matrix, norms, out=matrix, where=norms > 0)
        return matrix


class _Bucket:
    __slots__ = ("index", "sign")

    def __init__(self, index: int, sign: float) -> None:
        self.index = index
        self.sign = sign


def hash_bucket(token: str, dim: int) -> _Bucket:
    """Stable (index, sign) for a token, independent of PYTHONHASHSEED."""
    import hashlib

    digest = hashlib.md5(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "little") % dim
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return _Bucket(index, sign)
