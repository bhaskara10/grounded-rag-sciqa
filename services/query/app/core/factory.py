"""Build pipeline components from configuration.

The service reads its stack from environment variables so the same code path
serves production (real models) and hermetic tests (deterministic components):

- SCIQA_INDEX_DIR          where the persistent index lives (default: index)
- SCIQA_ENCODER            bi-encoder model name, or "hashing"
- SCIQA_RERANKER           cross-encoder model name, or "lexical"
- SCIQA_MIN_RERANK_SCORE   adaptive selection threshold (default: 0.5)
"""
from __future__ import annotations

import os
from pathlib import Path

from .encoders import DEFAULT_BI_ENCODER, HashingEncoder, SentenceTransformerEncoder, TextEncoder
from .grounding import GroundingVerifier
from .index_store import LocalIndexStore
from .pipeline import PipelineConfig, RagPipeline
from .rerank import DEFAULT_CROSS_ENCODER, CrossEncoderReranker, LexicalReranker, Reranker


def make_encoder(name: str) -> TextEncoder:
    if name == "hashing":
        return HashingEncoder()
    return SentenceTransformerEncoder(name)


def make_reranker(name: str) -> Reranker:
    if name == "lexical":
        return LexicalReranker()
    return CrossEncoderReranker(name)


def config_from_env() -> PipelineConfig:
    config = PipelineConfig()
    min_score = os.environ.get("SCIQA_MIN_RERANK_SCORE")
    if min_score is not None:
        config = config.model_copy(update={"min_rerank_score": float(min_score)})
    return config


def index_dir_from_env() -> Path:
    return Path(os.environ.get("SCIQA_INDEX_DIR", "index"))


def encoder_from_env() -> TextEncoder:
    return make_encoder(os.environ.get("SCIQA_ENCODER", DEFAULT_BI_ENCODER))


def reranker_from_env() -> Reranker:
    return make_reranker(os.environ.get("SCIQA_RERANKER", DEFAULT_CROSS_ENCODER))


def load_indexed_pipeline(
    store: LocalIndexStore,
    encoder: TextEncoder,
    reranker: Reranker,
    config: PipelineConfig,
) -> RagPipeline | None:
    """Build the corpus pipeline from the persistent index, if one exists."""
    if not store.exists():
        return None
    chunks, embeddings, manifest = store.load()
    if manifest.encoder_name != encoder.name:
        raise ValueError(
            f"index at {store.root} was embedded with '{manifest.encoder_name}' "
            f"but the service is configured for '{encoder.name}'; "
            "re-ingest or set SCIQA_ENCODER to match"
        )
    return RagPipeline(
        chunks,
        encoder,
        embeddings=embeddings,
        reranker=reranker,
        verifier=GroundingVerifier(),
        config=config,
    )
