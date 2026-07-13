"""Persistent local chunk + embedding index.

A deliberately simple on-disk layout — manifest.json (encoder + doc registry),
chunks.jsonl (one EvidenceChunk per line), embeddings.npy (row-aligned with
chunks.jsonl). All files are rewritten through a temp file + rename on every
save; at local corpus scale that buys crash consistency for free and keeps the
reader trivial. The same interfaces map onto OpenSearch when a corpus outgrows
one machine.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from sciqa_schema import EvidenceChunk

MANIFEST_FILE = "manifest.json"
CHUNKS_FILE = "chunks.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"


class IndexManifest(BaseModel):
    """Index-wide invariants: which encoder produced the vectors, which docs are in."""

    version: int = 1
    encoder_name: str
    embedding_dim: int
    chunk_count: int = 0
    documents: dict[str, str] = Field(default_factory=dict)  # doc_id -> content sha256


class LocalIndexStore:
    """Load/append persistent chunk embeddings for one corpus."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def exists(self) -> bool:
        return (self.root / MANIFEST_FILE).is_file()

    def load_manifest(self) -> IndexManifest | None:
        if not self.exists():
            return None
        return IndexManifest.model_validate_json(
            (self.root / MANIFEST_FILE).read_text()
        )

    def doc_id_for_sha(self, sha256: str) -> str | None:
        """Support idempotent ingestion: same bytes -> same doc_id."""
        manifest = self.load_manifest()
        if manifest is None:
            return None
        for doc_id, digest in manifest.documents.items():
            if digest == sha256:
                return doc_id
        return None

    def load(self) -> tuple[list[EvidenceChunk], np.ndarray, IndexManifest]:
        manifest = self.load_manifest()
        if manifest is None:
            raise FileNotFoundError(f"no index at {self.root}")
        chunks = [
            EvidenceChunk.model_validate_json(line)
            for line in (self.root / CHUNKS_FILE).read_text().splitlines()
            if line.strip()
        ]
        embeddings = np.load(self.root / EMBEDDINGS_FILE)
        if len(chunks) != manifest.chunk_count or len(embeddings) != manifest.chunk_count:
            raise ValueError(
                f"index at {self.root} is inconsistent: manifest says "
                f"{manifest.chunk_count} chunks, found {len(chunks)} chunks "
                f"and {len(embeddings)} embedding rows"
            )
        return chunks, embeddings, manifest

    def add_document(
        self,
        *,
        doc_id: str,
        sha256: str,
        chunks: Sequence[EvidenceChunk],
        embeddings: np.ndarray,
        encoder_name: str,
    ) -> None:
        """Append one document's chunks; rejects encoder or identity conflicts."""
        if not chunks:
            raise ValueError("cannot index a document with no chunks")
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be row-aligned")
        if any(chunk.doc_id != doc_id for chunk in chunks):
            raise ValueError("all chunks must belong to doc_id")

        if self.exists():
            existing_chunks, existing_embeddings, manifest = self.load()
            if manifest.encoder_name != encoder_name:
                raise ValueError(
                    f"index was built with encoder '{manifest.encoder_name}', "
                    f"refusing to mix in '{encoder_name}' vectors"
                )
            if doc_id in manifest.documents:
                raise ValueError(f"doc_id '{doc_id}' is already indexed")
            if embeddings.shape[1] != manifest.embedding_dim:
                raise ValueError("embedding dimension mismatch")
            all_chunks = existing_chunks + list(chunks)
            all_embeddings = np.vstack([existing_embeddings, embeddings])
        else:
            manifest = IndexManifest(
                encoder_name=encoder_name, embedding_dim=int(embeddings.shape[1])
            )
            all_chunks = list(chunks)
            all_embeddings = np.asarray(embeddings, dtype=np.float32)

        manifest.documents[doc_id] = sha256
        manifest.chunk_count = len(all_chunks)
        self._save(all_chunks, all_embeddings, manifest)

    def _save(
        self,
        chunks: list[EvidenceChunk],
        embeddings: np.ndarray,
        manifest: IndexManifest,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_atomic(
            CHUNKS_FILE, "".join(chunk.model_dump_json() + "\n" for chunk in chunks)
        )
        tmp_npy = self.root / (EMBEDDINGS_FILE + ".tmp.npy")
        np.save(tmp_npy, embeddings.astype(np.float32))
        os.replace(tmp_npy, self.root / EMBEDDINGS_FILE)
        self._write_atomic(MANIFEST_FILE, json.dumps(manifest.model_dump(), indent=2))

    def _write_atomic(self, filename: str, content: str) -> None:
        tmp = self.root / (filename + ".tmp")
        tmp.write_text(content)
        os.replace(tmp, self.root / filename)
