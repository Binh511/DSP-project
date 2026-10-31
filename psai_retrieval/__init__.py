"""Offline evidence retrieval, separate from frozen classifier/RAG studies."""

from .runtime import build_evidence, build_index, load_index, retrieve_document, search_index

__all__ = ["build_evidence", "build_index", "load_index", "retrieve_document", "search_index"]
