"""
Unit tests for RAG Knowledge Base Assistant
"""
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_load_documents_empty_dir(tmp_path):
    """Should return empty list for empty directory."""
    from rag_pipeline import load_documents
    docs = load_documents(str(tmp_path))
    assert isinstance(docs, list)
    assert len(docs) == 0


def test_load_documents_with_txt(tmp_path):
    """Should load a .txt file successfully."""
    from rag_pipeline import load_documents
    sample = tmp_path / "test.txt"
    sample.write_text("This is a test document about AI and RAG pipelines.")
    docs = load_documents(str(tmp_path))
    assert len(docs) >= 1


def test_query_response_structure():
    """Query result should contain answer and sources keys."""
    result = {"answer": "Test answer", "sources": ["doc1.txt"]}
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
