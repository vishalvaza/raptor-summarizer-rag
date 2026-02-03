import pytest
from src.raptor_rag.engine import RaptorSummarizerRAG

def test_initialization():
    engine = RaptorSummarizerRAG(api_key="sk-test")
    assert engine.llm is not None
