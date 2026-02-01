"""Tests for RAG pipeline."""

import pytest
from unittest.mock import Mock, MagicMock
import numpy as np


@pytest.fixture
def config():
    """Sample configuration for testing."""
    return {
        'rag': {
            'retrieval': {
                'top_k': 5,
                'similarity_threshold': 0.7,
                'rerank': True
            },
            'generation': {
                'max_new_tokens': 512,
                'include_sources': True,
                'system_prompt': 'You are a helpful assistant.'
            }
        }
    }


@pytest.fixture
def mock_components():
    """Create mock components for RAG pipeline."""
    embedding_gen = Mock()
    embedding_gen.encode_single.return_value = np.random.rand(384).astype('float32')
    
    vector_store = Mock()
    vector_store.search.return_value = (
        np.array([0.1, 0.2, 0.3]),  # distances
        np.array([0, 1, 2]),  # indices
        [
            {'text': 'Doc 1', 'filename': 'doc1.pdf', 'page_number': 1},
            {'text': 'Doc 2', 'filename': 'doc2.pdf', 'page_number': 2},
            {'text': 'Doc 3', 'filename': 'doc3.pdf', 'page_number': 1}
        ]
    )
    
    llm = Mock()
    llm.generate.return_value = "This is the generated answer."
    llm.format_prompt.return_value = "Formatted prompt"
    
    return embedding_gen, vector_store, llm


def test_rag_pipeline_initialization(config, mock_components):
    """Test RAG pipeline initialization."""
    from src.rag_pipeline import RAGPipeline
    
    embedding_gen, vector_store, llm = mock_components
    rag = RAGPipeline(config, embedding_gen, vector_store, llm)
    
    assert rag.top_k == 5
    assert rag.similarity_threshold == 0.7
    assert rag.rerank is True


def test_retrieve(config, mock_components):
    """Test document retrieval."""
    from src.rag_pipeline import RAGPipeline
    
    embedding_gen, vector_store, llm = mock_components
    rag = RAGPipeline(config, embedding_gen, vector_store, llm)
    
    texts, metadata, scores = rag.retrieve("Test query")
    
    assert len(texts) == 3
    assert len(metadata) == 3
    assert len(scores) == 3
    embedding_gen.encode_single.assert_called_once()
    vector_store.search.assert_called_once()


def test_query_with_results(config, mock_components):
    """Test full query with results."""
    from src.rag_pipeline import RAGPipeline
    
    embedding_gen, vector_store, llm = mock_components
    rag = RAGPipeline(config, embedding_gen, vector_store, llm)
    
    result = rag.query("What is the answer?")
    
    assert 'answer' in result
    assert 'confidence' in result
    assert 'sources' in result
    assert result['answer'] == "This is the generated answer."
    assert len(result['sources']) == 3


def test_query_no_results(config):
    """Test query when no documents are retrieved."""
    from src.rag_pipeline import RAGPipeline
    
    embedding_gen = Mock()
    embedding_gen.encode_single.return_value = np.random.rand(384).astype('float32')
    
    vector_store = Mock()
    vector_store.search.return_value = (np.array([]), np.array([]), [])
    
    llm = Mock()
    
    rag = RAGPipeline(config, embedding_gen, vector_store, llm)
    result = rag.query("What is the answer?")
    
    assert 'answer' in result
    assert result['confidence'] == 0.0
    assert len(result['sources']) == 0


def test_generate_context(config, mock_components):
    """Test context generation from retrieved documents."""
    from src.rag_pipeline import RAGPipeline
    
    embedding_gen, vector_store, llm = mock_components
    rag = RAGPipeline(config, embedding_gen, vector_store, llm)
    
    texts = ['Document 1 text', 'Document 2 text']
    metadata = [
        {'filename': 'doc1.pdf', 'page_number': 1},
        {'filename': 'doc2.pdf', 'page_number': 2}
    ]
    
    context = rag.generate_context(texts, metadata)
    
    assert 'Document 1' in context
    assert 'Document 2' in context
    assert 'doc1.pdf' in context
    assert 'doc2.pdf' in context
