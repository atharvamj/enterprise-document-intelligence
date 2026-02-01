"""Tests for vector_store module."""

import pytest
import numpy as np
from src.vector_store import VectorStore


@pytest.fixture
def config():
    """Sample configuration for testing."""
    return {
        'faiss': {
            'index_type': 'IndexFlatL2',
            'index_path': './test_outputs/test.index',
            'metadata_path': './test_outputs/test_metadata.pkl',
            'nlist': 10,
            'nprobe': 5
        }
    }


@pytest.fixture
def vector_store(config):
    """Create VectorStore instance."""
    return VectorStore(config, embedding_dim=384)


def test_vector_store_initialization(vector_store):
    """Test VectorStore initialization."""
    assert vector_store.embedding_dim == 384
    assert vector_store.doc_count == 0
    assert vector_store.index is None


def test_create_index(vector_store):
    """Test creating a FAISS index."""
    vector_store.create_index()
    
    assert vector_store.index is not None
    assert vector_store.index.d == 384


def test_add_embeddings(vector_store):
    """Test adding embeddings to index."""
    # Create sample embeddings
    embeddings = np.random.rand(10, 384).astype('float32')
    metadata = [{'id': i, 'text': f'Document {i}'} for i in range(10)]
    
    vector_store.add_embeddings(embeddings, metadata)
    
    assert vector_store.doc_count == 10
    assert len(vector_store.metadata) == 10


def test_search(vector_store):
    """Test searching the index."""
    # Add some embeddings first
    embeddings = np.random.rand(10, 384).astype('float32')
    metadata = [{'id': i} for i in range(10)]
    vector_store.add_embeddings(embeddings, metadata)
    
    # Search with a query
    query = np.random.rand(384).astype('float32')
    distances, indices, result_metadata = vector_store.search(query, k=5)
    
    assert len(distances) == 5
    assert len(indices) == 5
    assert len(result_metadata) == 5


def test_empty_search(vector_store):
    """Test searching an empty index."""
    query = np.random.rand(384).astype('float32')
    distances, indices, result_metadata = vector_store.search(query, k=5)
    
    assert len(distances) == 0
    assert len(indices) == 0
    assert len(result_metadata) == 0


def test_get_stats(vector_store):
    """Test getting index statistics."""
    embeddings = np.random.rand(5, 384).astype('float32')
    metadata = [{'id': i} for i in range(5)]
    vector_store.add_embeddings(embeddings, metadata)
    
    stats = vector_store.get_stats()
    
    assert stats['doc_count'] == 5
    assert stats['embedding_dim'] == 384
    assert stats['index_type'] is not None


def test_clear(vector_store):
    """Test clearing the index."""
    embeddings = np.random.rand(5, 384).astype('float32')
    metadata = [{'id': i} for i in range(5)]
    vector_store.add_embeddings(embeddings, metadata)
    
    vector_store.clear()
    
    assert vector_store.doc_count == 0
    assert vector_store.index is None
    assert len(vector_store.metadata) == 0
