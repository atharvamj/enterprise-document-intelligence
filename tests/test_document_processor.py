"""Tests for document_processor module."""

import pytest
import os
from pathlib import Path
from src.document_processor import DocumentProcessor, DocumentChunk


@pytest.fixture
def config():
    """Sample configuration for testing."""
    return {
        'document_processing': {
            'chunk_size': 100,
            'chunk_overlap': 20,
            'min_chunk_size': 30,
            'supported_formats': ['pdf', 'txt', 'docx']
        }
    }


@pytest.fixture
def doc_processor(config):
    """Create DocumentProcessor instance."""
    return DocumentProcessor(config)


def test_document_processor_initialization(doc_processor, config):
    """Test DocumentProcessor initialization."""
    assert doc_processor.chunk_size == config['document_processing']['chunk_size']
    assert doc_processor.chunk_overlap == config['document_processing']['chunk_overlap']


def test_chunk_text(doc_processor):
    """Test text chunking functionality."""
    text = "This is a test document. " * 20  # Create text longer than chunk_size
    metadata = {'filename': 'test.txt', 'page_number': 1}
    
    chunks = doc_processor.chunk_text(text, metadata)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    assert all(chunk.metadata['filename'] == 'test.txt' for chunk in chunks)


def test_chunk_overlap(doc_processor):
    """Test that chunks have proper overlap."""
    text = "A" * 200  # Simple repeating text
    metadata = {'filename': 'test.txt'}
    
    chunks = doc_processor.chunk_text(text, metadata)
    
    if len(chunks) > 1:
        # Check that chunks overlap
        chunk1_end = chunks[0].metadata['chunk_end']
        chunk2_start = chunks[1].metadata['chunk_start']
        assert chunk1_end > chunk2_start  # There should be overlap


def test_load_txt(doc_processor, tmp_path):
    """Test loading text files."""
    # Create temporary text file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document.")
    
    pages = doc_processor.load_txt(str(test_file))
    
    assert len(pages) == 1
    assert pages[0]['text'] == "This is a test document."
    assert pages[0]['filename'] == 'test.txt'


def test_process_document_txt(doc_processor, tmp_path):
    """Test processing a complete text document."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a longer test document. " * 10)
    
    chunks = doc_processor.process_document(str(test_file))
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)


def test_min_chunk_size(doc_processor):
    """Test that chunks below minimum size are not created."""
    text = "Short"
    metadata = {'filename': 'test.txt'}
    
    chunks = doc_processor.chunk_text(text, metadata)
    
    # Short text might not create any chunks if below min_chunk_size
    assert all(len(chunk.text) >= doc_processor.min_chunk_size for chunk in chunks)
