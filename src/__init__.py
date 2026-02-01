"""Enterprise Document Intelligence Package"""

__version__ = "1.0.0"
__author__ = "Enterprise Document Intelligence Team"

from .utils import load_config, setup_logging, ensure_directories
from .document_processor import DocumentProcessor, DocumentChunk
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .llm_interface import LLMInterface
from .rag_pipeline import RAGPipeline
from .evaluator import Evaluator

__all__ = [
    'load_config',
    'setup_logging',
    'ensure_directories',
    'DocumentProcessor',
    'DocumentChunk',
    'EmbeddingGenerator',
    'VectorStore',
    'LLMInterface',
    'RAGPipeline',
    'Evaluator'
]
