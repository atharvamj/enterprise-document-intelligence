"""
FAISS vector store for semantic search and document retrieval.
"""

import os
import pickle
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
from loguru import logger


class VectorStore:
    """
    Manage FAISS index for document embeddings.
    """
    
    def __init__(self, config: Dict[str, Any], embedding_dim: int):
        """
        Initialize vector store.
        
        Args:
            config: Configuration dictionary
            embedding_dim: Dimension of embeddings
        """
        self.config = config
        self.faiss_config = config['faiss']
        self.embedding_dim = embedding_dim
        self.index_path = self.faiss_config['index_path']
        self.metadata_path = self.faiss_config['metadata_path']
        
        # Initialize index
        self.index = None
        self.metadata = []
        self.doc_count = 0
        
        logger.info(f"VectorStore initialized with dimension {embedding_dim}")
    
    def create_index(self, index_type: Optional[str] = None) -> None:
        """
        Create a new FAISS index.
        
        Args:
            index_type: Type of index (IndexFlatL2, IndexIVFFlat, etc.)
        """
        if index_type is None:
            index_type = self.faiss_config['index_type']
        
        logger.info(f"Creating FAISS index: {index_type}")
        
        if index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif index_type == "IndexFlatIP":
            # Inner product (for normalized embeddings, equivalent to cosine)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif index_type == "IndexIVFFlat":
            # Inverted file index for faster search on large datasets
            nlist = self.faiss_config.get('nlist', 100)
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        elif index_type == "IndexHNSWFlat":
            # Hierarchical Navigable Small World graph
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            logger.warning(f"Unknown index type {index_type}, using IndexFlatL2")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        logger.info(f"FAISS index created: {type(self.index).__name__}")
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Numpy array of embeddings (n_docs x embedding_dim)
            metadata: List of metadata dictionaries for each embedding
        """
        if self.index is None:
            self.create_index()
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Train index if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        self.doc_count += len(embeddings)
        
        logger.info(f"Added {len(embeddings)} embeddings to index. Total: {self.doc_count}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        if self.index is None or self.doc_count == 0:
            logger.warning("Index is empty, cannot search")
            return np.array([]), np.array([]), []
        
        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Set nprobe for IVF index
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.faiss_config.get('nprobe', 10)
        
        # Search
        k = min(k, self.doc_count)  # Can't return more results than documents
        distances, indices = self.index.search(query_embedding, k)
        
        # Get metadata for results
        result_metadata = [self.metadata[idx] for idx in indices[0] if idx < len(self.metadata)]
        
        logger.debug(f"Search completed: {len(result_metadata)} results")
        return distances[0], indices[0], result_metadata
    
    def save(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None) -> None:
        """
        Save index and metadata to disk.
        
        Args:
            index_path: Path to save index (optional)
            metadata_path: Path to save metadata (optional)
        """
        if self.index is None:
            logger.warning("No index to save")
            return
        
        index_path = index_path or self.index_path
        metadata_path = metadata_path or self.metadata_path
        
        # Create directory if needed
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, index_path)
        logger.info(f"FAISS index saved to {index_path}")
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None) -> bool:
        """
        Load index and metadata from disk.
        
        Args:
            index_path: Path to load index from (optional)
            metadata_path: Path to load metadata from (optional)
            
        Returns:
            True if successful, False otherwise
        """
        index_path = index_path or self.index_path
        metadata_path = metadata_path or self.metadata_path
        
        try:
            # Load index
            self.index = faiss.read_index(index_path)
            logger.info(f"FAISS index loaded from {index_path}")
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Metadata loaded from {metadata_path}")
            
            self.doc_count = self.index.ntotal
            logger.info(f"Loaded index with {self.doc_count} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def clear(self) -> None:
        """Clear the index and metadata."""
        self.index = None
        self.metadata = []
        self.doc_count = 0
        logger.info("Index cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'doc_count': self.doc_count,
            'embedding_dim': self.embedding_dim,
            'index_type': type(self.index).__name__ if self.index else None,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }
