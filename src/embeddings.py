"""
Embedding generation module using sentence-transformers.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
import torch
from tqdm import tqdm


class EmbeddingGenerator:
    """
    Generate embeddings for text using sentence-transformers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize embedding generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.emb_config = config['embeddings']
        self.model_name = self.emb_config['model_name']
        self.batch_size = self.emb_config['batch_size']
        self.normalize = self.emb_config['normalize_embeddings']
        
        # Determine device
        device = self.emb_config.get('device', 'auto')
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        
        # Load model
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.emb_config.get('cache_folder')
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            logger.warning("Empty text list provided to encode")
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string
            
        Returns:
            Numpy array embedding
        """
        return self.encode([text], show_progress=False)[0]
    
    def encode_chunks(self, chunks: List[Any]) -> np.ndarray:
        """
        Generate embeddings for DocumentChunk objects.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Numpy array of embeddings
        """
        texts = [chunk.text for chunk in chunks]
        return self.encode(texts, show_progress=True)
    
    def get_dimension(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        if self.normalize:
            # If embeddings are normalized, dot product = cosine similarity
            return np.dot(embedding1, embedding2)
        else:
            # Calculate cosine similarity
            return np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
    
    def batch_similarity(self, query_embedding: np.ndarray, 
                        document_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarity between a query and multiple documents.
        
        Args:
            query_embedding: Query embedding (1D array)
            document_embeddings: Document embeddings (2D array)
            
        Returns:
            Array of similarity scores
        """
        if self.normalize:
            # Dot product for normalized embeddings
            return np.dot(document_embeddings, query_embedding)
        else:
            # Cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            doc_norms = np.linalg.norm(document_embeddings, axis=1)
            return np.dot(document_embeddings, query_embedding) / (doc_norms * query_norm)
