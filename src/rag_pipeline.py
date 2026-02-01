"""
RAG Pipeline for document retrieval and generation.
"""

from typing import Dict, Any, List, Tuple, Optional
from loguru import logger
import numpy as np


class RAGPipeline:
    """
    Orchestrate Retrieval Augmented Generation pipeline.
    """
    
    def __init__(self, config: Dict[str, Any], embedding_generator, 
                 vector_store, llm_interface):
        """
        Initialize RAG pipeline.
        
        Args:
            config: Configuration dictionary
            embedding_generator: EmbeddingGenerator instance
            vector_store: VectorStore instance
            llm_interface: LLMInterface instance
        """
        self.config = config
        self.rag_config = config['rag']
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.llm = llm_interface
        
        # Retrieval settings
        self.top_k = self.rag_config['retrieval']['top_k']
        self.similarity_threshold = self.rag_config['retrieval']['similarity_threshold']
        self.rerank = self.rag_config['retrieval']['rerank']
        
        # Generation settings
        self.max_new_tokens = self.rag_config['generation']['max_new_tokens']
        self.include_sources = self.rag_config['generation']['include_sources']
        self.system_prompt = self.rag_config['generation']['system_prompt']
        
        logger.info("RAG Pipeline initialized")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve (optional)
            
        Returns:
            Tuple of (texts, metadata, scores)
        """
        top_k = top_k or self.top_k
        
        # Generate query embedding
        logger.debug(f"Encoding query: {query[:50]}...")
        query_embedding = self.embedding_generator.encode_single(query)
        
        # Search vector store
        distances, indices, metadata = self.vector_store.search(query_embedding, k=top_k)
        
        # Convert distances to similarity scores
        # For L2 distance, convert to similarity (higher is better)
        scores = 1 / (1 + distances)
        
        # Filter by similarity threshold
        filtered_results = []
        for idx, (meta, score) in enumerate(zip(metadata, scores)):
            if score >= self.similarity_threshold:
                filtered_results.append((idx, meta, score))
        
        if not filtered_results:
            logger.warning(f"No documents found above similarity threshold {self.similarity_threshold}")
            return [], [], []
        
        # Extract results
        texts = []
        result_metadata = []
        result_scores = []
        
        for idx, meta, score in filtered_results:
            # The text should be in metadata if we stored it, otherwise we need to reconstruct
            # For this implementation, we'll assume text is in metadata
            if 'text' in meta:
                texts.append(meta['text'])
            else:
                texts.append(meta.get('chunk_text', ''))
            result_metadata.append(meta)
            result_scores.append(score)
        
        logger.info(f"Retrieved {len(texts)} documents for query")
        return texts, result_metadata, result_scores
    
    def rerank_results(self, query: str, texts: List[str], 
                      metadata: List[Dict], scores: List[float]) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Rerank retrieved documents (optional).
        
        Args:
            query: Query string
            texts: Retrieved texts
            metadata: Retrieved metadata
            scores: Initial scores
            
        Returns:
            Reranked (texts, metadata, scores)
        """
        # Simple reranking based on query term presence
        # In production, you might use a cross-encoder model
        query_terms = set(query.lower().split())
        
        reranked = []
        for text, meta, score in zip(texts, metadata, scores):
            # Count query term matches
            text_lower = text.lower()
            term_matches = sum(1 for term in query_terms if term in text_lower)
            
            # Boost score based on term matches
            boosted_score = score * (1 + 0.1 * term_matches)
            reranked.append((text, meta, boosted_score))
        
        # Sort by boosted score
        reranked.sort(key=lambda x: x[2], reverse=True)
        
        texts, metadata, scores = zip(*reranked) if reranked else ([], [], [])
        return list(texts), list(metadata), list(scores)
    
    def generate_context(self, texts: List[str], metadata: List[Dict]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            texts: Retrieved texts
            metadata: Retrieved metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for idx, (text, meta) in enumerate(zip(texts, metadata), 1):
            filename = meta.get('filename', 'Unknown')
            page = meta.get('page_number', 'N/A')
            
            context_parts.append(f"[Document {idx}] (Source: {filename}, Page: {page})\n{text}")
        
        return "\n\n".join(context_parts)
    
    def query(self, question: str, top_k: Optional[int] = None, 
             return_sources: Optional[bool] = None) -> Dict[str, Any]:
        """
        Execute full RAG pipeline for a question.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            return_sources: Whether to return sources
            
        Returns:
            Dictionary with answer and optionally sources
        """
        return_sources = return_sources if return_sources is not None else self.include_sources
        
        logger.info(f"Processing query: {question[:100]}...")
        
        # Retrieve relevant documents
        texts, metadata, scores = self.retrieve(question, top_k=top_k)
        
        if not texts:
            return {
                'answer': "I couldn't find any relevant documents to answer your question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Rerank if enabled
        if self.rerank:
            texts, metadata, scores = self.rerank_results(question, texts, metadata, scores)
        
        # Generate context
        context = self.generate_context(texts, metadata)
        
        # Format prompt
        prompt = self.llm.format_prompt(
            system_prompt=self.system_prompt,
            user_message=question,
            context=context
        )
        
        # Generate answer
        logger.debug("Generating answer...")
        answer = self.llm.generate(prompt, max_new_tokens=self.max_new_tokens)
        
        # Prepare result
        result = {
            'answer': answer,
            'confidence': float(np.mean(scores)) if scores else 0.0
        }
        
        if return_sources:
            result['sources'] = [
                {
                    'filename': meta.get('filename', 'Unknown'),
                    'page_number': meta.get('page_number', 'N/A'),
                    'score': float(score),
                    'text_preview': text[:200] + '...' if len(text) > 200 else text
                }
                for meta, score, text in zip(metadata, scores, texts)
            ]
        
        logger.info("Query processing complete")
        return result
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries.
        
        Args:
            questions: List of questions
            
        Returns:
            List of results
        """
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        return results
    
    def index_documents(self, chunks: List[Any]) -> None:
        """
        Index document chunks into vector store.
        
        Args:
            chunks: List of DocumentChunk objects
        """
        logger.info(f"Indexing {len(chunks)} document chunks...")
        
        # Generate embeddings
        embeddings = self.embedding_generator.encode_chunks(chunks)
        
        # Prepare metadata (including text for retrieval)
        metadata = []
        for chunk in chunks:
            meta = chunk.metadata.copy()
            meta['text'] = chunk.text  # Store text for retrieval
            metadata.append(meta)
        
        # Add to vector store
        self.vector_store.add_embeddings(embeddings, metadata)
        
        logger.info("Document indexing complete")
    
    def save_index(self) -> None:
        """Save the vector index to disk."""
        self.vector_store.save()
        logger.info("Index saved")
    
    def load_index(self) -> bool:
        """
        Load vector index from disk.
        
        Returns:
            True if successful, False otherwise
        """
        success = self.vector_store.load()
        if success:
            logger.info("Index loaded successfully")
        else:
            logger.warning("Failed to load index")
        return success
