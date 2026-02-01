"""
Batch processing example for large document collections.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    load_config, setup_logging, ensure_directories,
    DocumentProcessor, EmbeddingGenerator, VectorStore, RAGPipeline
)
from loguru import logger


def main():
    """Run batch processing example."""
    
    print("=" * 60)
    print("Enterprise Document Intelligence - Batch Processing")
    print("=" * 60)
    
    # Load configuration
    config = load_config('config.yaml')
    setup_logging(config)
    ensure_directories(config)
    
    print("\n[1/4] Initializing components...")
    
    # Initialize components (no LLM needed for just indexing)
    embedding_gen = EmbeddingGenerator(config)
    vector_store = VectorStore(config, embedding_gen.get_dimension())
    doc_processor = DocumentProcessor(config)
    
    # Set up data directory
    data_dir = config['paths']['data_dir'] + '/sample_docs'
    
    if not os.path.exists(data_dir):
        print(f"\nCreated directory: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
    
    # Count documents
    supported_formats = config['document_processing']['supported_formats']
    files = list(Path(data_dir).rglob('*'))
    doc_files = [f for f in files if f.suffix.lower().strip('.') in supported_formats]
    
    print(f"\nFound {len(doc_files)} documents in {data_dir}")
    
    if not doc_files:
        print("\nNo documents found!")
        print("Please add PDF, TXT, or DOCX files to:")
        print(f"  {data_dir}")
        return
    
    # Process documents
    print(f"\n[2/4] Processing {len(doc_files)} documents...")
    chunks = doc_processor.process_directory(data_dir, recursive=True)
    
    if not chunks:
        print("No chunks created from documents.")
        return
    
    print(f"\nCreated {len(chunks)} document chunks")
    print(f"Average chunk length: {sum(len(c.text) for c in chunks) / len(chunks):.0f} characters")
    
    # Generate embeddings
    print(f"\n[3/4] Generating embeddings for {len(chunks)} chunks...")
    embeddings = embedding_gen.encode_chunks(chunks)
    print(f"Generated embeddings with dimension {embeddings.shape[1]}")
    
    # Build index
    print(f"\n[4/4] Building FAISS index...")
    metadata = []
    for chunk in chunks:
        meta = chunk.metadata.copy()
        meta['text'] = chunk.text
        metadata.append(meta)
    
    vector_store.create_index()
    vector_store.add_embeddings(embeddings, metadata)
    
    # Save index
    vector_store.save()
    
    # Print statistics
    stats = vector_store.get_stats()
    print("\n" + "=" * 60)
    print("Index Statistics:")
    print("=" * 60)
    print(f"Total documents indexed: {stats['doc_count']}")
    print(f"Embedding dimension: {stats['embedding_dim']}")
    print(f"Index type: {stats['index_type']}")
    print(f"Index trained: {stats['is_trained']}")
    
    # Show sample documents
    print("\n" + "=" * 60)
    print("Sample Document Chunks:")
    print("=" * 60)
    for i, chunk in enumerate(chunks[:3], 1):
        meta = chunk.metadata
        print(f"\n{i}. {meta.get('filename', 'Unknown')} (Page {meta.get('page_number', 'N/A')})")
        print(f"   Text: {chunk.text[:150]}...")
    
    print("\n✅ Batch processing completed successfully!")
    print(f"\nIndex saved to: {vector_store.index_path}")
    print("You can now run queries using basic_query.py")


if __name__ == "__main__":
    main()
