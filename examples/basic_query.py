"""
Basic query example demonstrating the RAG pipeline.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    load_config, setup_logging, ensure_directories,
    DocumentProcessor, EmbeddingGenerator, VectorStore,
    LLMInterface, RAGPipeline
)
from loguru import logger


def main():
    """Run basic query example."""
    
    print("=" * 60)
    print("Enterprise Document Intelligence - Basic Query Example")
    print("=" * 60)
    
    # Load configuration
    config = load_config('config.yaml')
    setup_logging(config)
    ensure_directories(config)
    
    print("\n[1/6] Initializing components...")
    
    try:
        # Initialize embedding generator
        embedding_gen = EmbeddingGenerator(config)
        
        # Initialize vector store
        vector_store = VectorStore(config, embedding_gen.get_dimension())
        
        # Initialize LLM interface
        print("\n[2/6] Loading LLM (this may take a while)...")
        llm = LLMInterface(config)
        
        # Initialize RAG pipeline
        rag = RAGPipeline(config, embedding_gen, vector_store, llm)
        
        # Check if index exists
        print("\n[3/6] Checking for existing index...")
        index_exists = rag.load_index()
        
        if not index_exists:
            print("\nNo existing index found. Let's create one!")
            print("\n[4/6] Processing documents...")
            
            # Process documents
            doc_processor = DocumentProcessor(config)
            data_dir = config['paths']['data_dir'] + '/sample_docs'
            
            if not os.path.exists(data_dir) or not os.listdir(data_dir):
                print(f"\nNo documents found in {data_dir}")
                print("Please add PDF, TXT, or DOCX files to this directory and run again.")
                return
            
            chunks = doc_processor.process_directory(data_dir)
            
            if not chunks:
                print("No document chunks created. Please check your documents.")
                return
            
            # Index documents
            print(f"\n[5/6] Indexing {len(chunks)} document chunks...")
            rag.index_documents(chunks)
            
            # Save index
            rag.save_index()
            print("Index saved successfully!")
        else:
            print("Loaded existing index")
            stats = vector_store.get_stats()
            print(f"Index contains {stats['doc_count']} document chunks")
        
        # Run query
        print("\n[6/6] Running example queries...")
        print("-" * 60)
        
        queries = [
            "What are the key financial metrics mentioned in the documents?",
            "Summarize the main findings from the documents",
            "What risks are identified in the financial reports?"
        ]
        
        for idx, question in enumerate(queries, 1):
            print(f"\nQuery {idx}: {question}")
            print("-" * 60)
            
            result = rag.query(question)
            
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nConfidence: {result['confidence']:.2f}")
            
            if result.get('sources'):
                print(f"\nSources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['filename']} (Page {source['page_number']}) - Score: {source['score']:.3f}")
            
            print("-" * 60)
        
        print("\n✅ Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Error in example: {e}")
        print(f"\n❌ Error: {e}")
        print("\nPlease check:")
        print("1. You have installed all requirements: pip install -r requirements.txt")
        print("2. You have access to the LLaMA 3 model (or update config.yaml with an available model)")
        print("3. You have added documents to the data/sample_docs directory")


if __name__ == "__main__":
    main()
