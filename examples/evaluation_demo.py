"""
Evaluation demonstration example.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    load_config, setup_logging, ensure_directories,
    EmbeddingGenerator, VectorStore, LLMInterface,
    RAGPipeline, Evaluator
)
from loguru import logger


def main():
    """Run evaluation example."""
    
    print("=" * 60)
    print("Enterprise Document Intelligence - Evaluation Demo")
    print("=" * 60)
    
    # Load configuration
    config = load_config('config.yaml')
    setup_logging(config)
    ensure_directories(config)
    
    print("\n[1/5] Initializing components...")
    
    # Initialize components
    embedding_gen = EmbeddingGenerator(config)
    vector_store = VectorStore(config, embedding_gen.get_dimension())
    llm = LLMInterface(config)
    rag = RAGPipeline(config, embedding_gen, vector_store, llm)
    evaluator = Evaluator(config)
    
    # Load index
    print("\n[2/5] Loading index...")
    if not rag.load_index():
        print("\n❌ No index found. Please run batch_process.py first to create an index.")
        return
    
    stats = vector_store.get_stats()
    print(f"Loaded index with {stats['doc_count']} documents")
    
    # Test queries with evaluation
    print("\n[3/5] Running test queries...")
    
    test_cases = [
        {
            'query': "What are the revenue figures?",
            'expected_docs': ['financial_report.pdf'],  # Example
            'expected_answer': "Revenue figures are mentioned in the financial documents."
        },
        {
            'query': "What risks are identified?",
            'expected_docs': ['risk_assessment.pdf'],  # Example
            'expected_answer': "Several risks are identified including market volatility."
        },
        {
            'query': "What is the company outlook?",
            'expected_docs': ['company_overview.pdf'],  # Example
            'expected_answer': "The company outlook is positive with growth expected."
        }
    ]
    
    for idx, test_case in enumerate(test_cases, 1):
        print(f"\n{'-' * 60}")
        print(f"Test Case {idx}: {test_case['query']}")
        print('-' * 60)
        
        # Run query
        result = rag.query(test_case['query'])
        
        print(f"\nAnswer: {result['answer'][:200]}...")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Sources: {len(result.get('sources', []))}")
        
        # Evaluate retrieval
        if result.get('sources'):
            retrieval_metrics = evaluator.evaluate_retrieval(
                result['sources'],
                test_case.get('expected_docs', [])
            )
            print(f"\nRetrieval Metrics:")
            print(f"  Precision: {retrieval_metrics['precision']:.3f}")
            print(f"  Recall: {retrieval_metrics['recall']:.3f}")
            print(f"  F1 Score: {retrieval_metrics['f1_score']:.3f}")
            
            # Evaluate answer quality
            answer_metrics = evaluator.evaluate_answer_quality(
                result['answer'],
                test_case.get('expected_answer', '')
            )
            print(f"\nAnswer Quality Metrics:")
            print(f"  Token Overlap: {answer_metrics['token_overlap']:.3f}")
            print(f"  Jaccard Similarity: {answer_metrics['jaccard_similarity']:.3f}")
            
            # Log result
            evaluation_data = {
                **retrieval_metrics,
                **answer_metrics,
                'accuracy': answer_metrics['token_overlap']  # Simple proxy for accuracy
            }
            evaluator.log_query_result(test_case['query'], result, evaluation_data)
    
    # Generate report
    print(f"\n{'=' * 60}")
    print("[4/5] Generating evaluation report...")
    print('=' * 60)
    
    report = evaluator.generate_report()
    
    print(f"\nEvaluation Summary:")
    print(f"  Total queries: {report['total_queries']}")
    print(f"  Average confidence: {report['average_confidence']:.3f}")
    print(f"  Min confidence: {report['min_confidence']:.3f}")
    print(f"  Max confidence: {report['max_confidence']:.3f}")
    
    if 'accuracy_stats' in report:
        acc_stats = report['accuracy_stats']
        print(f"\nAccuracy Improvement:")
        print(f"  Baseline: {acc_stats['baseline_accuracy']:.1%}")
        print(f"  Current: {acc_stats['current_accuracy']:.1%}")
        print(f"  Improvement: {acc_stats['improvement_percentage']:.1f}%")
        print(f"  Target: {acc_stats['target_accuracy']:.1%}")
        print(f"  Progress: {acc_stats['progress_to_target']:.1%}")
        
        if acc_stats['target_reached']:
            print("\n  ✅ Target accuracy reached!")
        else:
            print(f"\n  📊 {(1 - acc_stats['progress_to_target']) * 100:.1f}% away from target")
    
    # Save reports
    print(f"\n{'=' * 60}")
    print("[5/5] Saving reports...")
    print('=' * 60)
    
    report_path = evaluator.save_report()
    history_path = evaluator.save_metrics_history()
    
    print(f"\nReport saved to: {report_path}")
    print(f"History saved to: {history_path}")
    
    print("\n✅ Evaluation demo completed successfully!")


if __name__ == "__main__":
    main()
