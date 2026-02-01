"""
Evaluation and metrics tracking module.
"""

from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime
from loguru import logger
import numpy as np


class Evaluator:
    """
    Evaluate RAG pipeline performance and track metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.eval_config = config['evaluation']
        self.output_dir = self.eval_config['output_dir']
        self.baseline_accuracy = self.eval_config.get('baseline_accuracy', 0.68)
        self.target_accuracy = self.eval_config.get('target_accuracy', 0.90)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.metrics_history = []
        logger.info("Evaluator initialized")
    
    def evaluate_retrieval(self, retrieved_docs: List[Dict], 
                          ground_truth_docs: List[str]) -> Dict[str, float]:
        """
        Evaluate retrieval quality.
        
        Args:
            retrieved_docs: List of retrieved documents with metadata
            ground_truth_docs: List of ground truth document IDs
            
        Returns:
            Dictionary of retrieval metrics
        """
        if not retrieved_docs or not ground_truth_docs:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Extract retrieved document IDs
        retrieved_ids = [doc.get('filename', '') for doc in retrieved_docs]
        
        # Calculate metrics
        true_positives = len(set(retrieved_ids) & set(ground_truth_docs))
        precision = true_positives / len(retrieved_ids) if retrieved_ids else 0.0
        recall = true_positives / len(ground_truth_docs) if ground_truth_docs else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def calculate_mrr(self, retrieved_docs: List[Dict], 
                     relevant_doc: str) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc: ID of the relevant document
            
        Returns:
            MRR score
        """
        for idx, doc in enumerate(retrieved_docs, 1):
            if doc.get('filename', '') == relevant_doc:
                return 1.0 / idx
        return 0.0
    
    def calculate_ndcg(self, retrieved_docs: List[Dict], 
                      relevance_scores: List[int], k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevance_scores: Relevance scores for each document (0-3 scale typically)
            k: Cutoff position
            
        Returns:
            NDCG@k score
        """
        if not retrieved_docs or not relevance_scores:
            return 0.0
        
        # DCG
        dcg = 0.0
        for i in range(min(k, len(relevance_scores))):
            rel = relevance_scores[i]
            dcg += (2 ** rel - 1) / np.log2(i + 2)
        
        # IDCG (Ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_scores):
            idcg += (2 ** rel - 1) / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_answer_quality(self, generated_answer: str, 
                               reference_answer: str) -> Dict[str, float]:
        """
        Evaluate answer quality (simplified version).
        For production, use ROUGE or BERT-Score.
        
        Args:
            generated_answer: Generated answer
            reference_answer: Reference answer
            
        Returns:
            Quality metrics
        """
        # Simple token overlap metric
        gen_tokens = set(generated_answer.lower().split())
        ref_tokens = set(reference_answer.lower().split())
        
        if not gen_tokens or not ref_tokens:
            return {'token_overlap': 0.0}
        
        intersection = gen_tokens & ref_tokens
        union = gen_tokens | ref_tokens
        
        overlap = len(intersection) / len(ref_tokens) if ref_tokens else 0.0
        jaccard = len(intersection) / len(union) if union else 0.0
        
        return {
            'token_overlap': overlap,
            'jaccard_similarity': jaccard
        }
    
    def log_query_result(self, query: str, result: Dict[str, Any], 
                        evaluation_data: Optional[Dict] = None) -> None:
        """
        Log query result and evaluation metrics.
        
        Args:
            query: Query string
            result: RAG pipeline result
            evaluation_data: Optional evaluation data (ground truth, etc.)
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': result.get('answer', ''),
            'confidence': result.get('confidence', 0.0),
            'num_sources': len(result.get('sources', []))
        }
        
        if evaluation_data:
            log_entry['evaluation'] = evaluation_data
        
        self.metrics_history.append(log_entry)
        logger.debug(f"Logged query result: {query[:50]}...")
    
    def calculate_accuracy_improvement(self, current_accuracy: float) -> Dict[str, Any]:
        """
        Calculate accuracy improvement over baseline.
        
        Args:
            current_accuracy: Current accuracy score
            
        Returns:
            Dictionary with improvement metrics
        """
        improvement = (current_accuracy - self.baseline_accuracy) / self.baseline_accuracy
        improvement_pct = improvement * 100
        
        progress_to_target = (current_accuracy - self.baseline_accuracy) / \
                           (self.target_accuracy - self.baseline_accuracy)
        
        return {
            'baseline_accuracy': self.baseline_accuracy,
            'current_accuracy': current_accuracy,
            'improvement': improvement,
            'improvement_percentage': improvement_pct,
            'target_accuracy': self.target_accuracy,
            'progress_to_target': progress_to_target,
            'target_reached': current_accuracy >= self.target_accuracy
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate evaluation report from logged metrics.
        
        Returns:
            Report dictionary
        """
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        # Calculate aggregate metrics
        confidences = [entry['confidence'] for entry in self.metrics_history]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(self.metrics_history),
            'average_confidence': avg_confidence,
            'min_confidence': min(confidences) if confidences else 0.0,
            'max_confidence': max(confidences) if confidences else 0.0,
        }
        
        # If we have evaluation data, calculate accuracy
        evaluated_entries = [e for e in self.metrics_history if 'evaluation' in e]
        if evaluated_entries:
            accuracies = [e['evaluation'].get('accuracy', 0) for e in evaluated_entries]
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0
            improvement_stats = self.calculate_accuracy_improvement(avg_accuracy)
            report['accuracy_stats'] = improvement_stats
        
        return report
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """
        Save evaluation report to file.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to saved report
        """
        report = self.generate_report()
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'evaluation_report_{timestamp}.json'
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {filepath}")
        return filepath
    
    def save_metrics_history(self, filename: Optional[str] = None) -> str:
        """
        Save full metrics history to file.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'metrics_history_{timestamp}.json'
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        logger.info(f"Metrics history saved to {filepath}")
        return filepath
