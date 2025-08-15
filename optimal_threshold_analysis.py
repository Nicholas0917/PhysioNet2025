#!/usr/bin/env python3
"""
Analyze transformer training results to find optimal threshold.
This will help understand why F1=0 and suggest better thresholds.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve
import matplotlib.pyplot as plt

def analyze_threshold_performance():
    """
    Simulate the transformer results to find optimal thresholds
    """
    # Simulate your training data distribution
    # ~8% positive class, 92% negative class
    n_samples = 10000
    n_positive = int(0.08 * n_samples)
    n_negative = n_samples - n_positive
    
    # Create simulated probabilities based on your results
    # Positive samples: higher probability (but still around 0.34-0.35)
    pos_probs = np.random.normal(0.35, 0.05, n_positive)
    pos_probs = np.clip(pos_probs, 0.01, 0.99)
    
    # Negative samples: slightly lower probability
    neg_probs = np.random.normal(0.33, 0.05, n_negative)
    neg_probs = np.clip(neg_probs, 0.01, 0.99)
    
    # Combine
    y_true = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])
    y_probs = np.concatenate([pos_probs, neg_probs])
    
    # Find optimal thresholds
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores = []
    precisions = []
    recalls = []
    
    print("Threshold Analysis Results:")
    print("=" * 50)
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        
        if thresh in [0.5, 0.3, 0.2, 0.1]:
            print(f"Threshold {thresh:.2f}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    print(f"\nOptimal Threshold: {optimal_threshold:.3f}")
    print(f"Optimal F1 Score: {optimal_f1:.4f}")
    print(f"Standard 0.5 Threshold F1: {f1_scores[45]:.4f}")  # 0.5 is at index 45
    
    print(f"\nWhy your F1=0:")
    print(f"- Your avg probability: ~0.34")
    print(f"- Standard threshold: 0.50")
    print(f"- Since 0.34 < 0.50, all predictions = negative")
    print(f"- Result: No true positives, F1 = 0")
    
    print(f"\nRecommended fixes:")
    print(f"1. Use threshold ~{optimal_threshold:.3f} instead of 0.5")
    print(f"2. Implement threshold optimization in validation")
    print(f"3. Consider class weights or cost-sensitive learning")

if __name__ == "__main__":
    analyze_threshold_performance()
