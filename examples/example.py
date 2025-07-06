#!/usr/bin/env python3
"""
FraudMetrics Library Usage Examples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fraudmetrics import (
    # Thresholded metrics
    get_binary_confusion_matrix,
    get_accuracy_score,
    get_precision_score,
    get_recall_score,
    get_f1_score,
    get_specificity_score,
    
    # Threshold-free metrics
    get_roc_auc_score,
    get_AP_score,
    
    # Rank-based metrics
    get_precision_at_topk,
    get_recall_at_topk,
    get_card_precision_at_topk,
    
    # Value-based metrics
    value_captured_at_k,
    proportion_value_captured_at_k,
    value_efficiency_at_k,
    
    # Plotting functions
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_cumulative_gains,
    plot_value_vs_alerts
)

def generate_sample_data(n_samples=1000, fraud_rate=0.1, seed=42):
    """
    Generate sample fraud detection data for demonstration.
    
    Args:
        n_samples: Number of transactions
        fraud_rate: Proportion of fraudulent transactions
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (y_true, y_pred_proba, transaction_values, card_ids)
    """
    np.random.seed(seed)
    
    # Generate true labels (0: legitimate, 1: fraud)
    n_fraud = int(n_samples * fraud_rate)
    y_true = np.concatenate([
        np.ones(n_fraud),  # Fraudulent transactions
        np.zeros(n_samples - n_fraud)  # Legitimate transactions
    ])
    np.random.shuffle(y_true)
    
    # Generate predicted probabilities (simulating model predictions)
    # Fraudulent transactions tend to have higher scores
    y_pred_proba = np.random.beta(2, 5, n_samples)  # Base distribution
    fraud_indices = np.where(y_true == 1)[0]
    y_pred_proba[fraud_indices] = np.random.beta(5, 2, len(fraud_indices))  # Higher scores for fraud
    
    # Generate transaction values (fraudulent transactions tend to be larger)
    transaction_values = np.random.exponential(100, n_samples)
    transaction_values[fraud_indices] *= 2  # Fraud transactions are larger on average
    
    # Generate card IDs (some cards have multiple transactions)
    n_cards = n_samples // 3
    card_ids = np.random.randint(0, n_cards, n_samples)
    
    return y_true, y_pred_proba, transaction_values, card_ids

def example_thresholded_metrics():
    """Demonstrate thresholded metrics with different thresholds."""
    print("=" * 60)
    print("THRESHOLDED METRICS EXAMPLES")
    print("=" * 60)
    
    # Generate sample data
    y_true, y_pred_proba, _, _ = generate_sample_data(n_samples=1000, fraud_rate=0.15)
    
    # Test different thresholds
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        print(f"\nResults for threshold = {threshold}:")
        print("-" * 40)
        
        # Get confusion matrix
        cm = get_binary_confusion_matrix(y_true, y_pred_proba, threshold=threshold)
        print(f"Confusion Matrix: TP={cm['tp']}, FP={cm['fp']}, FN={cm['fn']}, TN={cm['tn']}")
        
        # Calculate various metrics
        accuracy = get_accuracy_score(y_true, y_pred_proba, threshold=threshold)
        precision = get_precision_score(y_true, y_pred_proba, threshold=threshold)
        recall = get_recall_score(y_true, y_pred_proba, threshold=threshold)
        specificity = get_specificity_score(y_true, y_pred_proba, threshold=threshold)
        f1 = get_f1_score(y_true, y_pred_proba, threshold=threshold)
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"Specificity: {specificity:.3f}")
        print(f"F1-Score: {f1:.3f}")

def example_threshold_free_metrics():
    """Demonstrate threshold-free metrics."""
    print("\n" + "=" * 60)
    print("THRESHOLD-FREE METRICS EXAMPLES")
    print("=" * 60)
    
    # Generate sample data
    y_true, y_pred_proba, _, _ = generate_sample_data(n_samples=1000, fraud_rate=0.15)
    
    # Calculate ROC AUC and AP
    roc_auc = get_roc_auc_score(y_true, y_pred_proba)
    ap_score = get_AP_score(y_true, y_pred_proba)
    
    print(f"ROC AUC Score: {roc_auc:.3f}")
    print(f"Average Precision (AP): {ap_score:.3f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    print(f"- ROC AUC of {roc_auc:.3f} indicates {'excellent' if roc_auc > 0.9 else 'good' if roc_auc > 0.8 else 'fair' if roc_auc > 0.7 else 'poor'} discrimination")
    print(f"- AP of {ap_score:.3f} indicates {'excellent' if ap_score > 0.8 else 'good' if ap_score > 0.6 else 'fair' if ap_score > 0.4 else 'poor'} precision-recall performance")

def example_rank_based_metrics():
    """Demonstrate rank-based metrics."""
    print("\n" + "=" * 60)
    print("RANK-BASED METRICS EXAMPLES")
    print("=" * 60)
    
    # Generate sample data
    y_true, y_pred_proba, _, card_ids = generate_sample_data(n_samples=1000, fraud_rate=0.15)
    
    # Test different K values
    k_values = [10, 50, 100, 200]
    
    print("Transaction-level metrics:")
    print("-" * 30)
    for k in k_values:
        precision_at_k = get_precision_at_topk(y_true, y_pred_proba, k=k)
        recall_at_k = get_recall_at_topk(y_true, y_pred_proba, k=k)
        print(f"K={k:3d}: Precision@K={precision_at_k:.3f}, Recall@K={recall_at_k:.3f}")
    
    print("\nCard-level metrics (aggregating transactions per card):")
    print("-" * 50)
    for k in [5, 10, 20, 50]:
        card_precision = get_card_precision_at_topk(
            y_true, y_pred_proba, card_ids, k=k, aggregation_func='max'
        )
        print(f"K={k:2d}: Card Precision@K={card_precision:.3f}")

def example_value_based_metrics():
    """Demonstrate value-based metrics."""
    print("\n" + "=" * 60)
    print("VALUE-BASED METRICS EXAMPLES")
    print("=" * 60)
    
    # Generate sample data
    y_true, y_pred_proba, transaction_values, _ = generate_sample_data(n_samples=1000, fraud_rate=0.15)
    
    # Test different K values
    k_values = [10, 50, 100, 200, 500]
    
    print("Value-based performance analysis:")
    print("-" * 40)
    
    for k in k_values:
        value_captured = value_captured_at_k(y_true, y_pred_proba, transaction_values, k=k)
        prop_value = proportion_value_captured_at_k(y_true, y_pred_proba, transaction_values, k=k)
        efficiency = value_efficiency_at_k(y_true, y_pred_proba, transaction_values, k=k)
        
        print(f"K={k:3d}: Value Captured=${value_captured:8.2f}, "
              f"Proportion={prop_value:.3f}, "
              f"Efficiency=${efficiency:.2f}/alert")

def example_visualizations():
    """Demonstrate plotting capabilities."""
    print("\n" + "=" * 60)
    print("VISUALIZATION EXAMPLES")
    print("=" * 60)
    
    # Generate sample data
    y_true, y_pred_proba, transaction_values, _ = generate_sample_data(n_samples=1000, fraud_rate=0.15)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ROC Curve
    plot_roc_curve(y_true, y_pred_proba, ax=axes[0, 0])
    
    # 2. Precision-Recall Curve
    plot_pr_curve(y_true, y_pred_proba, ax=axes[0, 1])
    
    # 3. Confusion Matrix (using threshold 0.5)
    cm = get_binary_confusion_matrix(y_true, y_pred_proba, threshold=0.5)
    plot_confusion_matrix(
        cm, 
        class_names=["Legitimate", "Fraud"], 
        title="Confusion Matrix (Threshold=0.5)",
        ax=axes[1, 0]
    )
    
    # 4. Cumulative Gains Chart
    plot_cumulative_gains(y_true, y_pred_proba, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('examples/fraud_detection_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved comprehensive analysis plot to 'examples/fraud_detection_analysis.png'")
    
    # Create value analysis plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_value_vs_alerts(y_true, y_pred_proba, transaction_values, ax=ax)
    plt.savefig('examples/value_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved value analysis plot to 'examples/value_analysis.png'")

def example_comprehensive_evaluation():
    """Demonstrate a comprehensive fraud detection model evaluation."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    # Generate sample data
    y_true, y_pred_proba, transaction_values, card_ids = generate_sample_data(n_samples=2000, fraud_rate=0.12)
    
    print("Dataset Summary:")
    print(f"- Total transactions: {len(y_true)}")
    print(f"- Fraudulent transactions: {np.sum(y_true == 1)} ({np.mean(y_true == 1):.1%})")
    print(f"- Total transaction value: ${np.sum(transaction_values):,.2f}")
    print(f"- Average transaction value: ${np.mean(transaction_values):.2f}")
    print(f"- Unique cards: {len(np.unique(card_ids))}")
    
    # Optimal threshold analysis
    print("\nOptimal Threshold Analysis:")
    print("-" * 30)
    thresholds = np.arange(0.1, 0.9, 0.1)
    f1_scores = []
    
    for threshold in thresholds:
        f1 = get_f1_score(y_true, y_pred_proba, threshold=threshold)
        f1_scores.append(f1)
        print(f"Threshold {threshold:.1f}: F1-Score = {f1:.3f}")
    
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print(f"\nOptimal threshold (max F1): {optimal_threshold:.1f}")
    
    # Business impact analysis
    print("\nBusiness Impact Analysis (Top 100 alerts):")
    print("-" * 45)
    k = 100
    
    # Traditional metrics
    precision_at_k = get_precision_at_topk(y_true, y_pred_proba, k=k)
    recall_at_k = get_recall_at_topk(y_true, y_pred_proba, k=k)
    
    # Value metrics
    value_captured = value_captured_at_k(y_true, y_pred_proba, transaction_values, k=k)
    prop_value = proportion_value_captured_at_k(y_true, y_pred_proba, transaction_values, k=k)
    efficiency = value_efficiency_at_k(y_true, y_pred_proba, transaction_values, k=k)
    
    print(f"Precision@100: {precision_at_k:.3f} ({precision_at_k*100:.1f}% of alerts are fraud)")
    print(f"Recall@100: {recall_at_k:.3f} ({recall_at_k*100:.1f}% of all fraud detected)")
    print(f"Value Captured: ${value_captured:,.2f}")
    print(f"Proportion of Total Fraud Value: {prop_value:.3f} ({prop_value*100:.1f}%)")
    print(f"Value per Alert: ${efficiency:.2f}")

def main():
    """Run all examples."""
    print("FraudMetrics Library - Comprehensive Usage Examples")
    print("=" * 60)
    
    # Run all examples
    example_thresholded_metrics()
    example_threshold_free_metrics()
    example_rank_based_metrics()
    example_value_based_metrics()
    example_comprehensive_evaluation()
    example_visualizations()
    
    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("- Use thresholded metrics when you have a specific operating point")
    print("- Use threshold-free metrics (ROC AUC, AP) for model comparison")
    print("- Use rank-based metrics to evaluate top-K performance")
    print("- Use value-based metrics to assess business impact")
    print("- Visualizations help communicate results to stakeholders")

if __name__ == "__main__":
    main() 