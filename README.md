# FraudMetrics

A comprehensive Python library for evaluating fraud detection models and binary classification systems with specialized metrics for business impact analysis.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

FraudMetrics provides a complete toolkit for evaluating fraud detection models, combining traditional machine learning metrics with domain-specific measures that focus on business value and operational efficiency.

### Key Features

- **Traditional ML Metrics**: Accuracy, precision, recall, F1-score, ROC AUC, etc.
- **Threshold-Free Evaluation**: ROC curves, Precision-Recall curves, Average Precision
- **Rank-Based Metrics**: Precision@K, Recall@K for top-K predictions
- **Value-Based Analysis**: Monetary value captured, efficiency metrics
- **Built-in Visualizations**: Professional plots for analysis and reporting
- **Robust Validation**: Comprehensive input validation and edge case handling

## Installation

### From PyPI (Recommended)

```bash
pip install fraudmetrics
```

### From Source

```bash
# Clone the repository
git clone https://github.com/yashasnadigsyn/fraudmetrics.git
cd fraudmetrics

# Install in development mode
pip install -e .
```

### Using uv

```bash
# Install from PyPI
uv add fraudmetrics

# Or install from source
git clone https://github.com/yashasnadigsyn/fraudmetrics.git
cd fraudmetrics
uv sync
uv run pip install -e .
```

## 🚀 Quick Start

```python
import numpy as np
from fraudmetrics import (
    get_roc_auc_score,
    get_precision_at_topk,
    value_captured_at_k,
    plot_roc_curve
)

# Sample data
y_true = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1])
y_pred_proba = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.1, 0.9, 0.2, 0.8])
transaction_values = np.array([100, 50, 200, 75, 150, 25, 30, 300, 40, 180])

# Calculate metrics
roc_auc = get_roc_auc_score(y_true, y_pred_proba)
precision_at_5 = get_precision_at_topk(y_true, y_pred_proba, k=5)
value_captured = value_captured_at_k(y_true, y_pred_proba, transaction_values, k=5)

print(f"ROC AUC: {roc_auc:.3f}")
print(f"Precision@5: {precision_at_5:.3f}")
print(f"Value captured in top 5: ${value_captured:.2f}")

# Create visualization
import matplotlib.pyplot as plt
fig = plot_roc_curve(y_true, y_pred_proba)
plt.show()
```

## 📚 Usage Examples

### 1. Thresholded Metrics

Evaluate model performance at specific decision thresholds:

```python
from fraudmetrics import get_binary_confusion_matrix, get_f1_score

# Calculate confusion matrix and F1-score at threshold 0.5
cm = get_binary_confusion_matrix(y_true, y_pred_proba, threshold=0.5)
f1 = get_f1_score(y_true, y_pred_proba, threshold=0.5)

print(f"Confusion Matrix: {cm}")
print(f"F1-Score: {f1:.3f}")
```

### 2. Threshold-Free Metrics

Compare models without committing to a specific threshold:

```python
from fraudmetrics import get_roc_auc_score, get_AP_score

roc_auc = get_roc_auc_score(y_true, y_pred_proba)
ap_score = get_AP_score(y_true, y_pred_proba)

print(f"ROC AUC: {roc_auc:.3f}")
print(f"Average Precision: {ap_score:.3f}")
```

### 3. Rank-Based Metrics

Evaluate performance on top-K predictions:

```python
from fraudmetrics import get_precision_at_topk, get_recall_at_topk

# Evaluate top 100 predictions
precision_at_100 = get_precision_at_topk(y_true, y_pred_proba, k=100)
recall_at_100 = get_recall_at_topk(y_true, y_pred_proba, k=100)

print(f"Precision@100: {precision_at_100:.3f}")
print(f"Recall@100: {recall_at_100:.3f}")
```

### 4. Value-Based Metrics

Analyze business impact with monetary values:

```python
from fraudmetrics import (
    value_captured_at_k,
    proportion_value_captured_at_k,
    value_efficiency_at_k
)

# Calculate value-based metrics
value_captured = value_captured_at_k(y_true, y_pred_proba, transaction_values, k=100)
prop_value = proportion_value_captured_at_k(y_true, y_pred_proba, transaction_values, k=100)
efficiency = value_efficiency_at_k(y_true, y_pred_proba, transaction_values, k=100)

print(f"Value captured: ${value_captured:.2f}")
print(f"Proportion of total fraud value: {prop_value:.3f}")
print(f"Value per alert: ${efficiency:.2f}")
```

### 5. Visualizations

Create professional plots for analysis and reporting:

```python
import matplotlib.pyplot as plt
from fraudmetrics import plot_roc_curve, plot_pr_curve, plot_confusion_matrix

# Create multiple plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ROC Curve
plot_roc_curve(y_true, y_pred_proba, ax=axes[0])

# Precision-Recall Curve
plot_pr_curve(y_true, y_pred_proba, ax=axes[1])

# Confusion Matrix
cm = get_binary_confusion_matrix(y_true, y_pred_proba, threshold=0.5)
plot_confusion_matrix(cm, ["Legitimate", "Fraud"], ax=axes[2])

plt.tight_layout()
plt.show()
```

## 📊 Available Metrics

### Thresholded Metrics
- `get_binary_confusion_matrix()` - Confusion matrix components
- `get_accuracy_score()` - Accuracy
- `get_precision_score()` - Precision
- `get_recall_score()` - Recall/Sensitivity
- `get_specificity_score()` - Specificity
- `get_f1_score()` - F1-score
- `get_classification_error_score()` - Error rate
- `get_fnr_score()` - False Negative Rate
- `get_fpr_score()` - False Positive Rate
- `get_ber_score()` - Balanced Error Rate
- `get_gmean_score()` - Geometric Mean
- `get_npv_score()` - Negative Predictive Value
- `get_false_discovery_rate()` - False Discovery Rate
- `get_false_omission_rate()` - False Omission Rate

### Threshold-Free Metrics
- `get_roc_auc_score()` - Area Under ROC Curve
- `get_AP_score()` - Average Precision
- `get_roc_curve_points()` - ROC curve coordinates
- `get_pr_curve_points()` - Precision-Recall curve coordinates

### Rank-Based Metrics
- `get_precision_at_topk()` - Precision at top K predictions
- `get_recall_at_topk()` - Recall at top K predictions
- `get_card_precision_at_topk()` - Entity-level precision (for card fraud)

### Value-Based Metrics
- `value_captured_at_k()` - Total monetary value captured in top K
- `proportion_value_captured_at_k()` - Proportion of total fraud value captured
- `value_efficiency_at_k()` - Average value per alert investigated

### Visualization Functions
- `plot_roc_curve()` - ROC curve with AUC
- `plot_pr_curve()` - Precision-Recall curve with AP
- `plot_confusion_matrix()` - Confusion matrix heatmap
- `plot_cumulative_gains()` - Cumulative gains chart
- `plot_value_vs_alerts()` - Value captured vs. alerts investigated

## 🔧 Advanced Usage

### Card-Level Analysis

For fraud detection scenarios where you want to evaluate at the card/entity level:

```python
from fraudmetrics import get_card_precision_at_topk

# Card IDs for each transaction
card_ids = ['card_001', 'card_001', 'card_002', 'card_003', 'card_002']

# Calculate card-level precision (aggregates transaction scores per card)
card_precision = get_card_precision_at_topk(
    y_true, y_pred_proba, card_ids, k=10, aggregation_func='max'
)
print(f"Card Precision@10: {card_precision:.3f}")
```

### Comprehensive Model Evaluation

```python
def evaluate_fraud_model(y_true, y_pred_proba, transaction_values, thresholds=[0.3, 0.5, 0.7]):
    """Comprehensive fraud detection model evaluation."""
    
    results = {}
    
    # Threshold-free metrics
    results['roc_auc'] = get_roc_auc_score(y_true, y_pred_proba)
    results['ap_score'] = get_AP_score(y_true, y_pred_proba)
    
    # Thresholded metrics
    for threshold in thresholds:
        results[f'threshold_{threshold}'] = {
            'f1': get_f1_score(y_true, y_pred_proba, threshold=threshold),
            'precision': get_precision_score(y_true, y_pred_proba, threshold=threshold),
            'recall': get_recall_score(y_true, y_pred_proba, threshold=threshold)
        }
    
    # Value-based metrics
    results['value_metrics'] = {
        'value_captured_100': value_captured_at_k(y_true, y_pred_proba, transaction_values, k=100),
        'efficiency_100': value_efficiency_at_k(y_true, y_pred_proba, transaction_values, k=100)
    }
    
    return results
```

## 📋 Requirements

- Python 3.11+
- NumPy >= 2.3.1
- Pandas >= 2.3.0
- Matplotlib >= 3.10.3
- Seaborn >= 0.13.2

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This library is specifically designed for fraud detection and binary classification scenarios. For general multi-class classification, consider using scikit-learn's metrics module.
