## Functions in this file can be used for plotting
## Library used - Matplotlib and Seaborn

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Union
from .threshold_free import *

def plot_roc_curve(y_true: np.ndarray,
                   y_pred_proba: np.ndarray,
                   pos_label: any = 1,
                   title: str = "ROC Curve",
                   width: int = 6,
                   height: int = 6,
                   ax: plt.Axes = None) -> plt.Figure:
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Parameters:
    y_true : np.ndarray
        True binary labels.
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class.
    pos_label : any, default=1
        The label of the positive class.
    title : str, default="ROC Curve"
        Title of the plot.
    width : int, default=6
        Width of the plot in inches.
    height : int, default=6
        Height of the plot in inches.
    ax : plt.Axes, optional
        Matplotlib axes object to plot on. If None, creates new figure.

    Returns:
    plt.Figure
        The matplotlib figure object.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.get_figure()

    if len(y_true) == 0 or len(y_pred_proba) == 0:
        ax.text(0.5, 0.5, "No data to plot ROC curve.", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig

    # Check for single class in y_true
    unique_true_labels = np.unique(y_true)
    if len(unique_true_labels) < 2:
        auc_score_text = "N/A (single class in y_true)"
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
    else:
        try:
            fpr, tpr, _ = get_roc_curve_points(y_true, y_pred_proba, pos_label=pos_label)
            auc_score = get_roc_auc_score(y_true, y_pred_proba)
            auc_score_text = f"{auc_score:.3f}"
        except ValueError as e:
            print(f"Warning: Could not compute ROC curve points or AUC: {e}")
            auc_score_text = "N/A (computation error)"
            fpr, tpr = np.array([0, 1]), np.array([0, 1])

    # Plot ROC curve
    ax.plot(fpr, tpr, 'b-', marker='o', markersize=4, label=f'ROC Curve (AUC = {auc_score_text})')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'grey', linestyle='--', alpha=0.7, label='Random Classifier')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f"{title} (AUC: {auc_score_text})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pr_curve(y_true: np.ndarray,
                  y_pred_proba: np.ndarray,
                  pos_label: any = 1,
                  title: str = "Precision-Recall Curve",
                  width: int = 6,
                  height: int = 6,
                  ax: plt.Axes = None) -> plt.Figure:
    """
    Plot the Precision-Recall curve.

    Parameters:
    y_true : np.ndarray
        True binary labels.
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class.
    pos_label : any, default=1
        The label of the positive class.
    title : str, default="Precision-Recall Curve"
        Title of the plot.
    width : int, default=6
        Width of the plot in inches.
    height : int, default=6
        Height of the plot in inches.
    ax : plt.Axes, optional
        Matplotlib axes object to plot on. If None, creates new figure.

    Returns:
    plt.Figure
        The matplotlib figure object.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.get_figure()

    if len(y_true) == 0 or len(y_pred_proba) == 0:
        ax.text(0.5, 0.5, "No data to plot PR curve.", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig

    y_true_binary = (y_true == pos_label)
    num_positives = np.sum(y_true_binary)

    if num_positives == 0:
        ap_score_text = "N/A (no positive samples)"
        precision, recall = np.array([0, 0]), np.array([0, 1])
    else:
        try:
            precision, recall, _ = get_pr_curve_points(y_true, y_pred_proba, pos_label=pos_label)
            ap_score = get_AP_score(y_true, y_pred_proba, pos_label=pos_label)
            ap_score_text = f"{ap_score:.3f}"
        except ValueError as e:
            print(f"Warning: Could not compute PR curve points or AP: {e}")
            ap_score_text = "N/A (computation error)"
            precision, recall = np.array([0, 0]), np.array([0, 1])

    # Plot PR curve
    ax.plot(recall, precision, 'g-', marker='o', markersize=4, label=f'PR Curve (AP = {ap_score_text})')
    
    # Plot baseline (random classifier)
    baseline_value = num_positives / len(y_true_binary) if len(y_true_binary) > 0 else 0
    ax.axhline(y=baseline_value, color='grey', linestyle='--', alpha=0.7, 
               label=f'Baseline (AP = {baseline_value:.3f})')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f"{title} (AP: {ap_score_text})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm_values: Union[np.ndarray, dict],
                         class_names: list,
                         title: str = "Confusion Matrix",
                         normalize: str = None,
                         cmap: str = "Blues",
                         width: int = 6,
                         height: int = 6,
                         ax: plt.Axes = None) -> plt.Figure:
    """
    Plot the confusion matrix as a heatmap.
    
    Parameters:
    cm_values : np.ndarray (2x2) or dict
        The confusion matrix. If dict, should have keys 'TN', 'FP', 'FN', 'TP'.
    class_names : list of str (length 2)
        Names of the classes, e.g., ["Non-Fraud", "Fraud"].
        Order should be [Negative Class Name, Positive Class Name] for TN,FP,FN,TP mapping.
    title : str, default="Confusion Matrix"
        Title for the plot.
    normalize : str, optional
        Normalization mode: 'true' (columns), 'pred' (rows), 'all'. If None, raw counts.
    cmap : str, default="Blues"
        Colormap for the heatmap.
    width : int, default=6
        Width of the plot in inches.
    height : int, default=6
        Height of the plot in inches.
    ax : plt.Axes, optional
        Matplotlib axes object to plot on. If None, creates new figure.
    
    Returns:
    plt.Figure
        The matplotlib figure object.
    """
    if isinstance(cm_values, dict):
        tn = cm_values.get('TN', cm_values.get('tn', 0))
        fp = cm_values.get('FP', cm_values.get('fp', 0))
        fn = cm_values.get('FN', cm_values.get('fn', 0))
        tp = cm_values.get('TP', cm_values.get('tp', 0))
        
        cm = np.array([[tn, fp],
                       [fn, tp]])
    elif isinstance(cm_values, np.ndarray) and cm_values.shape == (2, 2):
        cm = cm_values
    else:
        raise ValueError("cm_values must be a 2x2 NumPy array or a dict with confusion matrix keys.")
    
    if len(class_names) != 2:
        raise ValueError("class_names must contain exactly two names for a binary confusion matrix.")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.get_figure()
    
    # Handle normalization
    cm_display = cm.copy().astype(float)
    if normalize == 'true':
        col_sum = cm.sum(axis=0, keepdims=True)
        cm_display = np.divide(cm, col_sum, out=np.zeros_like(cm_display), where=col_sum != 0)
    elif normalize == 'pred':
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm, row_sum, out=np.zeros_like(cm_display), where=row_sum != 0)
    elif normalize == 'all':
        total_sum = cm.sum()
        cm_display = np.divide(cm, total_sum, out=np.zeros_like(cm_display), where=total_sum != 0)
    
    # Create heatmap
    sns.heatmap(cm_display, annot=False, fmt='.2f', cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'}, ax=ax)
    
    # Add custom annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            raw_count = cm[i, j]
            if normalize:
                text = f"{cm_display[i, j]:.2%}\n({raw_count})"
            else:
                text = f"{raw_count}"
            
            # Choose text color based on background
            text_color = 'white' if cm_display[i, j] > cm_display.max() / 2 else 'black'
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', 
                   color=text_color, fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_cumulative_gains(y_true: np.ndarray,
                          y_pred_proba: np.ndarray,
                          pos_label: any = 1,
                          title: str = "Cumulative Gains Chart",
                          width: int = 8,
                          height: int = 6,
                          ax: plt.Axes = None) -> plt.Figure:
    """
    Plot the Cumulative Gains chart.

    Parameters:
        y_true : np.ndarray
            True binary labels.
        y_pred_proba : np.ndarray
            Predicted probabilities for the positive class.
        pos_label : any, default=1
            The label of the positive class.
        title : str, default="Cumulative Gains Chart"
            Title of the plot.
        width : int, default=8
            Width of the plot in inches.
        height : int, default=6
            Height of the plot in inches.
        ax : plt.Axes, optional
            Matplotlib axes object to plot on. If None, creates new figure.

    Returns:
        plt.Figure
            The matplotlib figure object.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.get_figure()

    if len(y_true) == 0 or len(y_pred_proba) == 0:
        ax.text(0.5, 0.5, "No data to plot cumulative gains.", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig

    y_true_binary = (y_true == pos_label).astype(int)
    total_positives = np.sum(y_true_binary)
    total_instances = len(y_true_binary)

    if total_instances == 0:
        ax.text(0.5, 0.5, "No data to plot cumulative gains.", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig

    desc_score_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true_binary[desc_score_indices]

    cumulative_pos = np.cumsum(y_true_sorted)
    
    perc_instances_model = np.concatenate(([0], np.arange(1, total_instances + 1) / total_instances * 100))
    if total_positives > 0:
        perc_pos_captured_model = np.concatenate(([0], cumulative_pos / total_positives * 100))
    else:
        perc_pos_captured_model = np.zeros(total_instances + 1)

    # Plot model curve
    ax.plot(perc_instances_model, perc_pos_captured_model, 'b-', linewidth=2, label='Model')
    
    # Plot baseline (random)
    ax.plot([0, 100], [0, 100], 'grey', linestyle='--', alpha=0.7, label='Baseline (Random)')
    
    # Plot perfect model
    if total_positives > 0:
        perfect_x = [0, (total_positives / total_instances) * 100, 100]
        perfect_y = [0, 100, 100]
        ax.plot(perfect_x, perfect_y, 'r--', alpha=0.7, label='Perfect Model')
    else:
        ax.plot([0, 100], [0, 0], 'r--', alpha=0.7, label='Perfect Model (No Positives)')

    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_xlabel('Percentage of Instances Targeted')
    ax.set_ylabel('Percentage of Positives Captured')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_value_vs_alerts(y_true: np.ndarray,
                         y_pred_proba: np.ndarray,
                         transaction_values: np.ndarray,
                         pos_label: any = 1,
                         max_alerts_to_plot: int = None,
                         title: str = "Value Captured vs. Alerts Investigated",
                         width: int = 8,
                         height: int = 6,
                         ax: plt.Axes = None) -> plt.Figure:
    """
    Plot the Value Captured vs. Number of Alerts Investigated.

    Parameters:
        y_true : np.ndarray
            True binary labels.
        y_pred_proba : np.ndarray
            Predicted probabilities for the positive class.
        transaction_values : np.ndarray
            The values of each transaction.
        pos_label : any, default=1
            The label of the positive class.
        max_alerts_to_plot : int, default=None
            Maximum number of alerts to plot.
        title : str, default="Value Captured vs. Alerts Investigated"
            Title of the plot.
        width : int, default=8
            Width of the plot in inches.
        height : int, default=6
            Height of the plot in inches.
        ax : plt.Axes, optional
            Matplotlib axes object to plot on. If None, creates new figure.

    Returns:
        plt.Figure
            The matplotlib figure object.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    transaction_values = np.asarray(transaction_values)

    if not (len(y_true) == len(y_pred_proba) == len(transaction_values)):
        raise ValueError("Input arrays y_true, y_pred_proba, and transaction_values must have the same length.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.get_figure()

    if len(y_true) == 0:
        ax.text(0.5, 0.5, "No data to plot value vs alerts.", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
                   
    y_true_binary = (y_true == pos_label).astype(int)

    desc_score_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true_binary[desc_score_indices]
    transaction_values_sorted = transaction_values[desc_score_indices]

    value_if_positive = np.where(y_true_sorted == 1, transaction_values_sorted, 0)
    cumulative_value_captured = np.cumsum(value_if_positive)
    
    num_alerts_axis = np.arange(1, len(y_true_sorted) + 1)

    plot_num_alerts = np.concatenate(([0], num_alerts_axis))
    plot_cum_value = np.concatenate(([0.0], cumulative_value_captured))

    if max_alerts_to_plot is not None and max_alerts_to_plot > 0:
        slice_idx = min(max_alerts_to_plot + 1, len(plot_num_alerts))
        plot_num_alerts = plot_num_alerts[:slice_idx]
        plot_cum_value = plot_cum_value[:slice_idx]

    # Plot the curve
    ax.plot(plot_num_alerts, plot_cum_value, 'purple', marker='o', markersize=4, linewidth=2)
    
    ax.set_xlabel('Number of Alerts Investigated')
    ax.set_ylabel('Cumulative Value Captured')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig