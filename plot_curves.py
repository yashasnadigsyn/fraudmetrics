## Functions in this file can be used for plotting
## Library used - Vega Altair

import altair as alt
import pandas as pd
import numpy as np
from threshold_free import *


def plot_roc_curve(y_true: np.ndarray,
                   y_pred_proba: np.ndarray,
                   pos_label: any = 1,
                   title: str = "ROC Curve",
                   width: int = 400,
                   height: int = 400) -> alt.Chart:
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
    width : int, default=400
        Width of the plot in pixels.
    height : int, default=400
        Height of the plot in pixels.

    Returns:
    alt.Chart
        The ROC curve plot.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if len(y_true) == 0 or len(y_pred_proba) == 0:
        return alt.Chart(pd.DataFrame({'text': ["No data to plot ROC curve."]})
                   ).mark_text(size=14).encode(text='text:N'
                   ).properties(width=width, height=height, title=title)

    ## Check for single class in y_true
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


    roc_df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})

    roc_line = alt.Chart(roc_df).mark_line(color='blue', point=True).encode(
        x=alt.X('False Positive Rate:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('True Positive Rate:Q', scale=alt.Scale(domain=[0, 1])),
        tooltip=['False Positive Rate', 'True Positive Rate']
    )

    diagonal = alt.Chart(pd.DataFrame({'x': [0, 1], 'y': [0, 1]})).mark_line(
        color='grey', strokeDash=[3, 3]
    ).encode(x='x:Q', y='y:Q')

    chart_title = f"{title} (AUC: {auc_score_text})"
    chart = (roc_line + diagonal).properties(
        title=chart_title, width=width, height=height
    )

    return chart


def plot_pr_curve(y_true: np.ndarray,
                  y_pred_proba: np.ndarray,
                  pos_label: any = 1,
                  title: str = "Precision-Recall Curve",
                  width: int = 400,
                  height: int = 400) -> alt.Chart:
    """
    Plot the Precision-Recall curve.

    Parameters:
    y_true : np.ndarray
        True binary labels.
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class.
    pos_label : any, default=1
        The label of the positive class.
    title : str, default="ROC Curve"
        Title of the plot.
    width : int, default=400
        Width of the plot in pixels.
    height : int, default=400
        Height of the plot in pixels.

    Returns:
    alt.Chart
        The ROC curve plot.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if len(y_true) == 0 or len(y_pred_proba) == 0:
        return alt.Chart(pd.DataFrame({'text': ["No data to plot PR curve."]})
                   ).mark_text(size=14).encode(text='text:N'
                   ).properties(width=width, height=height, title=title)

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
            precision, recall = np.array([0,0]), np.array([0,1])

    pr_df = pd.DataFrame({'Recall': recall, 'Precision': precision})

    pr_line = alt.Chart(pr_df).mark_line(color='green', point=True).encode(
        x=alt.X('Recall:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Precision:Q', scale=alt.Scale(domain=[0, 1])),
        tooltip=['Recall', 'Precision']
    )

    baseline_value = num_positives / len(y_true_binary) if len(y_true_binary) > 0 else 0
    baseline = alt.Chart(pd.DataFrame({'y_baseline': [baseline_value]})
                        ).mark_rule(color='grey', strokeDash=[3, 3]
                        ).encode(y='y_baseline:Q')

    chart_title = f"{title} (AP: {ap_score_text})"
    chart = (pr_line + baseline).properties(
        title=chart_title, width=width, height=height
    )

    return chart

def plot_confusion_matrix(cm_values: Union[np.ndarray, dict],
                         class_names: list,
                         title: str = "Confusion Matrix",
                         normalize: str = None,
                         cmap: str = "blues",
                         width: int = 300,
                         height: int = 300) -> alt.Chart:
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
    cmap : str, default="blues"
        Colormap for the heatmap.
    width : int, default=300
        Width of the plot.
    height : int, default=300
        Height of the plot.
    
    Returns:
    alt.Chart
        The confusion matrix heatmap plot.
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
    
    # Handle normalization
    cm_normalized = cm.copy().astype(float)
    if normalize == 'true':
        col_sum = cm.sum(axis=0, keepdims=True)
        cm_normalized = np.divide(cm, col_sum, out=np.zeros_like(cm_normalized), where=col_sum != 0)
    elif normalize == 'pred':
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(cm, row_sum, out=np.zeros_like(cm_normalized), where=row_sum != 0)
    elif normalize == 'all':
        total_sum = cm.sum()
        cm_normalized = np.divide(cm, total_sum, out=np.zeros_like(cm_normalized), where=total_sum != 0)
    
    plot_data = []
    for i, actual_class in enumerate(class_names):
        for j, predicted_class in enumerate(class_names):
            value_for_color = cm_normalized[i, j] if normalize else cm[i, j]
            raw_count = cm[i, j]
            text_label = f"{raw_count}"
            if normalize:
                text_label = f"{cm_normalized[i, j]:.2%}\n({raw_count})"
            
            plot_data.append({
                'Actual': actual_class,
                'Predicted': predicted_class,
                'Value': value_for_color,
                'Display Text': text_label
            })
    
    df = pd.DataFrame(plot_data)
    
    heatmap = alt.Chart(df).mark_rect().encode(
        x=alt.X('Predicted:O', title="Predicted", sort=class_names, axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Actual:O', title="Actual", sort=class_names),
        color=alt.Color('Value:Q',
                       scale=alt.Scale(scheme=cmap),
                       legend=alt.Legend(title="Proportion" if normalize else "Count"))
    )
    
    text_color_threshold = 0.5 if normalize else float(cm.max()) / 2.0
    text = alt.Chart(df).mark_text(baseline='middle', size=12, fontWeight='bold').encode(
        x=alt.X('Predicted:O', sort=class_names),
        y=alt.Y('Actual:O', sort=class_names),
        text=alt.Text('Display Text:N'),
        color=alt.condition(
            alt.datum.Value > text_color_threshold,
            alt.value('white'),
            alt.value('black')
        )
    )

    chart = (heatmap + text).properties(
        title=title,
        width=width, 
        height=height
    )
    
    return chart


def plot_cumulative_gains(y_true: np.ndarray,
                          y_pred_proba: np.ndarray,
                          pos_label: any = 1,
                          title: str = "Cumulative Gains Chart",
                          width: int = 450,
                          height: int = 400) -> alt.Chart:
    """
    Plot the Cumulative Gains chart.

    Parameters:
        y_true : np.ndarray
            True binary labels.
        y_pred_proba : np.ndarray
            Predicted probabilities for the positive class.
        pos_label : any, default=1
            The label of the positive class.
        title : str, default="ROC Curve"
            Title of the plot.
        width : int, default=400
            Width of the plot in pixels.
        height : int, default=400
            Height of the plot in pixels.

    Returns:
        alt.Chart
            The ROC curve plot.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if len(y_true) == 0 or len(y_pred_proba) == 0:
        return alt.Chart(pd.DataFrame({'text': ["No data to plot cumulative gains."]})
                   ).mark_text(size=14).encode(text='text:N'
                   ).properties(width=width, height=height, title=title)

    y_true_binary = (y_true == pos_label).astype(int)
    total_positives = np.sum(y_true_binary)
    total_instances = len(y_true_binary)

    if total_instances == 0:
        return alt.Chart(pd.DataFrame({'text': ["No data to plot cumulative gains."]})
                   ).mark_text(size=14).encode(text='text:N'
                   ).properties(width=width, height=height, title=title)

    desc_score_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true_binary[desc_score_indices]

    cumulative_pos = np.cumsum(y_true_sorted)
    
    perc_instances_model = np.concatenate(([0], np.arange(1, total_instances + 1) / total_instances * 100))
    if total_positives > 0:
        perc_pos_captured_model = np.concatenate(([0], cumulative_pos / total_positives * 100))
    else:
        perc_pos_captured_model = np.zeros(total_instances + 1)

    model_df = pd.DataFrame({
        'Percentage of Instances': perc_instances_model,
        'Percentage of Positives Captured': perc_pos_captured_model,
        'Curve': 'Model'
    })

    baseline_df = pd.DataFrame({
        'Percentage of Instances': [0, 100],
        'Percentage of Positives Captured': [0, 100],
        'Curve': 'Baseline (Random)'
    })
    
    if total_positives > 0:
        perfect_x = [0, (total_positives / total_instances) * 100, 100]
        perfect_y = [0, 100, 100]
        perfect_df = pd.DataFrame({
            'Percentage of Instances': perfect_x,
            'Percentage of Positives Captured': perfect_y,
            'Curve': 'Perfect Model'
        })
        combined_df = pd.concat([model_df, baseline_df, perfect_df], ignore_index=True)
    else:
        perfect_df = pd.DataFrame({
            'Percentage of Instances': [0,100],
            'Percentage of Positives Captured': [0,0],
            'Curve': 'Perfect Model (No Positives)'
        })
        combined_df = pd.concat([model_df, baseline_df, perfect_df], ignore_index=True)


    chart = alt.Chart(combined_df).mark_line(point=False).encode(
        x=alt.X('Percentage of Instances:Q', title="Percentage of Instances Targeted", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y('Percentage of Positives Captured:Q', title="Percentage of Positives Captured", scale=alt.Scale(domain=[0, 100])),
        color='Curve:N',
        strokeDash=alt.condition(
            alt.datum.Curve != 'Model',
            alt.value([3, 3]),
            alt.value([0])
        ),
        tooltip=['Curve', 'Percentage of Instances', 'Percentage of Positives Captured']
    ).properties(title=title, width=width, height=height)

    return chart

def plot_value_vs_alerts(y_true: np.ndarray,
                         y_pred_proba: np.ndarray,
                         transaction_values: np.ndarray,
                         pos_label: any = 1,
                         max_alerts_to_plot: int = None,
                         title: str = "Value Captured vs. Alerts Investigated",
                         width: int = 450,
                         height: int = 400) -> alt.Chart:
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
        title : str, default="ROC Curve"
            Title of the plot.
        width : int, default=400
            Width of the plot in pixels.
        height : int, default=400
            Height of the plot in pixels.

    Returns:
        alt.Chart
            The ROC curve plot.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    transaction_values = np.asarray(transaction_values)

    if not (len(y_true) == len(y_pred_proba) == len(transaction_values)):
        raise ValueError("Input arrays y_true, y_pred_proba, and transaction_values must have the same length.")

    if len(y_true) == 0:
        return alt.Chart(pd.DataFrame({'text': ["No data to plot value vs alerts."]})
                   ).mark_text(size=14).encode(text='text:N'
                   ).properties(width=width, height=height, title=title)
                   
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

    alerts_df = pd.DataFrame({
        'Number of Alerts Investigated': plot_num_alerts,
        'Cumulative Value Captured': plot_cum_value
    })

    chart = alt.Chart(alerts_df).mark_line(color='purple', point=True).encode(
        x=alt.X('Number of Alerts Investigated:Q'),
        y=alt.Y('Cumulative Value Captured:Q'),
        tooltip=['Number of Alerts Investigated', 'Cumulative Value Captured']
    ).properties(title=title, width=width, height=height)

    return chart
