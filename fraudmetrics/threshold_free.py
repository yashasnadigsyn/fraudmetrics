## Threshold Free Metrics

import numpy as np
from typing import Union, Any, Tuple
from .utils.validate_inputs import validate_binary_inputs

def get_roc_auc_score(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    pos_label: Any = 1
) -> float:
    """
    get_roc_auc_score(y_true, y_pred_proba, pos_label=1)

    Computes the Area Under the ROC Curve (AUC ROC) using the trapezoidal rule.

    Args:
        y_true (Union[list, np.ndarray]): True labels. Values will be
                                         binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
                                         Values should be in the range [0, 1].
                                         Must be the same length as `y_true`.
        pos_label (Any, optional): The label of the positive class in `y_true`.
                                   Defaults to 1.

    Returns:
        float: The AUC ROC value.
               Returns 1.0 if all true instances belong to the positive class.
               Returns 0.0 if all true instances belong to the negative class.
               Returns np.nan if inputs are empty or ROC curve cannot be determined.
    """
    ## Validate inputs
    y_true_arr, y_pred_proba_arr = validate_binary_inputs(y_true, y_pred_proba, pos_label)

    ## Handle empty case
    if len(y_true_arr) == 0:
        return np.nan

    ## Convert y_true to binary and count positives/negatives
    y_true_binary = (y_true_arr == pos_label).astype(int)
    num_positives = np.sum(y_true_binary == 1)
    num_negatives = len(y_true_binary) - num_positives

    ## Handle edge cases for num_positives or num_negatives
    if num_positives == 0:
        return 0.0
    if num_negatives == 0:
        return 1.0

    ## Sort scores and corresponding true labels
    desc_score_indices = np.argsort(y_pred_proba_arr, kind="mergesort")[::-1]
    y_pred_proba_sorted = y_pred_proba_arr[desc_score_indices]
    y_true_binary_sorted = y_true_binary[desc_score_indices]

    ## Compute TPR and FPR lists
    tp_count = 0
    fp_count = 0
    tpr_list = [0.0] 
    fpr_list = [0.0]
    
    ## Initialize with a value higher than any possible score
    last_score = np.inf 

    for i in range(len(y_pred_proba_sorted)):
        current_score = y_pred_proba_sorted[i]
        if current_score < last_score:
            # Add a point to ROC curve for the completed threshold segment
            tpr_list.append(tp_count / num_positives)
            fpr_list.append(fp_count / num_negatives)
            last_score = current_score

        if y_true_binary_sorted[i] == 1:
            tp_count += 1
        else:
            fp_count += 1

    ## Add the final point to the ROC curve
    tpr_list.append(tp_count / num_positives) ## 1.0
    fpr_list.append(fp_count / num_negatives) ## 1.0

    ## Convert to numpy arrays
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)

    ## Ensure fpr is monotonically increasing for trapezoid rule by taking unique points
    ## sorted by fpr, then tpr.
    unique_points = sorted(list(set(zip(fpr_arr, tpr_arr))))
    if not unique_points:
        return np.nan

    fpr_sorted = np.array([p[0] for p in unique_points])
    tpr_sorted = np.array([p[1] for p in unique_points])

    if len(fpr_sorted) < 2: # Need at least two points for trapezoid rule
        return np.nan

    ## Compute AUC using trapezoidal rule
    roc_auc = np.trapezoid(tpr_sorted, fpr_sorted)

    return float(roc_auc)

def get_AP_score(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    pos_label: Any = 1
) -> float:
    """
    get_AP_score(y_true, y_pred_proba, pos_label=1)

    Calculates the Average Precision (AP), approximating the area under the Precision-Recall curve.
    AP is calculated as Summation[(R(n) - R(n-1)) * P(n)],
    where P(n) and R(n) are precision and recall at the n-th threshold, and R(n) >= R(n-1).

    Args:
        y_true (Union[list, np.ndarray]): True labels. Values will be
                                         binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
                                         Values should be in the range [0, 1].
                                         Must be the same length as `y_true`.
        pos_label (Any, optional): The label of the positive class in `y_true`.
                                   Defaults to 1.

    Returns:
        float: The Average Precision score.
               Returns np.nan if inputs are empty, or if there are no true positive instances.
               Returns 1.0 if all instances are true positives and perfectly ranked.
    """

    ## Validate inputs
    y_true_arr, y_pred_proba_arr = validate_binary_inputs(y_true, y_pred_proba, pos_label)
    
    if len(y_true_arr) == 0:
        return np.nan
    
    ## Convert to binary labels
    y_true_binary = (y_true_arr == pos_label).astype(int)
    num_positives = np.sum(y_true_binary)
    
    ## Handle edge cases
    if num_positives == 0:
        return np.nan
    if num_positives == len(y_true_binary):
        return 1.0
    
    ## Sort by predicted probability (descending) and corresponding true labels
    sorted_indices = np.argsort(y_pred_proba_arr)[::-1]
    y_true_sorted = y_true_binary[sorted_indices]
    y_pred_sorted = y_pred_proba_arr[sorted_indices]
    
    # Find positions where prediction score changes
    score_changes = np.where(np.diff(y_pred_sorted) != 0)[0] + 1
    eval_points = np.concatenate([[0], score_changes, [len(y_true_sorted)]])
    
    # Calculate precision and recall at each evaluation point
    precision_values = []
    recall_values = []
    
    for i, pos in enumerate(eval_points):
        if pos == 0:
            precision_values.append(1.0)
            recall_values.append(0.0)
        else:
            tp = np.sum(y_true_sorted[:pos])
            fp = pos - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / num_positives
            
            precision_values.append(precision)
            recall_values.append(recall)
    
    ## Convert to arrays
    precision_arr = np.array(precision_values)
    recall_arr = np.array(recall_values)
    
    ## Calculate AP using sum (R_i - R_{i-1}) * P_i
    ap_score = 0.0
    for i in range(1, len(recall_arr)):
        delta_recall = recall_arr[i] - recall_arr[i-1]
        ap_score += delta_recall * precision_arr[i]
    
    return float(ap_score)

def get_roc_curve_points(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    pos_label: Any = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the points for the Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true (Union[list, np.ndarray]): True labels.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
        pos_label (Any, optional): The label of the positive class. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            fpr_array: False positive rates.
            tpr_array: True positive rates.
            thresholds_array: Corresponding thresholds (predicted scores).
    """
    y_true_arr, y_pred_proba_arr = validate_binary_inputs(y_true, y_pred_proba, pos_label)

    if len(y_true_arr) == 0:
        return np.array([]), np.array([]), np.array([]) # Empty arrays if no data

    y_true_binary = (y_true_arr == pos_label).astype(int)
    num_positives = np.sum(y_true_binary == 1)
    num_negatives = len(y_true_binary) - num_positives

    if num_negatives == 0:
        return np.array([0., 0., 1.]), np.array([0., 1., 1.]), np.array([np.inf, 0.5, -np.inf])
    if num_positives == 0:
        return np.array([0., 1., 1.]), np.array([0., 0., 1.]), np.array([np.inf, 0.5, -np.inf])

    desc_score_indices = np.argsort(y_pred_proba_arr, kind="mergesort")[::-1]
    y_pred_proba_sorted = y_pred_proba_arr[desc_score_indices]
    y_true_binary_sorted = y_true_binary[desc_score_indices]

    tp_count = 0
    fp_count = 0
    tpr_list = [0.0]
    fpr_list = [0.0]
    thresholds_for_plot = [np.inf]

    last_score = np.inf

    for i in range(len(y_pred_proba_sorted)):
        current_score = y_pred_proba_sorted[i]
        if current_score < last_score:
            thresholds_for_plot.append(last_score)
            tpr_list.append(tp_count / num_positives if num_positives > 0 else 0.0)
            fpr_list.append(fp_count / num_negatives if num_negatives > 0 else 0.0)
            last_score = current_score

        if y_true_binary_sorted[i] == 1:
            tp_count += 1
        else:
            fp_count += 1

    thresholds_for_plot.append(last_score)
    tpr_list.append(tp_count / num_positives if num_positives > 0 else 1.0)
    fpr_list.append(fp_count / num_negatives if num_negatives > 0 else 1.0)


    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)
    thresholds_array = np.array(thresholds_for_plot)
    
    unique_points_with_thresholds = sorted(list(set(zip(fpr_array, tpr_array, thresholds_array))), key=lambda x: (x[0], x[1]))
    
    if not unique_points_with_thresholds:
         return np.array([0.,1.]), np.array([0.,1.]), np.array([np.inf, -np.inf]) 
    
    final_fpr = np.array([p[0] for p in unique_points_with_thresholds])
    final_tpr = np.array([p[1] for p in unique_points_with_thresholds])
    final_thresholds = np.array([p[2] for p in unique_points_with_thresholds])

    return final_fpr, final_tpr, final_thresholds

def get_pr_curve_points(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    pos_label: Any = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the points for the Precision-Recall curve.

    Args:
        y_true (Union[list, np.ndarray]): True labels.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
        pos_label (Any, optional): The label of the positive class. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            precision_array: Precision values.
            recall_array: Recall values.
            thresholds_array: Corresponding thresholds (predicted scores).
    """
    y_true_arr, y_pred_proba_arr = validate_binary_inputs(y_true, y_pred_proba, pos_label)

    if len(y_true_arr) == 0:
        return np.array([]), np.array([]), np.array([])

    y_true_binary = (y_true_arr == pos_label).astype(int)
    num_positives = np.sum(y_true_binary)

    if num_positives == 0:
        return np.array([0.]), np.array([0.]), np.array([np.inf])

    desc_indices = np.argsort(y_pred_proba_arr, kind="mergesort")[::-1]
    y_true_sorted = y_true_binary[desc_indices]
    y_pred_proba_sorted = y_pred_proba_arr[desc_indices]
    
    tp_count = 0
    fp_count = 0
    precision_list = []
    recall_list = []
    thresholds_for_plot = []
    
    distinct_score_indices = np.where(np.diff(y_pred_proba_sorted))[0] + 1
    threshold_operational_indices = np.concatenate(([0], distinct_score_indices, [len(y_true_sorted)]))

    if num_positives > 0:
        precision_list.append(1.0)
        recall_list.append(0.0)
        thresholds_for_plot.append(np.inf if len(y_pred_proba_sorted) == 0 else y_pred_proba_sorted[0] + 1e-9)


    for k_items in threshold_operational_indices:
        if k_items == 0:
            continue

        current_y_true_segment = y_true_sorted[:k_items]
        tp = np.sum(current_y_true_segment)
        current_threshold = y_pred_proba_sorted[k_items-1]


        if (tp + (k_items - tp)) > 0:
            precision = tp / k_items
        else:
            precision = 1.0

        recall = tp / num_positives

        if not precision_list or recall != recall_list[-1] or precision != precision_list[-1]:
            precision_list.append(precision)
            recall_list.append(recall)
            thresholds_for_plot.append(current_threshold)

    precision_array = np.array(precision_list)
    recall_array = np.array(recall_list)
    thresholds_array = np.array(thresholds_for_plot)

    return precision_array, recall_array, thresholds_array