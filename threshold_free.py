## Threshold Free Metrics

import numpy as np
from typing import Union, Any
from utils.validate_inputs import validate_binary_inputs

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