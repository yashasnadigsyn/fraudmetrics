## Threshold Free Metrics

import numpy as np

def get_roc_auc_score(y_true: list, y_pred_proba: list, pos_label: int=1):
    """
    get_roc_auc_score(y_true: list, y_pred_proba: list, pos_label: int=1)

    Computes the Area Under the ROC Curve (AUC ROC) using trapezoidal rule.

    Args:
        y_true (list or np.ndarray): True binary labels.
        y_pred_proba (list or np.ndarray): Predicted probabilities for the positive class.
                                         Values should be in the range [0, 1].
                                         Must be the same length as `y_true`.
        pos_label (int, optional): The label of the positive class in `y_true`.
                                   Defaults to 1.

    Returns:
        Float (AUC ROC value)
    """

    ## Check for Input Validation
    if not isinstance(y_true, (list, np.ndarray)):
        raise TypeError(f"y_true must be a list or NumPy array, not {type(y_true)}")
    if not isinstance(y_pred_proba, (list, np.ndarray)):
        raise TypeError(f"y_pred_proba must be a list or NumPy array, not {type(y_pred_proba)}")
    if not isinstance(pos_label, (int)): 
        raise TypeError(f"pos_label must be an integer, not {type(pos_label)}")
    
    ## Convert list to numpy array
    y_true_arr = np.array(y_true)
    y_pred_proba_arr = np.array(y_pred_proba)

    ## Check if both y_true and y_pred_proba of same length
    if len(y_true_arr) != len(y_pred_proba_arr):
        raise ValueError("y_true and y_pred_proba must have the same length.")
    
    ## Return nan if y_true is empty
    if len(y_true_arr) == 0:
        return np.nan
    
    ## Check if probs are between 0 and 1 in y_pred_proba
    if not np.all((y_pred_proba_arr >= 0) & (y_pred_proba_arr <= 1)):
        raise ValueError("All y_pred_proba values must be between 0 and 1 (inclusive).")
    
    ## Convert y_true to binary and check number of positive and negative instances
    y_true_binary = (y_true_arr == pos_label).astype(int)
    num_positives = np.sum(y_true_binary == 1)
    num_negatives = np.sum(y_true_binary == 0)

    if num_negatives == 0:
        return 1.0
        
    if num_positives == 0:
        return 0.0
    
    ## Get unique probs to use as thresholds and in descending order
    desc_scores = np.argsort(y_pred_proba_arr, kind="mergesort")[::-1]
    y_pred_proba_sorted = y_pred_proba_arr[desc_scores]
    y_true_binary_sorted = y_true_binary[desc_scores]

    ## Compute TPR and FPR list
    tp_count = 0
    fp_count = 0

    tpr_list = [0.0]
    fpr_list = [0.0]

    last_score = np.inf

    for i in range(len(y_pred_proba_sorted)):
        current_score = y_pred_proba_sorted[i]
        if current_score < last_score: 
            tpr = tp_count / num_positives
            fpr = fp_count / num_negatives

            tpr_list.append(tpr)
            fpr_list.append(fpr)
            last_score = current_score

        if y_true_binary_sorted[i] == 1:
            tp_count += 1
        else:
            fp_count += 1
            
    tpr_list.append(1.0) 
    fpr_list.append(1.0)

    ## Convert to numpy array
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)

    points = sorted(list(zip(fpr_arr, tpr_arr)))
    fpr_sorted = np.array([p[0] for p in points])
    tpr_sorted = np.array([p[1] for p in points])
    
    if len(fpr_sorted) < 2:
        return np.nan 
        
    roc_auc = np.trapezoid(tpr_sorted, fpr_sorted)
    
    return float(roc_auc)