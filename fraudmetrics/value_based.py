## Value Based Metrics

from typing import Union, Any, Tuple
import numpy as np
from .utils.validate_inputs import _validate_value_inputs

def value_captured_at_k(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    transaction_values: Union[list, np.ndarray],
    k: int,
    pos_label: Any = 1
) -> float:
    """
    Calculates the total monetary value of true positives
    captured within the top K instances ranked by their scores.

    Args:
        y_true (Union[list, np.ndarray]): 
            True binary labels.
        y_pred_proba (Union[list, np.ndarray]): 
            Predicted probabilities or scores for the positive class. Higher scores
            indicate higher confidence in the instance being positive.
        transaction_values (Union[list, np.ndarray]): 
            Monetary value associated with each instance. Must be the same
            length as y_true and y_pred_scores.
        k (int): 
            The number of top-ranked instances to consider.
            If k is 0, 0.0 is returned.
            If k is larger than the total number of instances, it will be capped.
        pos_label (Any, optional): 
            The label of the positive class in `y_true`. Defaults to 1.

    Returns:
        float: The total monetary value of true positives captured in the top K.
               Returns 0.0 if k=0 or inputs are empty.
    """
    try:
        y_true_arr, y_scores_arr, trans_vals_arr, validated_k = _validate_value_inputs(
            y_true, y_pred_proba, transaction_values, k
        )
    except (TypeError, ValueError):
        raise  # Re-raise validation errors
    
    ## Handle empty inputs or k=0
    if len(y_true_arr) == 0 or k == 0:
        return 0.0

    ## Convert y_true to binary
    y_true_binary = (y_true_arr == pos_label).astype(int)
    
    ## Sort instances by predicted scores in descending order
    desc_score_indices = np.argsort(y_scores_arr, kind="mergesort")[::-1]
    y_true_binary_sorted = y_true_binary[desc_score_indices]
    trans_vals_sorted = trans_vals_arr[desc_score_indices]

    ## Select the top K instances
    y_true_topk = y_true_binary_sorted[:validated_k]
    trans_vals_topk = trans_vals_sorted[:validated_k]

    ## Calculate value captured
    value_captured = np.sum(trans_vals_topk[y_true_topk == 1])
    
    return float(value_captured)


def proportion_value_captured_at_k(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    transaction_values: Union[list, np.ndarray],
    k: int,
    pos_label: Any = 1
) -> float:
    """
    Calculates the proportion of total positive monetary value captured
    within the top K instances.

    Args:
        y_true (Union[list, np.ndarray]): True binary labels.
        y_pred_proba (Union[list, np.ndarray]): Predicted scores.
        transaction_values (Union[list, np.ndarray]): Monetary value per instance.
        k (int): Number of top-ranked instances.
        pos_label (Any, optional): Label for the positive class. Defaults to 1.

    Returns:
        float: Proportion of total positive value captured in the top K.
               Returns np.nan if total value of actual positives is zero or
               if k=0 or inputs are empty.
    """
    try:
        y_true_arr, _, trans_vals_arr, _ = _validate_value_inputs(
            y_true, y_pred_proba, transaction_values, k
        )
    except (TypeError, ValueError):
        raise  # Re-raise validation errors
    
    ## Handle empty inputs or k=0
    if len(y_true_arr) == 0 or k == 0:
        return np.nan

    ## Binarize y_true to identify all actual positives
    y_true_binary = (y_true_arr == pos_label).astype(int)
    
    ## Calculate the total value of all actual positive instances in the dataset
    total_positive_value = np.sum(trans_vals_arr[y_true_binary == 1])

    if total_positive_value == 0:
        return np.nan 

    ## Calculate value captured in top K
    val_cap_at_k = value_captured_at_k(
        y_true, y_pred_proba, transaction_values, k, pos_label
    )

    proportion = val_cap_at_k / total_positive_value
    
    return float(proportion)


def value_efficiency_at_k(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    transaction_values: Union[list, np.ndarray],
    k: int,
    pos_label: Any = 1
) -> float:
    """
    Calculates the average monetary value captured per instance investigated,
    when considering the top K instances.

    Args:
        y_true (Union[list, np.ndarray]): True binary labels.
        y_pred_proba (Union[list, np.ndarray]): Predicted scores.
        transaction_values (Union[list, np.ndarray]): Monetary value per instance.
        k (int): Number of top-ranked instances to investigate.
        pos_label (Any, optional): Label for the positive class. Defaults to 1.

    Returns:
        float: Average value captured per instance in the top K.
               Returns np.nan if k=0. Returns 0.0 if inputs are empty.
    """
    try:
        y_true_arr, _, _, _ = _validate_value_inputs(
            y_true, y_pred_proba, transaction_values, k
        )
    except (TypeError, ValueError):
        raise  # Re-raise validation errors
    
    ## Handle empty inputs
    if len(y_true_arr) == 0:
        return 0.0 

    ## Handle k=0
    if k == 0:
        return np.nan

    ## Calculate value captured in top K
    val_cap_at_k = value_captured_at_k(
        y_true, y_pred_proba, transaction_values, k, pos_label
    )

    efficiency = val_cap_at_k / k
    
    return float(efficiency)