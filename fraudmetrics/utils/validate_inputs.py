import numpy as np
from typing import Union, Tuple, Optional, Any
import warnings

def validate_binary_inputs(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    pos_label: Any,
    threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates inputs for binary classification metrics.

    Args:
        y_true (Union[list, np.ndarray]): True labels.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
        pos_label (Any): The label of the positive class in `y_true`.
        threshold (Optional[float]): The decision threshold. If provided, it's validated.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Validated y_true and y_pred_proba as NumPy arrays.

    Raises:
        TypeError: If inputs are not of the expected types.
        ValueError: If inputs have invalid values (e.g., length mismatch, probabilities out of range).
    """

    ## Check for Input Types
    if not isinstance(y_true, (list, np.ndarray)):
        raise TypeError(f"y_true must be a list or NumPy array, not {type(y_true)}")
    if not isinstance(y_pred_proba, (list, np.ndarray)):
        raise TypeError(f"y_pred_proba must be a list or NumPy array, not {type(y_pred_proba)}")
    if not isinstance(pos_label, (int, str, bool)):
        raise TypeError(f"pos_label must be an integer, string, or boolean, not {type(pos_label)}")
    if threshold is not None and not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be a numeric value, not {type(threshold)}")

    ## Convert list to numpy array
    y_true_arr = np.array(y_true)
    y_pred_proba_arr = np.array(y_pred_proba)

    ## Check for threshold error
    if threshold is not None and not (0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1 (inclusive).")

    if len(y_true_arr) != len(y_pred_proba_arr):
        raise ValueError("y_true and y_pred_proba must have the same length.")

    if len(y_true_arr) > 0:
        if not np.all((y_pred_proba_arr >= 0) & (y_pred_proba_arr <= 1)):
            raise ValueError("All y_pred_proba values must be between 0 and 1 (inclusive).")

        unique_labels_in_true = np.unique(y_true_arr)
        if len(unique_labels_in_true) > 2:
            raise ValueError(
                f"y_true should contain at most two unique labels for these binary metrics. Found: {unique_labels_in_true}"
            )
        ## Check if pos_label is actually in y_true
        if pos_label not in unique_labels_in_true and len(unique_labels_in_true) > 0 :
            warnings.warn(f"pos_label '{pos_label}' not found in y_true. All instances will be treated as negative.")

    return y_true_arr, y_pred_proba_arr

def _validate_value_inputs(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    transaction_values: Union[list, np.ndarray],
    k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Common validation function for value-based metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        transaction_values: Monetary values per transaction
        k: Number of top instances to consider
    
    Returns:
        Tuple of (y_true_array, y_pred_proba_array, transaction_values_array, validated_k)
    
    Raises:
        TypeError: If inputs have wrong types
        ValueError: If inputs have mismatched lengths or k is invalid
    """
    ## Input type validation
    if not isinstance(y_true, (list, np.ndarray)):
        raise TypeError("y_true must be a list or NumPy array.")
    if not isinstance(y_pred_proba, (list, np.ndarray)):
        raise TypeError("y_pred_scores must be a list or NumPy array.")
    if not isinstance(transaction_values, (list, np.ndarray)):
        raise TypeError("transaction_values must be a list or NumPy array.")
    if not isinstance(k, int):
        raise TypeError(f"k must be an integer, not {type(k)}")

    ## Convert to numpy arrays
    y_true_arr = np.array(y_true)
    y_scores_arr = np.array(y_pred_proba)
    trans_vals_arr = np.array(transaction_values)

    ## Length validation
    if not (len(y_true_arr) == len(y_scores_arr) == len(trans_vals_arr)):
        raise ValueError("All inputs (y_true, y_pred_scores, transaction_values) must have the same length.")

    ## k validation
    if k < 0:
        raise ValueError("k cannot be negative.")
    
    ## Cap k at the total number of samples
    validated_k = min(k, len(y_true_arr))
    
    return y_true_arr, y_scores_arr, trans_vals_arr, validated_k

def _prepare_ranked_data(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    k: int,
    pos_label: Any = 1
) -> Tuple[np.ndarray, int]:
    """
    Common preprocessing function for ranking-based metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        k: Number of top instances to consider
        pos_label: Label of positive class
    
    Returns:
        Tuple of (sorted_binary_labels, validated_k)
    
    Raises:
        ValueError: If k is negative or inputs are invalid
    """
    ## Validate inputs
    y_true_arr, y_pred_proba_arr = validate_binary_inputs(y_true, y_pred_proba, pos_label)

    ## Handle empty case
    if len(y_true) == 0:
        raise ValueError("Empty inputs provided")
    
    ## Handle edge cases for k
    if k < 0:
        raise ValueError("k cannot be negative.")
    if k == 0:
        raise ValueError("k cannot be zero.")
    
    k = min(k, len(y_true_arr))
    
    ## Convert y_true to binary 
    y_true_binary = (y_true_arr == pos_label).astype(int)
    
    ## Sort scores and corresponding true labels
    desc_score_indices = np.argsort(y_pred_proba_arr, kind="mergesort")[::-1]
    y_true_binary_sorted = y_true_binary[desc_score_indices]
    
    return y_true_binary_sorted, k