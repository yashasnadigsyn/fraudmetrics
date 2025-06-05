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