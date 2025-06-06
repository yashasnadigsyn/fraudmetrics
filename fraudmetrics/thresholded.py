## The metrics in this file are based on a given threshold

import numpy as np
from typing import Union, Dict, Any
from .utils.validate_inputs import validate_binary_inputs

def get_binary_confusion_matrix(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    threshold: float = 0.5,
    pos_label: Any = 1
) -> Dict[str, int]:
    """
    Computes the components of a binary confusion matrix.
    This function calculates True Positives (TP), False Positives (FP),
    False Negatives (FN), and True Negatives (TN) based on ground truth
    labels and predicted probabilities, using a threshold.

    Args:
        y_true (Union[list, np.ndarray]): True labels. Values will be
                                         binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
                                         Values should be in the range [0, 1].
                                         Must be the same length as `y_true`.
        threshold (float, optional): The decision threshold to convert probabilities
                                     into binary predictions. Defaults to 0.5.
        pos_label (Any, optional): The label of the positive class in `y_true`.
                                   Defaults to 1.

    Returns:
        dict: A dictionary containing the confusion matrix components:
            {
                "tp": int,  # True Positives
                "fp": int,  # False Positives
                "fn": int,  # False Negatives
                "tn": int   # True Negatives
            }
    """
    ## Validate inputs
    y_true_arr, y_pred_proba_arr = validate_binary_inputs(y_true, y_pred_proba, pos_label, threshold)

    ## Handle empty case after validation (length check is in validator)
    if len(y_true_arr) == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    ## Convert y_true to binary (1 for pos_label, 0 for others)
    y_true_binary = (y_true_arr == pos_label).astype(int)

    ## Predict labels based on threshold (1 for positive, 0 for negative)
    y_pred_binary = (y_pred_proba_arr >= threshold).astype(int)

    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))

    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

def get_accuracy_score(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    threshold: float = 0.5,
    pos_label: Any = 1
) -> float:
    """
    get_accuracy_score(y_true, y_pred_proba, threshold=0.5, pos_label=1)

    Computes the accuracy score.
    This function calculates accuracy score (TP + TN) / (TP + FP + FN + TN)

    Args:
        y_true (Union[list, np.ndarray]): True labels. Values will be
                                         binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
                                             Values should be in the range [0, 1].
                                             Must be the same length as `y_true`.
        threshold (float, optional): The decision threshold. Defaults to 0.5.
        pos_label (Any, optional): The label of the positive class. Defaults to 1.

    Returns:
        float: Accuracy score. Returns np.nan if the total number of instances is 0.
    """
    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)

    ## Compute accuracy score and return it
    tp = conf_dict["tp"]
    fp = conf_dict["fp"]
    fn = conf_dict["fn"]
    tn = conf_dict["tn"]
    N = tp + fp + fn + tn
    if N == 0:
        return np.nan
    else:
        accuracy = float((tp + tn) / N)
    return accuracy

def get_classification_error_score(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    threshold: float = 0.5,
    pos_label: Any = 1
) -> float:
    """
    get_classification_error_score(y_true, y_pred_proba, threshold=0.5, pos_label=1)

    Computes the misclassification error score.
    This function calculates misclassification error score (FP + FN) / (TP + FP + FN + TN)

    Args:
        y_true (Union[list, np.ndarray]): True labels. Binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
        threshold (float, optional): Decision threshold. Defaults to 0.5.
        pos_label (Any, optional): Label of the positive class. Defaults to 1.

    Returns:
        float: Error Rate. Returns np.nan if N=0.
    """
    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)

    ## Compute classification error score and return it
    tp = conf_dict["tp"]
    fp = conf_dict["fp"]
    fn = conf_dict["fn"]
    tn = conf_dict["tn"]
    N = tp + fp + fn + tn
    if N == 0:
        return np.nan
    else:
        classification_error = float((fp + fn) / N)
    return classification_error

## Column Wise Metrics
def get_recall_score(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    threshold: float = 0.5,
    pos_label: Any = 1
) -> float:
    """
    get_recall_score(y_true, y_pred_proba, threshold=0.5, pos_label=1)

    Computes the True Positive Rate (Recall or Sensitivity).
    This function calculates recall score (TP / (TP + FN))

    Args:
        y_true (Union[list, np.ndarray]): True labels. Binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
        threshold (float, optional): Decision threshold. Defaults to 0.5.
        pos_label (Any, optional): Label of the positive class. Defaults to 1.

    Returns:
        float: Recall score. Returns np.nan if (TP + FN) is 0.
    """
    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)

    ## Compute recall score and return it
    tp = conf_dict["tp"]
    fn = conf_dict["fn"]
    den = tp + fn
    if den == 0:
        return np.nan
    else:
        recall = float(tp / den)
    return recall

def get_specificity_score(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    threshold: float = 0.5,
    pos_label: Any = 1
) -> float:
    """
    get_specificity_score(y_true, y_pred_proba, threshold=0.5, pos_label=1)

    Computes the True Negative Rate (Specificity).
    This function calculates TNR score (TN / (TN + FP))

    Args:
        y_true (Union[list, np.ndarray]): True labels. Binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
        threshold (float, optional): Decision threshold. Defaults to 0.5.
        pos_label (Any, optional): Label of the positive class. Defaults to 1.

    Returns:
        float: True Negative Rate. Returns np.nan if (TN + FP) is 0.
    """
    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)

    ## Compute TNR score and return it
    fp = conf_dict["fp"]
    tn = conf_dict["tn"]
    den = tn + fp # Corrected denominator
    if den == 0:
        return np.nan
    else:
        specificity = float(tn / den)
    return specificity

def get_fnr_score(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    threshold: float = 0.5,
    pos_label: Any = 1
) -> float:
    """
    get_fnr_score(y_true, y_pred_proba, threshold=0.5, pos_label=1)

    Computes the False Negative Rate (Miss Rate).
    This function calculates FNR score (FN / (TP + FN))

    Args:
        y_true (Union[list, np.ndarray]): True labels. Binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
        threshold (float, optional): Decision threshold. Defaults to 0.5.
        pos_label (Any, optional): Label of the positive class. Defaults to 1.

    Returns:
        float: False Negative Rate. Returns np.nan if (TP + FN) is 0.
    """
    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)

    ## Compute FNR score and return it
    tp = conf_dict["tp"]
    fn = conf_dict["fn"]
    den = tp + fn
    if den == 0:
        return np.nan
    else:
        fnr = float(fn / den)
    return fnr

def get_fpr_score(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    threshold: float = 0.5,
    pos_label: Any = 1
) -> float:
    """
    get_fpr_score(y_true, y_pred_proba, threshold=0.5, pos_label=1)

    Computes the False Positive Rate (Fall-out).
    This function calculates FPR score (FP / (TN + FP))

    Args:
        y_true (Union[list, np.ndarray]): True labels. Binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
        threshold (float, optional): Decision threshold. Defaults to 0.5.
        pos_label (Any, optional): Label of the positive class. Defaults to 1.

    Returns:
        float: False Positive Rate. Returns np.nan if (TN + FP) is 0.
    """
    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)

    ## Compute FPR score and return it
    fp = conf_dict["fp"]
    tn = conf_dict["tn"]
    den = tn + fp # Corrected denominator
    if den == 0:
        return np.nan
    else:
        fpr = float(fp / den)
    return fpr

## Compute BER and G-mean
def get_ber_score(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    threshold: float = 0.5,
    pos_label: Any = 1
) -> float:
    """
    get_ber_score(y_true, y_pred_proba, threshold=0.5, pos_label=1)

    Computes the Balanced Error Rate.
    BER = 0.5 * (FNR + FPR)

    Args:
        y_true (Union[list, np.ndarray]): True labels. Binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
        threshold (float, optional): Decision threshold. Defaults to 0.5.
        pos_label (Any, optional): Label of the positive class. Defaults to 1.

    Returns:
        float: Balanced Error Rate. Returns np.nan if FNR or FPR is nan.
    """
    ## Get FNR and FPR
    fnr = get_fnr_score(y_true, y_pred_proba, threshold, pos_label)
    fpr = get_fpr_score(y_true, y_pred_proba, threshold, pos_label)

    ## Check for NaN values
    if np.isnan(fnr) or np.isnan(fpr):
        return np.nan

    ## Compute BER from FNR and FPR
    ber = 0.5 * (fnr + fpr)
    return float(ber)

def get_gmean_score(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    threshold: float = 0.5,
    pos_label: Any = 1
) -> float:
    """
    get_gmean_score(y_true, y_pred_proba, threshold=0.5, pos_label=1)

    Computes the Geometric Mean Score.
    G-mean = sqrt(TPR * TNR)

    Args:
        y_true (Union[list, np.ndarray]): True labels. Binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
        threshold (float, optional): Decision threshold. Defaults to 0.5.
        pos_label (Any, optional): Label of the positive class. Defaults to 1.

    Returns:
        float: Geometric Mean Score. Returns np.nan if TPR or TNR is nan.
    """
    ## Get TPR (Recall) and TNR (Specificity)
    tpr = get_recall_score(y_true, y_pred_proba, threshold, pos_label)
    tnr = get_specificity_score(y_true, y_pred_proba, threshold, pos_label)

    ## Check for NaN values
    if np.isnan(tpr) or np.isnan(tnr):
        return np.nan
    if tpr < 0 or tnr < 0: # Should not happen with correct TP/TN/FP/FN
        return np.nan


    ## Compute G-mean from TPR and TNR
    gmean = np.sqrt(tpr * tnr)
    return float(gmean)

## Row Wise Metrics
def get_precision_score(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    threshold: float = 0.5,
    pos_label: Any = 1
) -> float:
    """
    get_precision_score(y_true, y_pred_proba, threshold=0.5, pos_label=1)

    Computes the Precision Score (Positive Predictive Value).
    Precision = TP / (TP + FP)

    Args:
        y_true (Union[list, np.ndarray]): True labels. Binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
        threshold (float, optional): Decision threshold. Defaults to 0.5.
        pos_label (Any, optional): Label of the positive class. Defaults to 1.

    Returns:
        float: Precision Score. Returns np.nan if (TP + FP) is 0.
    """
    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)

    ## Compute precision score and return it
    tp = conf_dict["tp"]
    fp = conf_dict["fp"]
    den = tp + fp
    if den == 0:
        return np.nan
    else:
        precision = float(tp / den)
    return precision

def get_npv_score(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    threshold: float = 0.5,
    pos_label: Any = 1
) -> float:
    """
    get_npv_score(y_true, y_pred_proba, threshold=0.5, pos_label=1)

    Computes the Negative Predictive Value (NPV).
    NPV = TN / (TN + FN)

    Args:
        y_true (Union[list, np.ndarray]): True labels. Binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
        threshold (float, optional): Decision threshold. Defaults to 0.5.
        pos_label (Any, optional): Label of the positive class. Defaults to 1.

    Returns:
        float: NPV. Returns np.nan if (TN + FN) is 0.
    """
    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)

    ## Compute NPV and return it
    tn = conf_dict["tn"]
    fn = conf_dict["fn"]
    den = tn + fn
    if den == 0:
        return np.nan
    else:
        npv = float(tn / den)
    return npv

def get_false_discovery_rate(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    threshold: float = 0.5,
    pos_label: Any = 1
) -> float:
    """
    get_false_discovery_rate(y_true, y_pred_proba, threshold=0.5, pos_label=1)

    Computes the False Discovery Rate.
    FDR = FP / (TP + FP) = 1 - Precision

    Args:
        y_true (Union[list, np.ndarray]): True labels. Binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
        threshold (float, optional): Decision threshold. Defaults to 0.5.
        pos_label (Any, optional): Label of the positive class. Defaults to 1.

    Returns:
        float: FDR. Returns np.nan if (TP + FP) is 0.
    """
    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)

    ## Compute FDR and return it
    tp = conf_dict["tp"]
    fp = conf_dict["fp"]
    den = tp + fp
    if den == 0:
        return np.nan
    else:
        fdr = float(fp / den)
    return fdr

def get_false_omission_rate(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    threshold: float = 0.5,
    pos_label: Any = 1
) -> float:
    """
    get_false_omission_rate(y_true, y_pred_proba, threshold=0.5, pos_label=1)

    Computes the False Omission Rate.
    FOR = FN / (TN + FN) = 1 - NPV

    Args:
        y_true (Union[list, np.ndarray]): True labels. Binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
        threshold (float, optional): Decision threshold. Defaults to 0.5.
        pos_label (Any, optional): Label of the positive class. Defaults to 1.

    Returns:
        float: FOR. Returns np.nan if (TN + FN) is 0.
    """
    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)

    ## Compute FOR and return it
    fn = conf_dict["fn"]
    tn = conf_dict["tn"]
    den = tn + fn
    if den == 0:
        return np.nan
    else:
        FOR_val = float(fn / den) # Renamed variable to avoid conflict with 'for' keyword
    return FOR_val

def get_f1_score(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    threshold: float = 0.5,
    pos_label: Any = 1
) -> float:
    """
    get_f1_score(y_true, y_pred_proba, threshold=0.5, pos_label=1)

    Computes the F1 Score (Harmonic Mean of Precision and Recall).
    F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        y_true (Union[list, np.ndarray]): True labels. Binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
        threshold (float, optional): Decision threshold. Defaults to 0.5.
        pos_label (Any, optional): Label of the positive class. Defaults to 1.

    Returns:
        float: F1-Score. Returns np.nan if Precision or Recall is nan, or if their sum is 0.
    """
    ## Get Precision and Recall Scores
    precision = get_precision_score(y_true, y_pred_proba, threshold, pos_label)
    recall = get_recall_score(y_true, y_pred_proba, threshold, pos_label)

    ## Check for NaN values
    if np.isnan(precision) or np.isnan(recall):
        return np.nan

    ## Check for zero denominator
    den = precision + recall
    if den == 0:
        return np.nan # F1 is 0 if precision and recall are 0. Or np.nan if undefined.

    ## Compute F1-Score and return it
    f1 = 2 * (precision * recall) / den
    return float(f1)