## The metrics in this file are based on a given threshold
import numpy as np

def get_binary_confusion_matrix(y_true: list, y_pred_proba: list, threshold: float=0.5, pos_label: int=1):
    """
    Computes the components of a binary confusion matrix.
    This function calculates True Positives (TP), False Positives (FP),
    False Negatives (FN), and True Negatives (TN) based on ground truth
    labels and predicted probabilities, using a threshold.

    Args:
        y_true (list or np.ndarray): True binary labels.
        y_pred_proba (list or np.ndarray): Predicted probabilities for the positive class.
                                         Values should be in the range [0, 1].
                                         Must be the same length as `y_true`.
        threshold (float, optional): The decision threshold to convert probabilities
                                     into binary predictions. Defaults to 0.5.
        pos_label (int, optional): The label of the positive class in `y_true`.
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
    ## Check for Input Validation
    if not isinstance(y_true, (list, np.ndarray)):
        raise TypeError(f"y_true must be a list or NumPy array, not {type(y_true)}")
    if not isinstance(y_pred_proba, (list, np.ndarray)):
        raise TypeError(f"y_pred_proba must be a list or NumPy array, not {type(y_pred_proba)}")
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be a numeric value, not {type(threshold)}")
    if not isinstance(pos_label, (int)): 
        raise TypeError(f"pos_label must be an integer, not {type(pos_label)}")
    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1 (inclusive).")
    
    ## Convert list to numpy array
    y_true_arr = np.array(y_true)
    y_pred_proba_arr = np.array(y_pred_proba)

    ## Check if both y_true and y_pred_proba of same length
    if len(y_true_arr) != len(y_pred_proba_arr):
        raise ValueError("y_true and y_pred_proba must have the same length.")
    
    if len(y_true_arr) == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    if not np.all((y_pred_proba_arr >= 0) & (y_pred_proba_arr <= 1)):
        raise ValueError("All y_pred_proba values must be between 0 and 1 (inclusive).")

    unique_labels_in_true = np.unique(y_true_arr)
    if len(unique_labels_in_true) > 2:
        raise ValueError(f"y_true should contain at most two unique labels. Found: {unique_labels_in_true}")

    # Convert y_true to binary (1 for pos_label, 0 for others)
    y_true_binary = (y_true_arr == pos_label).astype(int)

    # Predict labels based on threshold (1 for positive, 0 for negative)
    y_pred_binary = (y_pred_proba_arr >= threshold).astype(int)

    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1)) 
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))

    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

def get_accuracy_score(y_true: list, y_pred_proba: list, threshold: float=0.5, pos_label: int=1):
    """
    accuracy_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the accuracy score. 
    This function calculates accuracy score (TP + TN) / (TP + FP + FN + TN)

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (int, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (Accuracy)
    """  

    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)
    
    ## Compute accuracy score and return it
    tp = conf_dict["tp"]
    fp = conf_dict["fp"]
    fn = conf_dict["fn"]
    tn = conf_dict["tn"]
    N = tp+fp+fn+tn
    if N == 0:
        return np.nan
    else:
        accuracy_score = float((tp+tn)/(N))

    return accuracy_score

def get_classification_error_score(y_true: list, y_pred_proba: list, threshold: float=0.5, pos_label: int=1):
    """
    get_classification_error_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the missclassification error score. 
    This function calculates missclassification error score (FP + FN) / (TP + FP + FN + TN)

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (int, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (Error Rate)
    """  

    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)
    
    ## Compute classification error score and return it
    tp = conf_dict["tp"]
    fp = conf_dict["fp"]
    fn = conf_dict["fn"]
    tn = conf_dict["tn"]
    N = tp+fp+fn+tn
    if N == 0:
        return np.nan
    else:
        classification_error_score = float((fp+fn)/(N))

    return classification_error_score

## Column Wise Metrics

def get_recall_score(y_true: list, y_pred_proba: list, threshold: float=0.5, pos_label: int=1):
    """
    get_recall_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the True Positive Rate. 
    This function calculates recall score (TP/(TP + FN))

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (int, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (Recall)
    """  

    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)
    
    ## Compute recall score and return it
    tp = conf_dict["tp"]
    fn = conf_dict["fn"]
    den = tp+fn
    if den == 0:
        return np.nan
    else:
        recall_score = float((tp)/(den))

    return recall_score

def get_specificity_score(y_true: list, y_pred_proba: list, threshold: float=0.5, pos_label: int=1):
    """
    get_specificity_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the True Negative Rate (or, Specificity). 
    This function calculates TNR score (TN/(TN + FP))

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (int, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (True Negative Rate)
    """  

    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)
    
    ## Compute TNR score and return it
    fp = conf_dict["fp"]
    tn = conf_dict["tn"]
    den = fp+tn
    if den == 0:
        return np.nan
    else:
        tnr_score = float((tn)/(den))

    return tnr_score

def get_fnr_score(y_true: list, y_pred_proba: list, threshold: float=0.5, pos_label: int=1):
    """
    get_fnr_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the False Negative Rate. 
    This function calculates FNR score (FN/(TP + FN))

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (int, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (False Negative Rate)
    """  

    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)
    
    ## Compute FNR score and return it
    tp = conf_dict["tp"]
    fn = conf_dict["fn"]
    den = tp+fn
    if den == 0:
        return np.nan
    else:
        fnr_score = float((fn)/(den))

    return fnr_score

def get_fpr_score(y_true: list, y_pred_proba: list, threshold: float=0.5, pos_label: int=1):
    """
    get_fpr_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the False Positive Rate. 
    This function calculates FPR score (FP/(TN + FP))

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (int, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (False Positive Rate)
    """  

    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)
    
    ## Compute FPR score and return it
    fp = conf_dict["fp"]
    tn = conf_dict["tn"]
    den = fp+tn
    if den == 0:
        return np.nan
    else:
        fpr_score = float((fp)/(den))

    return fpr_score

## Compute BER and G-mean

def get_ber_score(y_true: list, y_pred_proba: list, threshold: float=0.5, pos_label: int=1):
    """
    get_ber_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the Balanced Error Rate.
    BER = 0.5*(FNR+FPR)

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (int, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (Balanced Error Rate)
    """  

    ## Get FNR and FPR
    fnr = get_fnr_score(y_true, y_pred_proba, threshold, pos_label)
    fpr = get_fpr_score(y_true, y_pred_proba, threshold, pos_label)

    ## Check for NaN values
    if np.isnan(fnr) or np.isnan(fpr):
        return np.nan

    ## Compute BER from FNR and FPR
    ber_score = 0.5*(fnr+fpr)
    ber_score = float(ber_score)

    return ber_score

def get_gmean_score(y_true: list, y_pred_proba: list, threshold: float=0.5, pos_label: int=1):
    """
    get_gmean_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the Geometric Mean Score.
    A measure that aggregates the TNR and TPR is the geometric mean G-mean.
    G-mean = sqrt(TPR*TNR)

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (int, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (Geometric Mean Score)
    """  

    ## Get TPR and TNR
    tpr = get_recall_score(y_true, y_pred_proba, threshold, pos_label)
    tnr = get_specificity_score(y_true, y_pred_proba, threshold, pos_label)

    ## Check for NaN values
    if np.isnan(tpr) or np.isnan(tnr):
        return np.nan

    ## Compute G-mean from TPR and TNR
    gmean_score = np.sqrt(tpr*tnr)
    gmean_score = float(gmean_score)

    return gmean_score

## Row Wise Metrics

def get_precision_score(y_true: list, y_pred_proba: list, threshold: float=0.5, pos_label: int=1):
    """
    get_precision_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the Precision Score (or, Positive Predicted Value).
    This measures the the set of transactions that are predicted as fraudulent by the proportion of transactions that are indeed fraudulent.
    Precision = (TP)/(TP+FP)

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (int, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (Precision Score)
    """  

    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)
    
    ## Compute precision score and return it
    tp = conf_dict["tp"]
    fp = conf_dict["fp"]
    den = tp+fp
    if den == 0:
        return np.nan
    else:
        precision_score = float((tp)/(den))

    return precision_score

def get_npv_score(y_true: list, y_pred_proba: list, threshold: float=0.5, pos_label: int=1):
    """
    get_npv_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the Negative Prediction Value (NPV).
    NPV = (TN)/(TN+FN)

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (int, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (NPV)
    """  

    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)
    
    ## Compute NPV and return it
    tn = conf_dict["tn"]
    fn = conf_dict["fn"]
    den = tn+fn
    if den == 0:
        return np.nan
    else:
        npv_score = float((tn)/(den))

    return npv_score

def get_false_discovery_rate(y_true: list, y_pred_proba: list, threshold: float=0.5, pos_label: int=1):
    """
    get_false_discovery_rate(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the False Discovery Rate.
    FDR = (FP)/(TP+FP) = 1-precision

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (int, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (FDR)
    """  

    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)
    
    ## Compute FDR and return it
    tp = conf_dict["tp"]
    fp = conf_dict["fp"]
    den = tp+fp
    if den == 0:
        return np.nan
    else:
        fdr = float((fp)/(den))

    return fdr

def get_false_omission_rate(y_true: list, y_pred_proba: list, threshold: float=0.5, pos_label: int=1):
    """
    get_false_omission_rate(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the False Omission Rate.
    FOR = (FN)/(TN+FN) = 1-NPV

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (int, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (FOR)
    """  

    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)
    
    ## Compute FOR and return it
    fn = conf_dict["fn"]
    tn = conf_dict["tn"]
    den = tn+fn
    if den == 0:
        return np.nan
    else:
        FOR = float((fn)/(den))

    return FOR

def get_f1_score(y_true: list, y_pred_proba: list, threshold: float=0.5, pos_label: int=1):
    """
    get_f1_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the F1 Score.
    Harmonic Mean of Precision and Recall.
    F1-Score = 2*(precision*recall)/(precision+recall)

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (int, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (F1-Score)
    """  

    ## Get Precision and Recall Scores
    precision_score = get_precision_score(y_true, y_pred_proba, threshold, pos_label)
    recall_score = get_recall_score(y_true, y_pred_proba, threshold, pos_label)
    
    ## Check for NaN values
    if np.isnan(precision_score) or np.isnan(recall_score):
        return np.nan
    
    ## Check for zero denominator
    den = precision_score + recall_score
    if den == 0:
        return np.nan
    
    ## Compute F1-Score and return it
    f1_score = 2*(precision_score*recall_score)/(den)

    return f1_score