## The metrics in this file are based on a given threshold
import numpy as np

def get_binary_confusion_matrix(y_true: list, y_pred_proba: list, threshold: float=0.5, pos_label: int=1):
    """
    get_binary_confusion_matrix(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the components of a binary confusion matrix. 
    This function calculates True Positives (TP), False Positives (FP),
    False Negatives (FN), and True Negatives (TN) based on ground truth
    labels and predicted probabilities, using a threshold.

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (float, default=1): The label of the positive class in `y_true`.

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
        raise TypeError(f"y_true must be a python list and not {type(y_true)}")
    if not isinstance(y_pred_proba, (list, np.ndarray)):
        raise TypeError(f"y_pred_proba must be a python list and not {type(y_pred_proba)}")
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be a numeric value and not {type(threshold)}")
    if not isinstance(pos_label, (int)):
        raise TypeError(f"pos_label must be a binary integer(0,1) and not {type(pos_label)}")
    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1")
    
    # Convert lists to numpy array
    y_true_arr = np.array(y_true)
    y_pred_proba_arr = np.array(y_pred_proba)

    if len(y_true_arr) != len(y_pred_proba_arr):
        raise ValueError("y_true and y_pred_proba must have the same length.")
    if not np.all((y_pred_proba_arr >= 0) & (y_pred_proba_arr <= 1)):
        raise ValueError("y_pred_proba values must be between 0 and 1.")
    
    ## Check if y_true consists only pos_label and neg_label
    unique_labels = np.unique(y_true_arr)
    if len(unique_labels) > 2:
        raise ValueError("y_true can only have two unique labels.")
    
    ## Find Negative labels
    neg_label = None
    if len(unique_labels) == 2:
        neg_label = unique_labels[0] if unique_labels[1] == pos_label else unique_labels[1]
    elif len(unique_labels) == 1 and unique_labels[0] == pos_label:
        pass
    elif len(unique_labels) == 1 and unique_labels[0] != pos_label:
        neg_label = unique_labels[0]
    
    ## Compute Predicted labels using threshold
    y_pred_labels = np.where(y_pred_proba_arr >= threshold, pos_label, neg_label if neg_label is not None else 0)

    ## Raise Value Error if neg_label cannot be inferred safely
    if neg_label is None:
        if pos_label == 1: neg_label = 0
        elif pos_label == 0: neg_label = 1
        else:
            raise ValueError("Can't safely infer Negative label since pos_label is neither 0 nor 1")
    
    ## Compute confusion matrix components
    tp = np.sum((y_true_arr == pos_label) & (y_pred_labels == pos_label))
    fp = np.sum((y_true_arr == neg_label) & (y_pred_labels == pos_label))
    fn = np.sum((y_true_arr == pos_label) & (y_pred_labels == neg_label))
    tn = np.sum((y_true_arr == neg_label) & (y_pred_labels == neg_label))

    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

def get_accuracy_score(y_true: list, y_pred_proba: list, threshold: list=0.5, pos_label: int=1):
    """
    accuracy_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the accuracy score. 
    This function calculates accuracy score (TP + TN) / (TP + FP + FN + TN)

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (float, default=1): The label of the positive class in `y_true`.

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
    accuracy_score = float((tp+tn)/(tp+fp+fn+tn))

    return accuracy_score

def get_classification_error_score(y_true: list, y_pred_proba: list, threshold: list=0.5, pos_label: int=1):
    """
    get_classification_error_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the missclassification error score. 
    This function calculates missclassification error score (FP + FN) / (TP + FP + FN + TN)

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (float, default=1): The label of the positive class in `y_true`.

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
    classification_error_score = float((fp+fn)/(tp+fp+fn+tn))

    return classification_error_score

## Column Wise Metrics

def get_recall_score(y_true: list, y_pred_proba: list, threshold: list=0.5, pos_label: int=1):
    """
    get_recall_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the True Positive Rate. 
    This function calculates recall score (TP/(TP + FN))

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (float, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (Recall)
    """  

    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)
    
    ## Compute classification error score and return it
    tp = conf_dict["tp"]
    fp = conf_dict["fp"]
    fn = conf_dict["fn"]
    tn = conf_dict["tn"]
    recall_score = float((tp)/(tp+fn))

    return recall_score

def get_tnr_score(y_true: list, y_pred_proba: list, threshold: list=0.5, pos_label: int=1):
    """
    get_tnr_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the True Negative Rate. 
    This function calculates TNR score (TN/(TN + FP))

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (float, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (True Negative Rate)
    """  

    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)
    
    ## Compute classification error score and return it
    tp = conf_dict["tp"]
    fp = conf_dict["fp"]
    fn = conf_dict["fn"]
    tn = conf_dict["tn"]
    tnr_score = float((tn)/(tn+fp))

    return tnr_score

def get_fnr_score(y_true: list, y_pred_proba: list, threshold: list=0.5, pos_label: int=1):
    """
    get_fnr_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the False Negative Rate. 
    This function calculates FNR score (FN/(TP + FN))

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (float, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (False Negative Rate)
    """  

    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)
    
    ## Compute classification error score and return it
    tp = conf_dict["tp"]
    fp = conf_dict["fp"]
    fn = conf_dict["fn"]
    tn = conf_dict["tn"]
    fnr_score = float((fn)/(tp+fn))

    return fnr_score

def get_fpr_score(y_true: list, y_pred_proba: list, threshold: list=0.5, pos_label: int=1):
    """
    get_fpr_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the False Positive Rate. 
    This function calculates FPR score (FP/(TN + FP))

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (float, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (False Positive Rate)
    """  

    ## Get Binary Confusion Matrix components
    conf_dict = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)
    
    ## Compute classification error score and return it
    tp = conf_dict["tp"]
    fp = conf_dict["fp"]
    fn = conf_dict["fn"]
    tn = conf_dict["tn"]
    fpr_score = float((fp)/(tn+fp))

    return fpr_score

## Compute BER and G-mean

def get_ber_score(y_true: list, y_pred_proba: list, threshold: list=0.5, pos_label: int=1):
    """
    get_ber_score(y_true, y_pred_prob, threshold=0.5, pos_label=1)

    Computes the Balanced Error Rate.
    Taking the mean of the FNR and the TNR provides a balanced measure of 
    accuracy known as the Balanced Error Rate (BER).
    BER = 0.5*(FNR+FPR)

    Args:
        y_true(list or np.ndarray): True binary labels. Expected values are {0, 1}.
        y_pred_proba(list or np.ndarray): Predicted probabilities for the positive class. Values should be in the range [0, 1]. Must be the same length as `y_true`.
        threshold (float, default=0.5): The decision threshold to convert probabilities into binary predictions.
        pos_label (float, default=1): The label of the positive class in `y_true`.

    Returns:
        Float (Balanced Error Rate)
    """  

    ## Get FNR and FPR
    fnr = get_fnr_score(y_true, y_pred_proba, threshold, pos_label)
    fpr = get_fpr_score(y_true, y_pred_proba, threshold, pos_label)

    ## Compute BER from FNR and FPR
    ber_score = 0.5*(fnr+fpr)
    ber_score = float(ber_score)

    return ber_score

