# Warning: All examples here are generated using Ai for checking..

# examples.py
import numpy as np

# Assuming your metrics functions are in a file named 'thresholded_metrics.py'
# in the same directory, or you've structured it as a package.
# For a simple file structure:
from thresholded import (
    get_binary_confusion_matrix,
    get_accuracy_score,
    get_classification_error_score,
    get_recall_score,
    get_specificity_score,
    get_fnr_score,
    get_fpr_score,
    get_ber_score,
    get_gmean_score,
    get_precision_score,
    get_npv_score,
    get_false_discovery_rate,
    get_false_omission_rate,
    get_f1_score
)

def run_examples():
    print("--- Running Metric Examples ---")

    # --- Test Case 1: Standard Mix ---
    print("\n--- Test Case 1: Standard Mix ---")
    y_true_1 = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred_proba_1 = np.array([0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4, 0.1, 0.75, 0.25])
    threshold_1 = 0.5
    pos_label_1 = 1
    # Expected @ 0.5: y_pred_binary = [1,0,1,1,0,1,0,0,1,0]
    # y_true_binary: [1,0,1,1,0,1,0,0,1,0] (assuming pos_label=1)
    # TP=5 (all 1s match)
    # FP=0 (no 0 in true predicted as 1)
    # FN=0 (no 1 in true predicted as 0)
    # TN=5 (all 0s match) -> This is a perfect prediction scenario based on y_pred_binary

    # Let's make it more interesting:
    y_true_1 =       np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1]) # 6 Pos, 4 Neg
    y_pred_proba_1 = np.array([0.9, 0.2, 0.3, 0.7, 0.6, 0.6, 0.4, 0.1, 0.75, 0.45])
    # y_pred_binary @ 0.5:    [1,   0,   0,   1,   1,   1,   0,   0,   1,    0]
    # y_true_binary (pos=1):  [1,   0,   1,   1,   0,   1,   0,   0,   1,    1]
    # TP: (1,1) (1,1) (1,1) (1,1) = 4 (indices 0,3,5,8)
    # FP: (0,1) = 1 (index 4)
    # FN: (1,0) (1,0) = 2 (indices 2,9)
    # TN: (0,0) (0,0) (0,0) = 3 (indices 1,6,7)
    # Sum = 4+1+2+3 = 10. Correct.

    print(f"y_true: {y_true_1}")
    print(f"y_pred_proba: {y_pred_proba_1}")
    print(f"threshold: {threshold_1}, pos_label: {pos_label_1}")

    cm1 = get_binary_confusion_matrix(y_true_1, y_pred_proba_1, threshold_1, pos_label_1)
    print(f"Confusion Matrix: {cm1}") # Expected: {'tp': 4, 'fp': 1, 'fn': 2, 'tn': 3}

    print(f"Accuracy: {get_accuracy_score(y_true_1, y_pred_proba_1, threshold_1, pos_label_1):.4f}") # (4+3)/10 = 0.7
    print(f"Classification Error: {get_classification_error_score(y_true_1, y_pred_proba_1, threshold_1, pos_label_1):.4f}") # (1+2)/10 = 0.3
    print(f"Recall (TPR): {get_recall_score(y_true_1, y_pred_proba_1, threshold_1, pos_label_1):.4f}") # 4/(4+2) = 4/6 = 0.6667
    print(f"Specificity (TNR): {get_specificity_score(y_true_1, y_pred_proba_1, threshold_1, pos_label_1):.4f}") # 3/(3+1) = 3/4 = 0.75
    print(f"FNR: {get_fnr_score(y_true_1, y_pred_proba_1, threshold_1, pos_label_1):.4f}") # 2/(4+2) = 2/6 = 0.3333
    print(f"FPR: {get_fpr_score(y_true_1, y_pred_proba_1, threshold_1, pos_label_1):.4f}") # 1/(3+1) = 1/4 = 0.25
    print(f"Precision: {get_precision_score(y_true_1, y_pred_proba_1, threshold_1, pos_label_1):.4f}") # 4/(4+1) = 4/5 = 0.8
    print(f"NPV: {get_npv_score(y_true_1, y_pred_proba_1, threshold_1, pos_label_1):.4f}") # 3/(3+2) = 3/5 = 0.6
    print(f"FDR: {get_false_discovery_rate(y_true_1, y_pred_proba_1, threshold_1, pos_label_1):.4f}") # 1/(4+1) = 1/5 = 0.2
    print(f"FOR: {get_false_omission_rate(y_true_1, y_pred_proba_1, threshold_1, pos_label_1):.4f}") # 2/(3+2) = 2/5 = 0.4
    print(f"F1 Score: {get_f1_score(y_true_1, y_pred_proba_1, threshold_1, pos_label_1):.4f}") # 2*(0.8*0.66666)/(0.8+0.66666) = 0.7273
    print(f"BER: {get_ber_score(y_true_1, y_pred_proba_1, threshold_1, pos_label_1):.4f}") # 0.5 * (0.33333 + 0.25) = 0.2917
    print(f"G-Mean: {get_gmean_score(y_true_1, y_pred_proba_1, threshold_1, pos_label_1):.4f}") # sqrt(0.66666 * 0.75) = 0.7071


    # --- Test Case 2: All Positives Predicted Correctly (Perfect Recall) ---
    print("\n--- Test Case 2: All Positives Predicted Correctly ---")
    y_true_2 = np.array([1, 1, 0, 0])
    y_pred_proba_2 = np.array([0.9, 0.8, 0.2, 0.1]) # Preds @ 0.5: [1,1,0,0]
    threshold_2 = 0.5
    # TP=2, FP=0, FN=0, TN=2
    cm2 = get_binary_confusion_matrix(y_true_2, y_pred_proba_2, threshold_2)
    print(f"Confusion Matrix: {cm2}") # Expected: {'tp': 2, 'fp': 0, 'fn': 0, 'tn': 2}
    print(f"Recall: {get_recall_score(y_true_2, y_pred_proba_2, threshold_2):.4f}") # 2/(2+0) = 1.0
    print(f"Precision: {get_precision_score(y_true_2, y_pred_proba_2, threshold_2):.4f}") # 2/(2+0) = 1.0
    print(f"F1 Score: {get_f1_score(y_true_2, y_pred_proba_2, threshold_2):.4f}") # 1.0

    # --- Test Case 3: No Positives Predicted (Zero Recall, Zero Precision if no FP) ---
    print("\n--- Test Case 3: No Positives Predicted ---")
    y_true_3 = np.array([1, 1, 0, 0])
    y_pred_proba_3 = np.array([0.2, 0.1, 0.3, 0.05]) # Preds @ 0.5: [0,0,0,0]
    threshold_3 = 0.5
    # TP=0, FP=0, FN=2, TN=2
    cm3 = get_binary_confusion_matrix(y_true_3, y_pred_proba_3, threshold_3)
    print(f"Confusion Matrix: {cm3}") # Expected: {'tp': 0, 'fp': 0, 'fn': 2, 'tn': 2}
    print(f"Recall: {get_recall_score(y_true_3, y_pred_proba_3, threshold_3):.4f}") # 0/(0+2) = 0.0
    print(f"Precision: {get_precision_score(y_true_3, y_pred_proba_3, threshold_3):.4f}") # 0/(0+0) = nan
    print(f"F1 Score: {get_f1_score(y_true_3, y_pred_proba_3, threshold_3):.4f}") # nan

    # --- Test Case 4: All Predicted as Positive ---
    print("\n--- Test Case 4: All Predicted as Positive ---")
    y_true_4 = np.array([1, 0, 1, 0])
    y_pred_proba_4 = np.array([0.9, 0.8, 0.7, 0.6]) # Preds @ 0.5: [1,1,1,1]
    threshold_4 = 0.5
    # TP=2, FP=2, FN=0, TN=0
    cm4 = get_binary_confusion_matrix(y_true_4, y_pred_proba_4, threshold_4)
    print(f"Confusion Matrix: {cm4}") # Expected: {'tp': 2, 'fp': 2, 'fn': 0, 'tn': 0}
    print(f"Recall: {get_recall_score(y_true_4, y_pred_proba_4, threshold_4):.4f}") # 2/(2+0) = 1.0
    print(f"Precision: {get_precision_score(y_true_4, y_pred_proba_4, threshold_4):.4f}") # 2/(2+2) = 0.5
    print(f"Specificity: {get_specificity_score(y_true_4, y_pred_proba_4, threshold_4):.4f}") # 0/(0+2) = 0.0
    print(f"NPV: {get_npv_score(y_true_4, y_pred_proba_4, threshold_4):.4f}") # 0/(0+0) = nan

    # --- Test Case 5: Only Negative Class Present in y_true ---
    print("\n--- Test Case 5: Only Negative Class Present ---")
    y_true_5 = np.array([0, 0, 0, 0])
    y_pred_proba_5 = np.array([0.9, 0.2, 0.8, 0.1]) # Preds @ 0.5 (pos_label=1): [1,0,1,0]
    threshold_5 = 0.5
    pos_label_5 = 1
    # y_true_binary (pos=1): [0,0,0,0]
    # TP=0, FP=2, FN=0, TN=2
    cm5 = get_binary_confusion_matrix(y_true_5, y_pred_proba_5, threshold_5, pos_label_5)
    print(f"Confusion Matrix (pos_label=1): {cm5}") # Expected: {'tp': 0, 'fp': 2, 'fn': 0, 'tn': 2}
    print(f"Recall: {get_recall_score(y_true_5, y_pred_proba_5, threshold_5, pos_label_5):.4f}") # 0/(0+0) = nan
    print(f"Precision: {get_precision_score(y_true_5, y_pred_proba_5, threshold_5, pos_label_5):.4f}") # 0/(0+2) = 0.0

    # --- Test Case 6: Only Positive Class Present in y_true ---
    print("\n--- Test Case 6: Only Positive Class Present ---")
    y_true_6 = np.array([1, 1, 1, 1])
    y_pred_proba_6 = np.array([0.9, 0.2, 0.8, 0.1]) # Preds @ 0.5: [1,0,1,0]
    threshold_6 = 0.5
    pos_label_6 = 1
    # y_true_binary (pos=1): [1,1,1,1]
    # TP=2, FP=0, FN=2, TN=0
    cm6 = get_binary_confusion_matrix(y_true_6, y_pred_proba_6, threshold_6, pos_label_6)
    print(f"Confusion Matrix (pos_label=1): {cm6}") # Expected: {'tp': 2, 'fp': 0, 'fn': 2, 'tn': 0}
    print(f"Specificity: {get_specificity_score(y_true_6, y_pred_proba_6, threshold_6, pos_label_6):.4f}") # 0/(0+0) = nan
    print(f"Precision: {get_precision_score(y_true_6, y_pred_proba_6, threshold_6, pos_label_6):.4f}") # 2/(2+0) = 1.0

    # --- Test Case 7: Empty Inputs ---
    print("\n--- Test Case 7: Empty Inputs ---")
    y_true_7 = np.array([])
    y_pred_proba_7 = np.array([])
    threshold_7 = 0.5
    cm7 = get_binary_confusion_matrix(y_true_7, y_pred_proba_7, threshold_7)
    print(f"Confusion Matrix: {cm7}") # Expected: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    # All metrics should then correctly return nan due to 0/0
    print(f"Accuracy: {get_accuracy_score(y_true_7, y_pred_proba_7, threshold_7):.4f}") # nan
    print(f"Recall: {get_recall_score(y_true_7, y_pred_proba_7, threshold_7):.4f}")     # nan
    print(f"Precision: {get_precision_score(y_true_7, y_pred_proba_7, threshold_7):.4f}") # nan

    # --- Test Case 8: Different pos_label (e.g., 0 is positive) ---
    print("\n--- Test Case 8: pos_label = 0 ---")
    y_true_8 =       np.array([1, 0, 1, 0, 0, 1]) # 3 Neg (1s), 3 Pos (0s)
    y_pred_proba_8 = np.array([0.3, 0.8, 0.4, 0.7, 0.6, 0.2]) # Probabilities FOR CLASS 1
    threshold_8 = 0.5 # Threshold for predicting class 1
    pos_label_8 = 0   # We are interested in class 0 as positive

    # If y_pred_proba are probs for class 1, to get probs for class 0 (our pos_label):
    y_pred_proba_for_0 = 1 - y_pred_proba_8 # Probs for class 0: [0.7, 0.2, 0.6, 0.3, 0.4, 0.8]
    # Preds for class 0 (pos_label=0) @ threshold_0 = 0.5:
    #   [1,0,1,0,0,1] (where 1 means predicted as class 0)

    # y_true_binary (pos=0):  [0, 1, 0, 1, 1, 0]
    # y_pred_binary (pos=0):  [1, 0, 1, 0, 0, 1] (using y_pred_proba_for_0 and new threshold)
    # This example is tricky because y_pred_proba is usually defined as prob for `pos_label`.
    # Let's assume the function is smart or we adjust input.

    # Let's re-state: y_pred_proba should be probabilities for `pos_label`.
    # If pos_label=0, then y_pred_proba should be P(class=0).
    # Let original_y_pred_proba be P(class=1) = [0.3, 0.8, 0.4, 0.7, 0.6, 0.2]
    # If we want to evaluate class 0 as positive:
    # y_true_for_0 = [1,0,1,0,0,1] (original labels)
    # pos_label_for_0 = 0
    # y_pred_proba_for_0 = 1 - original_y_pred_proba = [0.7, 0.2, 0.6, 0.3, 0.4, 0.8]
    # Now use threshold for these probabilities, e.g. 0.5
    # y_pred_binary_for_0 @ 0.5: [1,0,1,0,0,1] (where 1 means predicted as class 0)

    # Compare with y_true:   [1,0,1,0,0,1] (original y_true)
    # pos_label=0            [F,T,F,T,T,F] (y_true_binary for pos_label=0)
    # preds for class 0:     [T,F,T,F,F,T] (y_pred_binary for pos_label=0)

    # TP (true=0, pred=0): indices 1, 3, 4 => 3
    # FP (true=1, pred=0): index 5 => 1
    # FN (true=0, pred=1): index 0, 2 => 2
    # TN (true=1, pred=1): (none, as preds are 0 or 1 for class 0) -> TN means true=1, pred=1 (where pred 1 means not class 0)
    # This mapping gets confusing. The internal (y_true_arr == pos_label) is key.

    # Let's use the provided function directly assuming y_pred_proba is for the given pos_label
    # y_true: [1,0,1,0,0,1]
    # y_pred_proba (for class 0): [0.7, 0.2, 0.6, 0.3, 0.4, 0.8]
    # pos_label: 0
    # threshold: 0.5
    # y_true_binary (pos=0): [0,1,0,1,1,0]
    # y_pred_binary (pos=0): [1,0,1,0,0,1] (1 if prob >= 0.5, else 0)
    # TP (true_bin=1, pred_bin=1): index 1 (0.2<0.5->0) no, index 3(0.3<0.5->0) no, index 4(0.4<0.5->0) no. -> 0
    # FP (true_bin=0, pred_bin=1): index 0(0.7>=0.5->1), index 2(0.6>=0.5->1), index 5(0.8>=0.5->1) -> 3
    # FN (true_bin=1, pred_bin=0): index 1(0.2<0.5->0), index 3(0.3<0.5->0), index 4(0.4<0.5->0) -> 3
    # TN (true_bin=0, pred_bin=0): (none) -> 0
    # Sum = 6. Okay.

    print(f"y_true: {y_true_8}")
    print(f"y_pred_proba (for pos_label=0): {y_pred_proba_for_0}") # Use the transformed probabilities
    print(f"threshold: {threshold_8}, pos_label: {pos_label_8}")
    cm8 = get_binary_confusion_matrix(y_true_8, y_pred_proba_for_0, threshold_8, pos_label_8)
    print(f"Confusion Matrix (pos_label=0): {cm8}") # Expected: {'tp': 0, 'fp': 3, 'fn': 3, 'tn': 0}
    print(f"Recall (pos_label=0): {get_recall_score(y_true_8, y_pred_proba_for_0, threshold_8, pos_label_8):.4f}") # 0/(0+3) = 0.0
    print(f"Precision (pos_label=0): {get_precision_score(y_true_8, y_pred_proba_for_0, threshold_8, pos_label_8):.4f}")# 0/(0+3) = 0.0

    # Note on Case 8: The key is that `y_pred_proba` MUST be the probabilities
    # of the class specified by `pos_label`. If a model outputs P(class=1) and
    # you want to evaluate for pos_label=0, you must transform y_pred_proba to P(class=0)
    # before passing it to these functions (i.e., 1 - P(class=1)).

    print("\n--- Example run complete ---")

if __name__ == "__main__":
    # Add the missing validation checks to get_binary_confusion_matrix in thresholded_metrics.py first!
    # Example of how to add them back (partial):
    """
    # In get_binary_confusion_matrix, after converting to np.array:
    if len(y_true_arr) != len(y_pred_proba_arr):
        raise ValueError("y_true and y_pred_proba must have the same length.")
    if len(y_true_arr) == 0: # Handle empty input arrays (already in your code but good place)
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    # Optional: Stricter check for probability values
    if not np.all((y_pred_proba_arr >= 0) & (y_pred_proba_arr <= 1)):
         raise ValueError("All y_pred_proba values must be between 0 and 1 (inclusive).")
    """
    run_examples()