## Rank based metrics

import numpy as np
import pandas as pd
from typing import Union, Any, Callable
from .utils.validate_inputs import _prepare_ranked_data

def get_precision_at_topk(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    k: int,
    pos_label: Any = 1
) -> float:
    """      
    precision_at_topk(y_true, y_pred_proba, k, pos_label)

    Calculates the proportion of true positive instances
    among the top K instances ranked by their predicted probabilities.

    Args:
        y_true (Union[list, np.ndarray]): True labels. Values will be
                                         binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
                                         Values should be in the range [0, 1].
                                         Must be the same length as `y_true`.
        k (int): 
            The number of top-ranked instances to consider.
            If k is 0, np.nan is returned.
            If k is larger than the total number of instances, it will be capped
            at the total number of instances.
        pos_label (Any, optional): The label of the positive class in `y_true`.
                                   Defaults to 1.

    Returns:
        float: The precision@K score. Returns np.nan if k=0 or if inputs are empty
               and precision cannot be computed.
    """
    try:
        y_true_binary_sorted, validated_k = _prepare_ranked_data(y_true, y_pred_proba, k, pos_label)
    except ValueError:
        return np.nan
    
    ## Take topk values and compute precision_at_topk
    y_true_topk = y_true_binary_sorted[:validated_k]
    tp_at_k = np.sum(y_true_topk)
    precision_at_topk = float(tp_at_k / validated_k)

    return precision_at_topk

def get_recall_at_topk(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    k: int,
    pos_label: Any = 1
) -> float:
    """
    get_recall_at_topk(y_true, y_pred_proba, k, pos_label)

    Calculates the proportion of True Positives in topk divided by Total Actual Positives.

    Args:
        y_true (Union[list, np.ndarray]): True labels. Values will be
                                         binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
                                         Values should be in the range [0, 1].
                                         Must be the same length as `y_true`.
        k (int): 
            The number of top-ranked instances to consider.
            If k is 0, np.nan is returned.
            If k is larger than the total number of instances, it will be capped
            at the total number of instances.
        pos_label (Any, optional): The label of the positive class in `y_true`.
                                   Defaults to 1.

    Returns:
        float: The recall@K score. Returns np.nan if k=0 or if inputs are empty
               and recall cannot be computed.
    """
    try:
        y_true_binary_sorted, validated_k = _prepare_ranked_data(y_true, y_pred_proba, k, pos_label)
    except ValueError:
        return np.nan
    
    ## Calculate total actual positives from original unsorted data
    y_true_arr = np.array(y_true)
    total_actual_positives = np.sum(y_true_arr == pos_label)
    
    if total_actual_positives == 0:
        return np.nan
    
    ## Take topk values and compute recall_at_topk
    y_true_topk = y_true_binary_sorted[:validated_k]
    tp_at_k = np.sum(y_true_topk)
    recall_at_topk = float(tp_at_k / total_actual_positives)

    return recall_at_topk

def get_card_precision_at_topk(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    ids: Union[list, np.ndarray],
    k: int,
    aggregation_func: Union[str, Callable] = 'max',
    pos_label: Any = 1
) -> float:
    """
    get_card_precision_at_topk(y_true, y_pred_proba, ids, k, pos_label)

    This metric ranks cards/entities based on an aggregation of their transaction
    scores, selects the top K entities, and then calculates the proportion
    of these top K entities that are "truly positive" (i.e., had at
    least one transaction belonging to the positive class).

    Args:
        y_true (Union[list, np.ndarray]): True labels. Values will be
                                         binarized based on `pos_label`.
        y_pred_proba (Union[list, np.ndarray]): Predicted probabilities for the positive class.
                                         Values should be in the range [0, 1].
                                         Must be the same length as `y_true`.
        ids (Union[list, np.ndarray]): ID (e.g., card ID) for each transaction.
        k (int): 
            The number of top-ranked instances to consider.
            If k is 0, np.nan is returned.
            If k is larger than the total number of instances, it will be capped
            at the total number of instances.
        aggregation_func (Union[str, Callable], optional): 
            How to aggregate transaction scores to an entity score.
            Can be 'max', 'mean', 'sum', or a custom callable function
            that takes a pandas Series of scores and returns a single score.
            Defaults to 'max'.
        pos_label (Any, optional): The label of the positive class in `y_true`.
                                   Defaults to 1.

    Returns:
        float: The precision@K score. Returns np.nan if k=0 or if inputs are empty, or
        other undefined conditions.
    """

    ## Validate inputs
    if not isinstance(y_true, (list, np.ndarray)):
        raise TypeError("y_true_transactions must be a list or NumPy array.")
    if not isinstance(y_pred_proba, (list, np.ndarray)):
        raise TypeError("y_pred_scores_transactions must be a list or NumPy array.")
    if not isinstance(ids, (list, np.ndarray)):
        raise TypeError("entity_ids_transactions must be a list or NumPy array.")
    if not isinstance(k, int):
        raise TypeError(f"k must be an integer, not {type(k)}")

    ## Convert to numpy array
    y_true_arr = np.array(y_true)
    y_pred_proba_arr = np.array(y_pred_proba)
    ids_arr = np.array(ids)

    ## Check for length
    if not (len(y_true_arr) == len(y_pred_proba_arr) == len(ids_arr)):
        raise ValueError("All transaction-level inputs (y_true, y_pred_scores, entity_ids) must have the same length.")

    ## Handle empty case
    if len(y_true_arr) == 0:
        return np.nan
    
    ## Handle edge cases for k
    if k < 0:
        raise ValueError("k cannot be negative.")
    if k == 0:
        return np.nan
    
    ## Convert to pandas dataframe
    df = pd.DataFrame({
        'entity_id': ids_arr,
        'score': y_pred_proba_arr,
        'is_positive_transaction': (y_true_arr == pos_label).astype(int)
    })

    ## Aggregate transactions for each entity
    if aggregation_func == 'max':
        entity_scores = df.groupby('entity_id')['score'].max()
    elif aggregation_func == 'mean':
        entity_scores = df.groupby('entity_id')['score'].mean()
    elif aggregation_func == 'sum':
        entity_scores = df.groupby('entity_id')['score'].sum()
    elif callable(aggregation_func):
        entity_scores = df.groupby('entity_id')['score'].apply(aggregation_func)
    else:
        raise ValueError("aggregation_func must be 'max', 'mean', 'sum', or a callable.")
    
    entity_true_positive_status = df.groupby('entity_id')['is_positive_transaction'].max()

    ## Combine into a new DataFrame for entities
    entity_df = pd.DataFrame({
        'agg_score': entity_scores,
        'is_truly_positive_entity': entity_true_positive_status
    }).reset_index()

    num_unique_entities = len(entity_df)

    ## Cap k at the total number of unique entities
    k = min(k, num_unique_entities)
    if k == 0:
        return np.nan
    
    ## Sort entities by their aggregated scores
    entity_df_sorted = entity_df.sort_values(by='agg_score', ascending=False, kind='mergesort')

    ## Select the top K entities
    top_k_entities_df = entity_df_sorted.head(k)

    ## Calculate precision for these top K entities
    tp_entities_at_k = top_k_entities_df['is_truly_positive_entity'].sum()
    card_precision_at_topk = tp_entities_at_k / k
    
    return float(card_precision_at_topk)