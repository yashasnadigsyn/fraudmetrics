import numpy as np
from typing import Union, Tuple, Optional, Any
import warnings
from .validate_inputs import validate_binary_inputs

def validate_binary_inputs(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    pos_label: Any,
    threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]: