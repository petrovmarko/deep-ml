import torch

def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculates the precision metric for binary classification.
    
    Precision is defined as the ratio of true positives to the sum of 
    true positives and false positives.
    
    Args:
        y_true: True binary labels (1D tensor)
        y_pred: Predicted binary labels (1D tensor)
    
    Returns:
        Precision value as a scalar tensor
    """
    # Your implementation here
    A, B = y_pred, y_true
    if (A == 1).any() == False:
        return 0
    return ((A == B) & (A == 1)).sum() / (A == 1).sum()
