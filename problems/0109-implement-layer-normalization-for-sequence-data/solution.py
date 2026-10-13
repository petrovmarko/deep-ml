import torch

def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Layer Normalization.
    """
    # Your code here
    u = X.mean(dim=-1, keepdim=True)
    var = X.var(dim=-1, correction=0, keepdim=True)
    out = gamma * (X - u) / (var + epsilon) ** 0.5 + beta
    return out