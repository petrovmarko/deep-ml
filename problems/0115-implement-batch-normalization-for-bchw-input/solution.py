import torch

def batch_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Perform Batch Normalization on a 4D tensor in BCHW format."""
     
    B, C, H, W = X.shape
    D = X.transpose(0,1).contiguous().view(C,H*W*B)
    uB = D.mean(dim=1).view(1,C,1,1)
    sB2 = D.var(dim=1, unbiased=False).view(1,C,1,1)
    X = (X - uB) / (sB2 + epsilon)**0.5
    y = gamma * X + beta
    return y