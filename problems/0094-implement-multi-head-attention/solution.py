import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    # (seq, d_model) @ (head, d_model, d_model)
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return (Q, K, V)

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    return torch.softmax((Q @ K.transpose(-2,-1)) / (K.shape[-1] ** 0.5), dim=-1) @ V

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    
    seq_len, d_model = Q.shape[0], Q.shape[1]
    dk = d_model // n_heads

    Qs = Q.view(seq_len, n_heads, dk).transpose(0,1)
    Ks = K.view(seq_len, n_heads, dk).transpose(0,1)
    Vs = V.view(seq_len, n_heads, dk).transpose(0,1)
    
    out = self_attention(Qs, Ks, Vs)
    # (heads, seq_len, dk)
    out = out.transpose(0,1).contiguous().view(seq_len, d_model)
    return out