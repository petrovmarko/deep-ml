import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: store d_head, create q/k/v/out projections (all bias=False)
        self.Wq = nn.Parameter(torch.randn(d_model, d_model))
        self.Wk = nn.Parameter(torch.randn(d_model, d_model))
        self.Wv = nn.Parameter(torch.randn(d_model, d_model))
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.Wo = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None
        # TODO: project, reshape into heads, scaled dot-product, mask, softmax, combine, reshape back, out_proj
        #  (B, T, D) - > (B, T, n_heads, dk) -> (B, n_heads, T, dk)
        
        Q = (x @ self.Wq).view(x.shape[0],x.shape[1], self.num_heads, self.dk).transpose(1,2)
        K = (x @ self.Wk).view(x.shape[0],x.shape[1], self.num_heads, self.dk).transpose(1,2)
        V = (x @ self.Wv).view(x.shape[0],x.shape[1], self.num_heads, self.dk).transpose(1,2)

        # Q @ K.T -> (B, n_heads, T, T)
        # (B, n_heads, T, T) @ V -> (B, n_heads, T, dk)
        QK = Q @ K.transpose(-2,-1) / self.dk**0.5
        if mask is not None:
            QK = QK + mask
        QK = torch.softmax(QK, dim=-1)
        out = (QK) @ V
        out = out.transpose(-2,-3).contiguous().view(x.shape[0], x.shape[1], self.d_model)
        return out @ self.Wo
