import torch

def grad_of_quadratic(x_value: float) -> float:
    # TODO: build a tracked leaf for x, compute f(x), run backprop, return df/dx as a 
    x = torch.tensor(x_value)
    x.requires_grad=True
    f = x ** 2 + 3 * x + 2
    f.backward()
    return x.grad.item()
