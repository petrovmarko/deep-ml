import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    # Your code here

	updated_weights = initial_weights.clone()
	updated_bias = torch.tensor(initial_bias)
	n = features.shape[0]
	mse_values = []
	for epoch in range(epochs):
		out = torch.sigmoid(features @ updated_weights + updated_bias)
		L = ((out - labels) ** 2).mean()
        dout = out * (1 - out)
        dW = (2/n) * (features.T @ ((out - labels) * dout))
        dB = (2/n) * ((out - labels) * dout).sum()
		updated_bias = updated_bias - learning_rate * dB
		updated_weights = updated_weights - learning_rate * dW
		mse_values.append(round(L.item(),4))
	return updated_weights.tolist(), updated_bias.item(), mse_values