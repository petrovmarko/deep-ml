import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    N = len(predicted_probs)
    M = len(predicted_probs[0])
    entropy = 0
    for i in range(N):
        for j in range(M):
            entropy += true_labels[i][j] * np.log(predicted_probs[i][j] + epsilon)
    entropy /= N 
    return -entropy
