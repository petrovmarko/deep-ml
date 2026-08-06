import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    scores = np.array(scores)
    scores = scores - scores.max()
    scores = np.exp(scores) / np.exp(scores).sum()
    return scores