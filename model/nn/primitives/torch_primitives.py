import torch
import torch.nn.functional as functional
from math import sqrt

ARRAY = torch.tensor

def LAYER_NORM(x, weight, bias, eps=1e-5):
    return functional.layer_norm(x, weight.shape, weight, bias, eps=eps)

def AFFINE(x, W, b=None):
    """
    Returns x @ W + b, the affine projection. If b is None, the result is x @ W.
    
    Note: It's slightly simpler in PyTorch to compute W @ x + b, but it's mathematically
          more natural to compute x @ W + b. Thus, we go with the latter.

    Note 2: This function only works if W is a degree 2 tensor. Sorry. If you want to use
            a higher degree tensor you will need to reshape it before and after applying
            AFFINE.
    """
    y = x @ W
    if b is not None:
        y = y + b
    return y

GELU = functional.gelu

SOFTMAX = functional.softmax

def SWAPAXES(x, axis1, axis2):
    return x.swapaxes(axis1, axis2)

def UNBIND(x, dim=0):
    return x.unbind(dim)

def CAUSAL_MASK(num_queries, num_keys):
    return torch.tril(torch.ones((num_queries, num_keys))).bool()


def GET_EMBEDDINGS(embeddings, inputs):
    return functional.embedding(inputs, embeddings)

SQRT = sqrt

def SAMPLE_CATEGORICAL(logits):
    dist = torch.distributions.Categorical(logits=logits)
    return dist.sample()