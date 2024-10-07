import torch
import torch.nn.functional as functional
from math import sqrt

ARRAY = torch.tensor

def LAYER_NORM(x, weight, bias, eps=1e-5):
    return functional.layer_norm(x, weight.shape, weight, bias, eps=eps)

def AFFINE(x, W, b=None):
    """Returns x @ W + b, the affine projection. If b is None, the result is x @ W."""
    y = x @ W
    if b is not None:
        y = y + b
    return y

GELU = functional.gelu

SOFTMAX = functional.softmax

def MATRIX_TRANSPOSE(x):
    return x.swapaxes(-1, -2)

def UNBIND_QKV(qkv):
    new_shape = qkv.shape[:-1] + (3, qkv.shape[-1] // 3)
    reshaped = qkv.reshape(new_shape)
    return reshaped.unbind(-2)

def SPLIT_HEADS(num_heads):
    def func(embeddings):
        """
        Splits a tensor along its last dimension into multi-headed queries, keys, or values.
        """
        new_shape = embeddings.shape[:-1] + (num_heads, embeddings.shape[-1] // num_heads)
        reshaped = embeddings.reshape(new_shape)
        return reshaped.swapaxes(-2, -3) # Transpose so the head dimension is before the position dimension
    return func

def JOIN_HEADS(x):
    """
    Joins the last two dimensions of the tensor
    """
    x = x.swapaxes(-2, -3) # Swap so the head dimnesion is next to the embedding dimension
    new_shape = x.shape[:-2] + (x.shape[-2] * x.shape[-1],)
    return x.reshape(new_shape)

def CAUSAL_MASK(num_queries, num_keys):
    return torch.tril(torch.ones((num_queries, num_keys))).bool()


def GET_EMBEDDINGS(embeddings, inputs):
    return functional.embedding(inputs, embeddings)

SQRT = sqrt

def SAMPLE_CATEGORICAL(logits):
    dist = torch.distributions.Categorical(logits=logits)
    return dist.sample()