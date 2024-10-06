
# TODO: Allow multiple ML frameworks by implementing in pytorch, etc.,
#       and then use an environment variable to select which framework to
#       work in

import torch
import torch.nn.functional as functional
from math import sqrt

LAYER_NORM = functional.layer_norm
def AFFINE(x, W, b=None):
    """Returns x @ W + b, the affine projection. If b is None, the result is x @ W."""
    y = x @ W
    if b is not None:
        y = y + b
    return y

GELU = functional.gelu

SOFTMAX = functional.softmax

def MATRIX_TRANSPOSE(x):
    return x.transpose(-1, -2)

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
        return reshaped.transpose(-2, -3) # Transpose so the head dimension is before the position dimension
    return func

def JOIN_HEADS(x):
    """
    Joins the last two dimensions of the tensor
    """
    x = x.transpose(-2, -3) # Swap so the head dimnesion is next to the embedding dimension
    new_shape = x.shape[:-2] + (x.shape[-2] * x.shape[-1],)
    return x.reshape(new_shape)

def CAUSAL_MASK(num_queries, num_keys):
    return torch.triu(torch.ones(num_queries, num_keys), diagonal=1).bool()


def GET_EMBEDDINGS(embeddings, inputs):
    return functional.embedding(inputs, embeddings)

SQRT = sqrt