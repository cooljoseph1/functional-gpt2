import jax.numpy as jnp
import jax
import math

ARRAY = jnp.array

def LAYER_NORM(x, weight, bias, eps=1e-5):
    """
    Perform a layer norm along the last dimensions of x.

    Adapted from the implementation in Equinox by Patrick Kidger.
    """
    num_dims = len(weight.shape)
    dims = tuple(range(-num_dims, 0))
    mean = x.mean(axis=dims, keepdims=True)
    variance = x.var(axis=dims, keepdims=True)
    variance = jnp.maximum(0.0, variance)
    inverse = jax.lax.rsqrt(variance + eps)
    out = (x - mean) * (inverse * weight) + bias
    return out

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

GELU = jax.nn.gelu

def SOFTMAX(x, dim=-1):
    return jax.nn.softmax(x, axis=dim)

def SWAPAXES(x, axis1, axis2):
    return x.swapaxes(axis1, axis2)

def UNBIND(x, dim=0):
    return jnp.unstack(x, axis=dim)

def CAUSAL_MASK(num_queries, num_keys):
    return jnp.tril(jnp.ones((num_queries, num_keys))).astype(bool)


def GET_EMBEDDINGS(embeddings, inputs):
    return embeddings[inputs]

SQRT = math.sqrt


_jax_key = jax.random.PRNGKey(3872624641)
def SAMPLE_CATEGORICAL(logits):
    global _jax_key
    _jax_key, subkey = jax.random.split(_jax_key)
    return jax.random.categorical(subkey, logits)