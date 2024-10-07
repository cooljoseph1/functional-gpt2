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
    """Returns x @ W + b, the affine projection. If b is None, the result is x @ W."""
    y = x @ W
    if b is not None:
        y = y + b
    return y

GELU = jax.nn.gelu

def SOFTMAX(x, dim=-1):
    return jax.nn.softmax(x, axis=dim)

def MATRIX_TRANSPOSE(x):
    return x.swapaxes(-1, -2)

def UNBIND_QKV(qkv):
    new_shape = qkv.shape[:-1] + (3, qkv.shape[-1] // 3)
    reshaped = qkv.reshape(new_shape)
    return jnp.unstack(reshaped, axis=-2)

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
    return jnp.tril(jnp.ones((num_queries, num_keys))).astype(bool)


def GET_EMBEDDINGS(embeddings, inputs):
    return embeddings[inputs]

SQRT = math.sqrt


_jax_key = jax.random.PRNGKey(3872624641)
def SAMPLE_CATEGORICAL(logits):
    global _jax_key
    _jax_key, subkey = jax.random.split(_jax_key)
    return jax.random.categorical(subkey, logits)