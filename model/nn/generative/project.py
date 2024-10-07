from math import prod
from ..primitives import AFFINE

def projectp(in_shape, out_shape, bias=True):
    if isinstance(in_shape, int):
        in_shape = (in_shape,)
    if isinstance(out_shape, int):
        out_shape = (out_shape,)
    weight_shape = (*in_shape, *out_shape)
    bias_shape = out_shape

    def predicate_no_bias(pytree):
        return ("weight" in pytree
                and pytree["weight"].shape == weight_shape)

    def predicate_bias(pytree):
        return (predicate_no_bias(pytree)
                and "bias" in pytree
                and pytree["bias"].shape == bias_shape)
    
    predicate = predicate_bias if bias else predicate_no_bias
    return predicate

def project_forward(pytree, in_dims=1, out_dims=1):
    weight = pytree["weight"]
    bias = pytree.get("bias", None)
    
    # Reshape weight so that all the in_dims are combined together
    in_shape, out_shape = weight.shape[:in_dims], weight.shape[-out_dims:]
    in_size, out_size = prod(in_shape), prod(out_shape)
    reshaped_weight = weight.reshape((in_size, out_size))
    reshaped_bias = bias.reshape((out_size,))

    def forward(x):
        batch_shape = x.shape[:-in_dims]
        x  = x.reshape(batch_shape + (in_size,))

        y = AFFINE(x, reshaped_weight, reshaped_bias)

        y = y.reshape(batch_shape + out_shape)
        return y

    return forward