from ..primitive_functions import AFFINE

def linearp(in_shape, out_shape, bias=True):
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

def linear_forward(pytree):
    weight = pytree["weight"]
    bias = pytree.get("bias", None)
    
    def forward(embeddings):
        return AFFINE(embeddings, weight, bias)

    return forward