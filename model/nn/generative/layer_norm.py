from ..primitive_functions import LAYER_NORM

def layer_normp(config):
    embedding_size = config["embedding_size"]
    
    def predicate(pytree):
        return ("weight" in pytree
                and pytree["weight"].shape == (embedding_size,)
                and "bias" in pytree
                and pytree["bias"].shape == (embedding_size,))
    
    return predicate

def layer_norm_forward(pytree):
    weight = pytree["weight"]
    bias = pytree["bias"]

    # For some reason pytorch requires normalized_shape to be given, even when
    # weight and bias are also provided:
    normalized_shape = weight.shape

    def forward(embeddings):
        return LAYER_NORM(
            embeddings, 
            normalized_shape,
            weight=weight,
            bias=bias
        )
    
    return forward