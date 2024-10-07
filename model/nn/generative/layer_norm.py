from ..primitives import LAYER_NORM

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

    def forward(embeddings):
        return LAYER_NORM(
            embeddings,
            weight=weight,
            bias=bias
        )
    
    return forward