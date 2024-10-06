from .attention import attention_forward, attentionp
from .feed_forward import feed_forward_forward, feed_forwardp
from .layer_norm import layer_norm_forward, layer_normp

def transformerp(config):
    def predicate(pytree):
        return ("attention" in pytree
                and attentionp(config)(pytree["attention"])
                and "feed_forward" in pytree
                and feed_forwardp(config)(pytree["feed_forward"])
                and "layer_norm1" in pytree
                and layer_normp(config)(pytree["layer_norm1"])
                and "layer_norm2" in pytree
                and layer_normp(config)(pytree["layer_norm2"]))
    
    return predicate

def transformer_forward(config, pytree):
    """
    Construct a forward function for a transformer pytree
    """
    layer_norm1_f = layer_norm_forward(pytree["layer_norm1"])
    attention_f = attention_forward(config, pytree["attention"])
    layer_norm2_f = layer_norm_forward(pytree["layer_norm2"])
    feed_forward_f = feed_forward_forward(pytree["feed_forward"])

    def forward(embeddings):
        normed_embeddings = layer_norm1_f(embeddings)
        embeddings = embeddings + attention_f(normed_embeddings)
        normed_embeddings = layer_norm2_f(embeddings)
        embeddings = embeddings + feed_forward_f(normed_embeddings)
        return embeddings
    
    return forward