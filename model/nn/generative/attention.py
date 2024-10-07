from .project import project_forward, projectp
from ..primitives import SQRT, UNBIND, SOFTMAX, CAUSAL_MASK, SWAPAXES

def attentionp(config):
    embedding_size = config["embedding_size"]
    num_heads = config["num_heads"]

    def predicate(pytree):
        return ("project_qkv" in pytree
                and projectp(in_shape=embedding_size,
                            out_shape=(3, num_heads, embedding_size // num_heads),
                            bias=True)(pytree["project_qkv"])
                and "project_out" in pytree
                and projectp(in_shape=(num_heads, embedding_size // num_heads),
                            out_shape=embedding_size,
                            bias=True)(pytree["project_out"]))
    
    return predicate

def attention_forward(pytree):
    project_qkv_f = project_forward(pytree["project_qkv"], out_dims=3)
    project_out_f = project_forward(pytree["project_out"], in_dims=2)

    def forward(embeddings):
        qkv = project_qkv_f(embeddings)
        qkv = SWAPAXES(qkv, -2, -4) # Swap position and head axes
        queries, keys, values = UNBIND(qkv, -3)

        num_queries = queries.shape[-2] 
        num_keys = keys.shape[-2]
        causal_mask = CAUSAL_MASK(num_queries, num_keys)

        weights = queries @ SWAPAXES(keys, -1, -2)
        weights = weights / SQRT(num_keys)
        weights = weights - 1.0E9 * (~causal_mask)
        weights = SOFTMAX(weights, dim=-1)
        answers = weights @ values
        answers = SWAPAXES(answers, -2, -3) # Swap position and head axes
        embeddings = project_out_f(answers)
        return embeddings
    
    return forward
