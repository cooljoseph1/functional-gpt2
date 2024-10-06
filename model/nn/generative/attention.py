from .linear import linear_forward, linearp
from ..primitive_functions import SQRT, UNBIND_QKV, SPLIT_HEADS, JOIN_HEADS, SOFTMAX, CAUSAL_MASK, MATRIX_TRANSPOSE

def attentionp(config):
    embedding_size = config["embedding_size"]
    qkv_dim = 3 * embedding_size

    def predicate(pytree):
        return ("project_qkv" in pytree
                and linearp(in_shape=embedding_size,
                            out_shape=qkv_dim,
                            bias=True)(pytree["project_qkv"])
                and "project_out" in pytree
                and linearp(in_shape=embedding_size,
                            out_shape=embedding_size,
                            bias=True)(pytree["project_out"]))
    
    return predicate

def attention_forward(config, pytree):
    num_heads = config["num_heads"]
    head_splitter = SPLIT_HEADS(num_heads)
    project_qkv_f = linear_forward(pytree["project_qkv"])
    project_out_f = linear_forward(pytree["project_out"])

    def forward(embeddings):
        qkv = project_qkv_f(embeddings)
        queries, keys, values = UNBIND_QKV(qkv)
        queries = head_splitter(queries)
        keys = head_splitter(keys)
        values = head_splitter(values)

        num_queries = queries.shape[-2]
        num_keys = keys.shape[-2]
        causal_mask = CAUSAL_MASK(num_queries, num_keys)

        weights = queries @ MATRIX_TRANSPOSE(keys)
        weights = weights / SQRT(num_keys)
        weights = weights - 1.0E9 * causal_mask
        weights = SOFTMAX(weights, dim=-1)
        answers = weights @ values
        
        embeddings = JOIN_HEADS(answers)
        embeddings = project_out_f(embeddings)
        return embeddings
    
    return forward
