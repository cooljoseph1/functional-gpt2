"""
An embedder from text tokens to vectors.

A pytree for this embedder looks like:
{
    "embeddings": [vocab_size, embedding_size]
}

A config looks like:
{
    "vocab_size": 50257,
    "embedding_size": 768
}
"""

from ..primitives import GET_EMBEDDINGS, SWAPAXES

def embedp(config):
    vocab_size = config["vocab_size"]
    embedding_size = config["embedding_size"]

    def predicate(pytree):
        return ("embeddings" in pytree
                and pytree["embeddings"].shape == (vocab_size, embedding_size))
    
    return predicate


def embed_forward(pytree):
    embeddings = pytree["embeddings"]

    def forward(ids):
        return GET_EMBEDDINGS(embeddings, ids)

    return forward


def unembed_forward(pytree):
    keys = SWAPAXES(pytree["embeddings"], -1, -2)

    def forward(query_embeddings):
        return query_embeddings @ keys
    
    return forward