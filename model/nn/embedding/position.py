"""
An embedder from text tokens to vectors.

A pytree for this embedder looks like:
{
    "embeddings": [max_position, embedding_size]
}

A config looks like:
{
    "max_position": 1024,
    "embedding_size": 768
}
"""

def embedp(config):
    max_position = config["max_position"]
    embedding_size = config["embedding_size"]

    def predicate(pytree):
        return ("embeddings" in pytree
                and pytree["embeddings"].shape == (max_position, embedding_size))
    
    return predicate


def embed_forward(pytree):
    embeddings = pytree["embeddings"]

    def forward(ids):
        return embeddings[:ids.shape[-1]]

    return forward