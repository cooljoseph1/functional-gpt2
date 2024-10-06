"""
An example pytree for a GPT 2 model:

{
    "transformers": [
        {
            "attention": {
                "project_qkv": {
                    "weight": [768, 3*12*64],
                    "bias": [3*12*64]
                },
                "project_out": {
                    "weight": [12*64, 768],
                    "bias": [768]
                }
            },
            "feed_forward": {
                "project_hidden": {
                    "weight": [768, 3072],
                    "bias": [3072]
                },
                "project_out": {
                    "weight": [3072, 768],
                    "bias": [768]
                }
            },
            "layer_norm1": {
                ...
            },

            "layer_norm2": {
                ...
            }
        },
        ...
    ],

    "final_layer_norm": {
        "weight": [768],
        "bias": [768]
    }
}

Note that the embedder is *NOT* a part of the GPT 2 model--this is because
converting things to embeddings is ultimately *NOT* the same as reasoning about those
embeddings. From a moral position, embeddings belong elsewhere. This is also important
because often embeddings are created in completely different ways--maybe an image
needs embeddings too.

The config should look like

{
    "embedding_size": 768,
    "num_transformers": 12,
    "num_heads": 12,
    "hidden_size": 3072
}

Note that the number of heads must be a factor of the embedding size.
"""

from .layer_norm import layer_norm_forward, layer_normp
from .transformer import transformer_forward, transformerp

def gpt2p(config):
    """
    Returns a predicate to determine whether the given pytree is a gpt2 model.
    """
    def predicate(pytree):
        """Returns True if the given pytree is a gpt2 model"""
        return ("transformers" in pytree
                and len(pytree["transformers"]) == config["num_transformers"]
                and all(transformerp(config)(transformer) for transformer in pytree["transformers"])
                and "final_layer_norm" in pytree
                and layer_normp(config)(pytree["final_layer_norm"]))

    return predicate

def gpt2_forward(config, pytree):
    """
    Construct the forward function for a gpt2 pytree
    """
    transformer_fs = [
        transformer_forward(config, transformer_pytree)
        for transformer_pytree in pytree["transformers"]
    ]
    final_layer_norm_f = layer_norm_forward(pytree["final_layer_norm"])
    
    def forward(embeddings):
        for transformer_f in transformer_fs:
            embeddings = transformer_f(embeddings)
        embeddings = final_layer_norm_f(embeddings)
        return embeddings

    return forward