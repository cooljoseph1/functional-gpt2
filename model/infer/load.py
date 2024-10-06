import os
from safetensors import safe_open

from .trees import dict_to_pytree, print_tree

tensors = {}
weight_path = os.path.join(
    os.path.dirname(__file__),
    "../weights/gpt2_renamed.safetensors"
)
with safe_open(weight_path, framework="pt", device="cpu") as f:
   for key in f.keys():
       tensors[key] = f.get_tensor(key)



weights_pytree = dict_to_pytree(tensors)

gpt2_config = {
    "embedding_size": 768,
    "num_transformers": 12,
    "num_heads": 12,
    "hidden_size": 3072
}

gpt2_pytree = {
    "transformers": weights_pytree["transformers"],
    "final_layer_norm": weights_pytree["final_layer_norm"]
}

text_embedder_config =  {
    "vocab_size": 50257,
    "embedding_size": 768
}

text_embedder_pytree = {
    "embeddings": weights_pytree["embedder"]["token_embeddings"]
}

position_embedder_config = {
    "max_position": 1024,
    "embedding_size": 768
}

position_embedder_pytree = {
    "embeddings": weights_pytree["embedder"]["position_embeddings"]
}


### Make sure that everything was loaded correctly ###
from ..nn.generative.gpt2 import gpt2p
from ..nn.embedding import text, position

assert gpt2p(gpt2_config)(gpt2_pytree)
assert text.embedp(text_embedder_config)(text_embedder_pytree)
assert position.embedp(position_embedder_config)(position_embedder_pytree)