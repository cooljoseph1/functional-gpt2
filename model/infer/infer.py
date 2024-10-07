from .tokenizer import gpt2_tokenizer
from .load import (gpt2_config, gpt2_pytree,
                   text_embedder_config, text_embedder_pytree,
                   position_embedder_config, position_embedder_pytree)

from ..nn.generative.gpt2 import gpt2_forward
from ..nn.embedding import text, position
from ..nn.primitives import ARRAY, SAMPLE_CATEGORICAL

text_embed_f = text.embed_forward(text_embedder_pytree)
text_unembed_f = text.unembed_forward(text_embedder_pytree)
position_embed_f = position.embed_forward(position_embedder_pytree)
gpt2_f = gpt2_forward(gpt2_pytree)

def forward(tokens):
    text_embeddings = text_embed_f(tokens)
    position_embeddings = position_embed_f(tokens)
    embeddings = text_embeddings + position_embeddings
    embeddings = gpt2_f(embeddings)
    all_logits = text_unembed_f(embeddings)
    return all_logits


def sample_token_id(logits):
    return SAMPLE_CATEGORICAL(logits).item()

def predict_next_token(tokens):
    tokens = ARRAY(tokens)
    all_logits = forward(tokens)
    logits = all_logits[-1]
    return sample_token_id(logits)

def infer_bytes(text: str, num_tokens: int = -1):
    """Infer the next `num_tokens` tokens for the given text, returning the result as a string"""
    token_ids = list(gpt2_tokenizer.encode(text))
    count = 0
    while count != num_tokens:
        next_token_id = predict_next_token(token_ids)
        yield gpt2_tokenizer.decode_single_token_bytes(next_token_id)
        token_ids.append(next_token_id)
        count += 1