from ..primitives import GELU
from .project import project_forward, projectp

def feed_forwardp(config):
    embedding_size = config["embedding_size"]
    hidden_size = config["hidden_size"]

    def predicate(pytree):
        return ("project_hidden" in pytree
                and projectp(in_shape=(embedding_size,),
                            out_shape=(hidden_size,),
                            bias=True)(pytree["project_hidden"])
                and "project_out" in pytree
                and projectp(in_shape=(hidden_size,),
                            out_shape=(embedding_size,),
                            bias=True)(pytree["project_out"]))
    
    return predicate

def feed_forward_forward(pytree):
    project_hidden_f = project_forward(pytree["project_hidden"])
    project_out_f = project_forward(pytree["project_out"])

    def forward(embeddings):
        embeddings = project_hidden_f(embeddings)
        embeddings = GELU(embeddings) # activation function
        embeddings = project_out_f(embeddings)
        return embeddings
    
    return forward