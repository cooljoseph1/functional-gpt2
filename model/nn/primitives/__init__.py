from ...env_vars import ML_FRAMEWORK

if ML_FRAMEWORK.lower() == "torch":
    from .torch_primitives import *
elif ML_FRAMEWORK.lower() == "jax":
    # Default to using JAX
    from .jax_primitives import *
else:
    raise RuntimeError("Invalid ML_FRAMEWORK", ML_FRAMEWORK)