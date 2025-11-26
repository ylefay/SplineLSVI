import jax.numpy as jnp
import jax
from typing import Callable


def lsvi(OP_key: jax.Array, log_concave_sampler_builder: Callable, tgt_log_density: Callable,
         n_iter: int, n_samples: int,
         regression, lr_schedule=1.0, return_all=False, sanity=lambda _: False, target_residual_schedule=jnp.inf):
    raise NotImplementedError
