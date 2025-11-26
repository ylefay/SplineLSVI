from jax.scipy import optimize
import jax.numpy as jnp

def find_mode(f, warm_start=0.0, bounds=None):
    """
    Find mode of a univariate function f(x) by minimizing -f(x).
    Uses jax.scipy.optimize.minimize with BFGS.
    """

    def objective(x):
        return -f(x[0])  # minimize expects vector input

    x0 = jnp.array([warm_start])
    result = optimize.minimize(objective, x0, method="bfgs")

    return result.x[0]
