import jax.numpy as jnp
from typing import Callable
import scipy
import numpy as np
import jax
from jax.typing import ArrayLike


def bfgs_wrapper(loss: Callable, init: np.ndarray):
    res = scipy.optimize.minimize(loss, init, method="BFGS")
    return res.x, res.hess_inv


def laplace_approximation(log_density: Callable, init: np.ndarray, optimization_method=bfgs_wrapper):
    """
    Compute the Laplace approximation of a density.
    """

    def loss(theta):
        return -log_density(theta)

    x, hess_inv = optimization_method(loss, init)
    return -log_density(x), x, hess_inv


def newton_descent(loss: Callable, init: ArrayLike):
    """
    Newton descent algorithm.
    """

    def update(x):
        grad = jax.grad(loss)(x)
        hess = jax.hessian(loss)(x)
        x = x - jax.scipy.linalg.solve(hess, grad)
        return x

    x = jax.lax.fori_loop(0, 100, lambda i, x: update(x), init)
    return x, jnp.linalg.pinv(jax.hessian(loss)(x))
