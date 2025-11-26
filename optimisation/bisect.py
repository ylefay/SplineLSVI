import jax
import jax.scipy.optimize as optimize
import jax.numpy as jnp
from jax import lax


def jax_bisect(f, a, b, maxiter=100, tol=1e-7):
    """Find root of f(x)=0 on [a,b] assuming f(a)*f(b) <= 0."""
    fa = f(a)
    fb = f(b)

    def body(state):
        a, b, fa, fb, i = state
        m = 0.5 * (a + b)
        fm = f(m)

        left = fa * fm <= 0

        a_new = jnp.where(left, a, m)
        fa_new = jnp.where(left, fa, fm)

        b_new = jnp.where(left, m, b)
        fb_new = jnp.where(left, fm, fb)

        return (a_new, b_new, fa_new, fb_new, i + 1)

    def cond(state):
        a, b, fa, fb, i = state
        return jnp.logical_and(jnp.abs(b - a) > tol, i < maxiter)

    a_fin, b_fin, _, _, _ = lax.while_loop(
        cond,
        body,
        (a, b, fa, fb, 0)
    )
    return 0.5 * (a_fin + b_fin)
