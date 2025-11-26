import jax.numpy as jnp
import jax
from jax import lax, random
from optimisation.bisect import jax_bisect
from jax.scipy.optimize import minimize


def failback_find_sz(f, rho):
    def solver(f, x0):
        return minimize(f, x0, method="bfgs").x

    gs = lambda x: f(-jnp.abs(x)) + rho
    gz = lambda x: f(jnp.abs(x)) + rho

    s = jnp.abs(solver(gs, x0=0)[0])
    z = jnp.abs(solver(gz, x0=0)[0])
    return s, z


def find_sz(f, rho, interval_for_finding_sz):
    left, right = interval_for_finding_sz
    g = lambda x: f(x) + rho

    # If bisection fails assumptions, fallback automatically.
    def use_bisect(_):
        s = -jax_bisect(g, left, 0.0)
        z = jax_bisect(g, 0.0, right)
        return (s, z)

    def use_fallback(_):
        return failback_find_sz(f, rho)

    # Check conditions
    cond_good = jnp.logical_and(
        jnp.logical_and(left <= 0, right >= 0),
        jnp.logical_and(g(right) * g(0.0) <= 0, g(left) * g(0.0) <= 0),
    )

    return lax.cond(cond_good, use_bisect, use_fallback, operand=None)


def log_concave_sampler(psi_dpsi, rho, interval_for_finding_sz):
    r"""
        This is an implementation of the generic algorithm proposed in
        Devroye, Random variate generation for the generalized inverse Gaussian distribution, 2014,
        for sampling from univariate log-concave distributions p\propto e^{\psi}
        """
    psi, dpsi = psi_dpsi

    def chi(x, rho, s_tilde, z_tilde, dzeta, xi):
        # vectorized piecewise version
        return jnp.where(
            (-s_tilde <= x) & (x <= z_tilde),
            1.0,
            jnp.where(
                x >= z_tilde,
                jnp.exp(-rho - dzeta * (x - z_tilde)),
                jnp.exp(-rho + xi * (x + s_tilde)),
            ),
        )

    s, z = find_sz(psi, rho, interval_for_finding_sz)

    dzeta, xi = -dpsi(z), dpsi(-s)
    p, r = 1.0 / xi, 1.0 / dzeta
    z_tilde, s_tilde = z - r * rho, s - p * rho
    q = z_tilde + s_tilde

    # one-sample drawing loop for a single PRNG key
    def draw_one(key):
        def body(state):
            key = state["key"]

            key_u, key_v, key_w, key_next = random.split(key, 4)
            U = random.uniform(key_u)
            V = random.uniform(key_v)
            W = random.uniform(key_w)

            # proposal
            qp = q + p + r
            x = jnp.where(
                U * qp < q,
                -s_tilde + q * V,
                jnp.where(
                    U * qp <= (q + r),
                    z_tilde - r * jnp.log(V),
                    -s_tilde + p * jnp.log(V),
                ),
            )

            accept = W * Chi(x, rho, s_tilde, z_tilde, dzeta, xi) <= jnp.exp(psi(x))

            new_state = {
                "x": jnp.where(accept, x, state["x"]),
                "key": key_next,
                "done": accept
            }
            return new_state

        def cond(state):
            return ~state["done"]

        init_state = {"x": 0.0, "key": key, "done": False}
        final_state = lax.while_loop(cond, body, init_state)
        return final_state["x"]

    def sampler(keys):
        return jax.vmap(draw_one)(keys)

    return sampler
