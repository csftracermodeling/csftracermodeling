"""
This module contains the implementation of integration routines.

"""
import jax.numpy as jnp
from jax import random


class EvolutionaryIntegrator:
    """
    First Implementation of an evolutionary integrator following
    the proposed algorithm of Daw et al in "Mitigating Propagation
    Failure in PINNs using Evolutionary Sampling".

    """

    def __init__(self, domain, key, N=50, K=None):
        self._domain = domain
        self._N = N
        self._key = key
        self._x = self._domain.random_integration_points(self._key, N)
        self._key = random.split(self._key, num=1)
        self._K = K

        if K is not None:
            splits = [i * len(self._x) // K for i in range(1, K)]
            self.x_split = jnp.split(self._x, splits, axis=0)

    def __call__(self, f):
        """
        Integration happens here, f must map (n,...) to (n,...)
        """

        mean = 0
        if self._K is not None:
            means = []
            for x in self.x_split:
                means.append(jnp.mean(f(x), axis=0))

            mean = jnp.mean(jnp.array(means), axis=0)

        else:
            y = f(self._x)
            mean = jnp.mean(y, axis=0)

        return self._domain.measure() * mean

    def new_rand_points(self):
        # new_key = random.split(self._key[0], num=1)
        self._key = random.split(self._key[0], num=1)
        new_key = self._key
        self._x = self._domain.random_integration_points(new_key, self._N)

        if self._K is not None:
            splits = [i * len(self._x) // self._K for i in range(1, self._K)]
            self.x_split = jnp.split(self._x, splits, axis=0)

    def update(self, residual):
        # compute fitness from residual
        fitness = jnp.abs(residual(self._x))

        # set the threshold
        threshold = jnp.mean(fitness)

        # remove non-fit collocation points
        mask = jnp.where(fitness > threshold, False, True)
        x_fit = jnp.delete(self._x, mask, axis=0)

        # add new uniformly drawn collocation points to fill up
        N_fit = len(self._x) - len(x_fit)
        x_add = self._domain.random_integration_points(self._key[0], N_fit)
        self._x = jnp.concatenate([x_fit, x_add], axis=0)

        # advance random number generator
        self._key = random.split(self._key[0], num=1)
