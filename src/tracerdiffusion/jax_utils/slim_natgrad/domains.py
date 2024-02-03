from jax import random
import jax.numpy as jnp


class DummyDomain:
    def __init__(self, pdepoints):
        # shape (n, 2) general (n,d)
        self._pdepoints = jnp.array(pdepoints)
        self._num = len(self._pdepoints)

    def random_integration_points(self, key, N):
        idx = random.randint(key, shape=(N,), minval=0, maxval=self._num)
        # return self._pdepoints[idx,:]  !nachlesen!
        return self._pdepoints[idx]

    def measure(self):
        return 1.0


class TimeDomain:
    """
    Implements I x Omega, where I is given through two floats
    and Omega is a domain class. If t_0 = t_1 then its the timeslice
    with value t_1.

    todos: alternative constructor with TimeInterval,
    proper documentation, checks if arguments are within bounds etc.

    """

    def __init__(self, t_0, t_1, domain):
        self._t_0 = t_0
        self._t_1 = t_1
        self._omega = domain

    def measure(self) -> float:
        # NOT GOOD for time slices!
        return 1.0
        if self._t_0 == self._t_1:
            return self._omega.measure()
        else:
            return (self._t_1 - self._t_0) * self._omega.measure()

    def random_integration_points(self, key, N: int = 50):
        key_1, key_2 = random.split(key, num=2)

        t = random.uniform(
            key_1,
            shape=(N, 1),
            minval=jnp.broadcast_to(
                self._t_0,
                shape=(N, 1),
            ),
            maxval=jnp.broadcast_to(
                self._t_1,
                shape=(N, 1),
            ),
        )
        x = self._omega.random_integration_points(key_2, N)
        return jnp.concatenate((t, x), axis=1)
