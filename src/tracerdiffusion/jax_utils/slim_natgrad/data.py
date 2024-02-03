"""
Module to handle data related functionality.

Contains a DataSet class that is similar to a TensorDataset. 
"""


from jax import random
import jax.numpy as jnp


# something like a tensor dataset...
class DataSet:
    def __init__(self, inputs, labels):
        """
        inputs is a tensor of shape (n, d) and corresponds to the n
        data points where we have observations. labels is of shape
        (n, dim_data) where n is again the number of observations
        and dim_data depends on whether the observations are scalar
        or vectors.

        """
        if len(inputs) != len(labels):
            raise ValueError(
                f"[Constructor DataSet: inputs and labels not same " f"length]"
            )

        self._inputs = jnp.array(inputs)
        self._labels = jnp.array(labels)
        self._length = len(inputs)

    def sample(self, key, N=50):
        idx = random.randint(
            key,
            shape=(N,),
            minval=0,
            maxval=self._length,
        )

        return self._inputs[idx], self._labels[idx]


class DataIntegrator:
    """
    Something is off here as now the dataset
    is both present in the DataSet class as a member variable
    and in the integrator...

    """

    def __init__(
        self,
        key,
        dataset,
        N=50,
        loss=lambda x: x**2,
    ):
        self._N = N
        self._key = key
        self._dataset = dataset
        self._loss = loss
        self._x, self._y = self._dataset.sample(self._key, self._N)

    def __call__(self, f):
        """Inteded to pass to gramian"""
        return jnp.mean(f(self._x), axis=0)

    def data_loss(self, f):
        """Intended to use in loss function"""
        return 0.5 * jnp.mean(self._loss(f(self._x) - self._y))

    def new_rand_points(self):
        self._key = random.split(self._key, num=1)[0]
        self._x, self._y = self._dataset.sample(self._key, self._N)
