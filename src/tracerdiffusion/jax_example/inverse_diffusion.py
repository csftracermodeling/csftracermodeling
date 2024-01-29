import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import optax

import slim_natgrad.mlp as mlp
from slim_natgrad.integrators import EvolutionaryIntegrator
from slim_natgrad.derivatives import del_i
from slim_natgrad.domains import DummyDomain, TimeDomain
from slim_natgrad.data import DataSet, DataIntegrator


jax.config.update("jax_enable_x64", True)

# The following block of code culminates in defining omega.
# This needs to integrate smoothly with Ludmils potato brain.
# --------------------------------------------------------------#
inputs = jnp.load("jax_example/input.npy")
targets = jnp.load("jax_example/target.npy")
x = jnp.reshape(inputs[:, 0], (len(inputs), 1))
y = jnp.reshape(inputs[:, 1], (len(inputs), 1))
t = jnp.reshape(inputs[:, 2], (len(inputs), 1))
inputs = jnp.concatenate([t, x, y], axis=1)


minimum = jnp.load("jax_example/minimum.npy")
x = minimum[0]
y = minimum[1]
t = minimum[2]
minimum = jnp.array([t, x, y])

maximum = jnp.load("jax_example/maximum.npy")
x = maximum[0]
y = maximum[1]
t = maximum[2]
maximum = jnp.array([t, x, y])
assert minimum[0] != maximum[0]

# (x,y)
pdepoints = jnp.load("jax_example/pdepoints.npy")

omega = DummyDomain(pdepoints)

# -------------------------------------------------------------#

# random seed
seed = 0

# domains
interior = TimeDomain(minimum[0], maximum[0], omega)

# dataset
dataset = DataSet(inputs, targets)

# integrators
interior_integrator = EvolutionaryIntegrator(
    interior, key=random.PRNGKey(seed), N=10000
)
data_integrator = DataIntegrator(random.PRNGKey(0), dataset, N=1000)

# model
activation = lambda x: jnp.tanh(x)
layer_sizes = [3, 64, 1]
params = mlp.init_params(layer_sizes, random.PRNGKey(seed))
unnormalized_model = mlp.mlp(activation)


def model(params, x):
    # normalize with non-trainable layer
    y = 2 * (x - minimum) / (maximum - minimum) - 1
    return unnormalized_model(params, y)


v_model = vmap(model, (None, 0))


# diffusivity
def model_d(params_d, x):
    w, b = params_d[0]
    d = jax.nn.sigmoid(w + 0.0 * b) + jnp.array([0.000001])
    return jnp.reshape(d, ())


w = jnp.array([[-5.0]])
b = jnp.array([0.0])
params_d = [(w, b)]

# differential operators
dt = lambda g: del_i(g, 0)
ddx_1 = lambda g: del_i(del_i(g, 1), 1)
ddx_2 = lambda g: del_i(del_i(g, 2), 2)


def heat_operator(u, params_d):
    return lambda tx: dt(u)(tx) - model_d(params_d, jnp.array([0.0])) * (
        ddx_1(u)(tx) + ddx_2(u)(tx)
    )


# trick to get the signature (params, params_d, v_x) -> v_residual(params, params_d, v_x)
_residual = lambda params, params_d: heat_operator(lambda x: model(params, x), params_d)
residual = lambda params, params_d, x: _residual(params, params_d)(x)
v_residual = jit(vmap(residual, (None, None, 0)))


# loss terms
@jit
def loss_interior(params, params_d):
    return interior_integrator(lambda x: v_residual(params, params_d, x) ** 2)


@jit
def loss_data(params):
    return data_integrator.data_loss(lambda x: v_model(params, x))


@jit
def loss(params, params_d):
    return loss_interior(params, params_d) + loss_data(params)


# learning rate schedule
exponential_decay = optax.exponential_decay(
    init_value=0.001,
    transition_steps=10000,
    transition_begin=15000,
    decay_rate=0.5,
    end_value=0.0000001,
)

# optimizers
optimizer_u = optax.adam(learning_rate=exponential_decay)
opt_state_u = optimizer_u.init(params)

optimizer_d = optax.adam(learning_rate=exponential_decay)
opt_state_d = optimizer_d.init(params_d)

# Training Loop
for iteration in range(100000):
    grad_u = grad(loss, 0)(params, params_d)
    grad_d = grad(loss, 1)(params, params_d)

    updates_u, opt_state_u = optimizer_u.update(grad_u, opt_state_u)
    params = optax.apply_updates(params, updates_u)

    updates_d, opt_state_d = optimizer_d.update(grad_d, opt_state_d)
    params_d = optax.apply_updates(params_d, updates_d)

    if iteration % 1000 == 0:
        # update PDE residual points according to high PDE residual
        interior_integrator.update(lambda tx: v_residual(params, params_d, tx))

        # update data points randomly
        data_integrator.new_rand_points()

    if iteration % 1000 == 0:
        print(
            f"Adam Iteration: {iteration} with loss: "
            f"{loss(params, params_d)} "
            f"Diffusivity: {model_d(params_d, jnp.array([0.]))}"
        )
