import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import optax
import nibabel
import numpy as np
import pickle
import tracerdiffusion.jax_example.slim_natgrad.mlp as mlp
from tracerdiffusion.jax_example.slim_natgrad.integrators import EvolutionaryIntegrator
from tracerdiffusion.jax_example.slim_natgrad.derivatives import del_i
from tracerdiffusion.jax_example.slim_natgrad.domains import DummyDomain, TimeDomain
from tracerdiffusion.jax_example.slim_natgrad.data import DataSet, DataIntegrator

# try:
#     from domains import ImageDomain
#     from data import Voxel_Data
# except ModuleNotFoundError:
from tracerdiffusion.domains import ImageDomain
from tracerdiffusion.data import Voxel_Data
import pathlib, shutil, os, json

jax.config.update("jax_enable_x64", True)


# The following block of code should eventually be removed and happen
# automatically through the correct merge of DummyDomain and ImageDomain
#------------------------------------------------------------------------#
domainmask = "./roi12/parenchyma_mask_roi.mgz"
boundarymask = "./roi12/parenchyma_mask_boundary.mgz"
datapath="./data/mridata3d/CONCENTRATIONS/"
mask = "./roi12/parenchyma_mask_roi.mgz"

outputpath = "./pinn_outputs/"

hyperparameters = {
    "pdeweight": 1e8,
    "epochs": 2e5,
    "pdepoints": 1e4,
    "datapoints": 1e3,
    "domainmask": domainmask, 
    "boundarymask": boundarymask,
    "datapath": datapath,
    "mask": mask,
    "Tmax": 24 * 3,
    "layer_sizes": [4, 64, 64, 1],
    "seed": 0,
    "pixelsizes": [1, 1, 1], # in mm
}

outfolder = pathlib.Path(outputpath)
if outfolder.is_dir():
    print("Deleting existing outputfolder", outfolder)
    shutil.rmtree(outfolder)

os.makedirs(outfolder, exist_ok=True)

with open(outfolder / 'hyperparameters.json', 'w') as outfile:
    json.dump(hyperparameters, outfile, sort_keys=True, indent=4)




if domainmask.endswith("npy"):
    mask = np.load(domainmask)
    boundarymask = np.load(boundarymask)
else:
    mask = nibabel.load(domainmask).get_fdata().astype(bool)
    boundarymask = nibabel.load(boundarymask).get_fdata().astype(bool)



# Sample from the domain
potato = ImageDomain(mask=mask, pixelsizes=hyperparameters["pixelsizes"])
pdepoints = potato.sample(n=100000)
#--------------------------------------------------------------------------#


# Only load images up to 3 * 24 hours after baseline:


data = Voxel_Data(datapath=datapath, mask=mask, pixelsizes=hyperparameters["pixelsizes"], Tmax=hyperparameters["Tmax"])

# Sample randomly from voxels and times:
inputs, targets = data.sample(n=10000)
#---------------------------------------------------------------------------#


# spatial domain, no time coordinates
omega = DummyDomain(pdepoints)

# normalization constants
minimum, maximum = data.bounds()

hyperparameters["minimum"] = minimum.tolist()
hyperparameters["maximum"] = maximum.tolist()

with open(outfolder / 'hyperparameters.json', 'w') as outfile:
    json.dump(hyperparameters, outfile, sort_keys=True, indent=4)


# domains
interior = TimeDomain(minimum[0], maximum[0], omega)

# dataset
dataset = DataSet(inputs, targets)

# integrators
interior_integrator = EvolutionaryIntegrator(interior, key=random.PRNGKey(hyperparameters["seed"]), N=int(hyperparameters["pdepoints"]))
data_integrator = DataIntegrator(random.PRNGKey(0), dataset, N=int(hyperparameters["datapoints"]))

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = hyperparameters["layer_sizes"]
params = mlp.init_params(layer_sizes, random.PRNGKey(hyperparameters["seed"]))
unnormalized_model = mlp.mlp(activation)

def model(params, x):
    # normalize with non-trainable layer
    x = 2 * (x - minimum) / (maximum - minimum) - 1
    return unnormalized_model(params, x)

v_model = vmap(model, (None, 0))

# diffusivity
def model_d(params_d, x):
        w, b = params_d[0]
        #d = jnp.array(1e-4) * (jax.nn.sigmoid(w + 0. * b) + jnp.array([0.5]))
        d = jnp.array(1e-4) * (jnp.array([2.]) * jax.nn.sigmoid(w + 0. * b))
        return jnp.reshape(d, ())

w = jnp.array([[0.]])
b = jnp.array([0.])
params_d = [(w, b)]


# Add reaction term r*u to PDE loss
def model_r(params_r, x):
    w, b = params_r[0]
    r = jnp.array(1e-5) * (5. * jax.nn.sigmoid(w + 0. * b) + jnp.array([0.5]))
    return jnp.reshape(r, ())

wr = jnp.array([[0.5]])
br = jnp.array([0.])
params_r = [(wr, br)]

output = open(outfolder / 'nn_init_params.pkl', 'wb')
pickle.dump(params, output)
output.close()

output = open(outfolder / 'd_init_params.pkl', 'wb')
pickle.dump(params_d, output)
output.close()

output = open(outfolder / 'r_init_params.pkl', 'wb')
pickle.dump(params_r, output)
output.close()

# differential operators
dt    = lambda g: del_i(g, 0)
ddx_1 = lambda g: del_i(del_i(g, 1), 1)
ddx_2 = lambda g: del_i(del_i(g, 2), 2)
ddx_3 = lambda g: del_i(del_i(g, 3), 3)

def heat_operator(u, params_d, params_r):
    return lambda tx: dt(u)(tx) - model_d(params_d, jnp.array([0.])) * (ddx_1(u)(tx) + ddx_2(u)(tx) + ddx_3(u)(tx)) + model_r(params_r, jnp.array([0.])) * u(tx)

# trick to get the signature (params, params_d, v_x) -> v_residual(params, params_d, v_x)
_residual = lambda params, params_d, params_r: heat_operator(lambda x: model(params, x), params_d, params_r)
residual = lambda params, params_d, params_r, x: _residual(params, params_d, params_r)(x)
v_residual =  jit(vmap(residual, (None, None, None, 0)))

# loss terms
@jit
def loss_interior(params, params_d, params_r):
    return interior_integrator(lambda x: v_residual(params, params_d, params_r, x)**2)

@jit
def loss_data(params):
    return data_integrator.data_loss(
        lambda x: v_model(params, x)
    )

pde_weight = hyperparameters["pdeweight"]

@jit
def loss(params, params_d, params_r):
    return pde_weight * loss_interior(params, params_d, params_r) + loss_data(params)   

# learning rate schedule
exponential_decay = optax.exponential_decay(
    init_value=0.001, 
    transition_steps=10000,
    transition_begin=15000,
    decay_rate=0.5,
    end_value=0.0000001
)

# optimizers
optimizer_u = optax.adam(learning_rate=exponential_decay)
opt_state_u = optimizer_u.init(params)

optimizer_d = optax.adam(learning_rate=exponential_decay)
opt_state_d = optimizer_d.init(params_d)

optimizer_r = optax.adam(learning_rate=exponential_decay)
opt_state_r = optimizer_r.init(params_r)


# Training Loop
for iteration in range(int(hyperparameters["epochs"])):
    grad_u = grad(loss, 0)(params, params_d, params_r)
    grad_d = grad(loss, 1)(params, params_d, params_r)
    grad_r = grad(loss, 2)(params, params_d, params_r)
    
    updates_u, opt_state_u = optimizer_u.update(grad_u, opt_state_u)
    params = optax.apply_updates(params, updates_u)

    updates_d, opt_state_d = optimizer_d.update(grad_d, opt_state_d)
    params_d = optax.apply_updates(params_d, updates_d)

    updates_r, opt_state_r = optimizer_r.update(grad_r, opt_state_r)
    params_r = optax.apply_updates(params_r, updates_r)

    if iteration % 100 == 0:
        # update PDE residual points according to high PDE residual
        interior_integrator.update(lambda tx: v_residual(params, params_d, params_r, tx))
        
        # update data points randomly 
        data_integrator.new_rand_points()

    
    if iteration % 1e2 == 0:

        loss_value = loss(params, params_d, params_r)
        dataloss_value = loss_data(params)
        pdeloss_value = pde_weight * loss_interior(params, params_d, params_r)
        diffusion_coefficient = model_d(params_d, jnp.array([0.]))

        reaction_rate = model_r(params_r, jnp.array([0.]))

        J_file = open(outfolder / 'J.txt', 'a')
        J_d_file = open(outfolder / 'J_d.txt', 'a')
        J_pde_file = open(outfolder / 'J_pde.txt', 'a')
        D_file = open(outfolder / 'D.txt', 'a')
        r_file = open(outfolder / 'r.txt', 'a')
        E_file = open(outfolder / 'Epoch.txt', 'a')

        J_file.write(str(loss_value) + ",")
        J_d_file.write(str(dataloss_value) + ",")
        J_pde_file.write(str(pdeloss_value) + ",")
        D_file.write(str(diffusion_coefficient) + ",")
        r_file.write(str(reaction_rate) + ",")
        E_file.write(str(iteration) + ",")

        J_file.close()
        J_d_file.close()
        J_pde_file.close()
        D_file.close()
        r_file.close()
        E_file.close()

        print(
            f'Adam Iteration: {iteration} with loss: '
            f'{loss_value} '
            f'Loss Data {dataloss_value} '
            f'Loss PDE {pdeloss_value} '
            f'Diffusivity: {diffusion_coefficient}'
            f'Reaction: {reaction_rate}'
        )

        output = open(outfolder / 'nn_params.pkl', 'wb')
        pickle.dump(params, output)
        output.close()

        output = open(outfolder / 'd_params.pkl', 'wb')
        pickle.dump(params_d, output)
        output.close()

        output = open(outfolder / 'r_params.pkl', 'wb')
        pickle.dump(params_r, output)
        output.close()

# print("*"*100, v_model(params, x_test), "*"*100)



# x_test = jnp.zeros((1, 4))
# x_test.at[:].set(1)

# print("*"*100, v_model(params, x_test), "*"*100)
