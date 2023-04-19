import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import optax
import argparse
import os
import nibabel
import numpy as np
import pathlib
import json
import matplotlib.pyplot as plt
from tracerdiffusion.utils import cut_to_box
from tracerdiffusion.data import Voxel_Data
import tracerdiffusion.jax_example.slim_natgrad.mlp as mlp

import pickle


# from IPython import embed
def load_nn(parserargs: dict, hyperparameters: dict):
    
    pkl_file = open(parserargs["outfolder"] / "nn_params.pkl", 'rb')
    nn_params = pickle.load(pkl_file)
    pkl_file.close()


    minimum = jnp.array(hyperparameters["minimum"])
    maximum = jnp.array(hyperparameters["maximum"])


    # model
    activation = lambda x : jnp.tanh(x)
    layer_sizes = hyperparameters["layer_sizes"]
    # params = mlp.init_params(layer_sizes, random.PRNGKey(hyperparameters["seed"]))
    unnormalized_model = mlp.mlp(activation)

    def model(params, x):
        # normalize with non-trainable layer
        x = 2 * (x - minimum) / (maximum - minimum) - 1
        return unnormalized_model(params, x)

    v_model = vmap(model, (None, 0))

    def nn(x):
        return v_model(nn_params, x)
    
    return nn

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outfolder", required=True,
                        help="""Path to output folder storing the PINN results"""
                        )

    parser.add_argument("--mask", default="./roi/parenchyma_mask_roi.mgz",
                    help="path to mask from which mesh was made.")
    

    parserargs = vars(parser.parse_args())

    parserargs["outfolder"] = pathlib.Path(parserargs["outfolder"])

    with open(parserargs["outfolder"] / 'hyperparameters.json') as data_file:    
        hyperparameters = json.load(data_file)


    epochs = np.genfromtxt(parserargs["outfolder"] / "Epoch.txt", delimiter=",")

    files = ["J.txt",  
             "D.txt",
             "r.txt"
             ]
    labels = [r"PINN Loss $\mathcal{J} = \mathcal{J}_d + \mathcal{J}_p$",
              "diffusion coefficient $D$ ($10^{-4}$ mm$^2$/s)", 
              "reaction rate $r$ ($10^{-5}$/s)"
              ]
    scales = [1, 1e4, 1e5]

    plotfuns = [plt.semilogy, plt.plot]

    for file, label, scale, plotfun in zip(files, labels, scales, plotfuns):
        
        plt.figure()
        history = np.genfromtxt(parserargs["outfolder"] / file, delimiter=",")

        plotfun(epochs, history * scale)
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.tight_layout()

    files = ["J.txt",  
             "J_d.txt", "J_pde.txt",

             ]
    labels = [r"$\mathcal{J} = \mathcal{J}_d + w_{pde}\mathcal{J}_{pde}$",
              r"$\mathcal{J}_d$",
              r"$\mathcal{J}_{pde}$",
              ]
    
    scales = [1, 1, 1, 1e4, 1e5]


    plt.figure()
    for file, label, scale in zip(files, labels, scales):
        
        
        history = np.genfromtxt(parserargs["outfolder"] / file, delimiter=",")

        plt.semilogy(epochs, history * scale, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Loss during training")
        plt.legend()
        plt.tight_layout()


    ############################################################################################################################
    # Plotting
    # Select a slice
    # Load the first image after baseline
    time_idx = 2
    slice_idx = 133
    slice_ax = 2

    mask = hyperparameters["mask"]

    if mask.endswith("npy"):
        mask = np.load(mask)
    else:
        mask = nibabel.load(mask).get_fdata().astype(bool)


    data = Voxel_Data(datapath=hyperparameters["datapath"], mask=mask, pixelsizes=[1,1, 1], Tmax=hyperparameters["Tmax"])

    nn = load_nn(parserargs=parserargs, hyperparameters=hyperparameters)

    # plt.close("all")

    if len(mask.shape) == 3:

        maskslice = np.take(mask, axis=slice_ax, indices=slice_idx)
        if maskslice.sum() == 0:
            print("Warning: the slice you have chosen for plotting does not intersect with the ROI!")
            print("--> Exiting script.")
            exit()
    try:
        image = np.load(data.files[time_idx]) * mask
        
    except ValueError:
        image = nibabel.load(data.files[time_idx])
        affine = image.affine
        image = image.get_fdata()

    image *= mask

    roislice = np.rot90(maskslice)

    imageslice = np.take(image, axis=slice_ax, indices=slice_idx)
    imageslice = np.rot90(imageslice)

    imageslice = np.where(roislice, imageslice, np.nan)

    try:
        imageslice = cut_to_box(image=imageslice, mask=roislice, box_bounds=None)

    except ModuleNotFoundError:
        print("Some modulels for plotting could not be imported, will not zoom to data")

    ############################################################################################################################
    # Plot the image

    vmin, vmax = None, 0.25

    plt.figure()
    plt.title("Slice through data" + " at t=" + format(data.measurement_times()[time_idx]/3600, ".2f") + " hours")
    plt.imshow(imageslice, vmin=vmin, vmax=vmax)
    plt.colorbar()

    t = data.measurement_times()[time_idx]
    predimg = np.zeros_like(image)

    xyz = data.voxel_center_coordinates - data.domain.dx

    t_xyz = np.zeros((xyz.shape[0], xyz.shape[1] + 1))
            
    t_xyz[:, 0] = t

    t_xyz[:, 1:] = xyz

    pred = nn(t_xyz)

    predimg[mask] = pred


    predslice = np.take(predimg, axis=slice_ax, indices=slice_idx)
    predslice = np.rot90(predslice)

    predslice = np.where(roislice, predslice, np.nan)

    try:
        predslice = cut_to_box(image=predslice, mask=roislice, box_bounds=None)

    except ModuleNotFoundError:
        print("Some modulels for plotting could not be imported, will not zoom to data")


    ############################################################################################################################
    # Plot the Network prediction

    plt.figure()
    plt.title("NN output" + " at t=" + format(t / 3600, ".2f") + " hours")
    plt.imshow(predslice, vmin=vmin, vmax=vmax)
    plt.colorbar()



    ############################################################################################################################
    # Store all predictions as MRI

    for tkey, filename in data.time_filename_mapping.items():


        t_xyz[:, 0] = float(tkey)

        pred = nn(t_xyz)

        predimg[mask] = pred

        nn_nii = nibabel.Nifti1Image(predimg, affine)

        filename = str(parserargs["outfolder"] / ( format(float(tkey) / 3600, ".0f") + "h.mgz"))

        nibabel.save(nn_nii, filename)

        print("Stored", filename)

    # exit()
    plt.show()