import argparse
import os
import nibabel
import numpy as np
import pathlib
import json
import matplotlib.pyplot as plt
from tracerdiffusion.data import Voxel_Data
import pickle
from tracerdiffusion.utils import cut_to_box


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pinnfolder", required=True,
                        help="""Path to output folder storing the PINN results"""
                        )
    parser.add_argument("--femfolder", required=True,
                        help="""Path to output folder storing the FEM results"""
                        )
    

    parserargs = vars(parser.parse_args())

    pinnfolder = pathlib.Path(parserargs["pinnfolder"])
    femfolder= pathlib.Path(parserargs["femfolder"])

    with open(pinnfolder / 'hyperparameters.json') as data_file:    
        pinn_hyperparameters = json.load(data_file)


    print("*"*100)
    print("*"*100)

    for qty, label in zip(["D.txt", "r.txt"],["Final D (mm^2 / s):", "Final r (1 / s):"]):

        pinnhistory = np.genfromtxt(pinnfolder / qty, delimiter=",")

        if np.isnan(pinnhistory[-1]):
            pinnhistory = pinnhistory[:-1]

        assert np.sum(np.isnan(pinnhistory)) == 0

        femhistory = np.genfromtxt(femfolder / qty, delimiter=",")

        print(label)
        print("PINN", format(pinnhistory[-1], ".2e"))
        print("FEM ", format(femhistory[-1], ".2e"))
        print()

    print("*"*100)
    print("*"*100)

    time_idx = 2
    slice_idx = 133
    slice_ax = 2

    pinn_images = [pathlib.Path(parserargs["pinnfolder"]) / x for x in os.listdir(pathlib.Path(parserargs["pinnfolder"])) if x.endswith(".mgz")]

    if len(pinn_images) == 0:
        raise ValueError("pinn-postprocess.py need to be run on ", parserargs["pinnfolder"], "first to create .mgz images")

    fem_images = [pathlib.Path(parserargs["femfolder"]) / x for x in os.listdir(pathlib.Path(parserargs["femfolder"])) if x.endswith(".mgz")]

    if len(fem_images) == 0:
        raise ValueError("fem-postprocess.py need to be run on ", parserargs["femfolder"], "first to create .mgz images")

    
    pinn_images = sorted(pinn_images, key=lambda x: float(x.name.replace("h.mgz", "")))        

    fem_images = sorted(fem_images, key=lambda x: float(x.name.replace("h.mgz", "")))
    

    ############################################################################################################################
    # Plotting
    # Select a slice
    # Load the first image after baseline
    time_idx = 2
    slice_idx = 133
    slice_ax = 2
    vmin, vmax = None, 0.15

    mask = pinn_hyperparameters["mask"]

    if mask.endswith("npy"):
        mask = np.load(mask)
    else:
        mask = nibabel.load(mask).get_fdata().astype(bool)

    data = Voxel_Data(datapath=pinn_hyperparameters["datapath"], mask=mask, pixelsizes=[1,1, 1], Tmax=pinn_hyperparameters["Tmax"], verbosity=0)


    if len(mask.shape) == 3:

        maskslice = np.take(mask, axis=slice_ax, indices=slice_idx)
        if maskslice.sum() == 0:
            print("Warning: the slice you have chosen for plotting does not intersect with the ROI!")
            print("--> Exiting script.")
            exit()



    mri = nibabel.load(data.files[time_idx])
    mri = mri.get_fdata()

    mri *= mask


    roislice = np.rot90(maskslice)

    ############################################################################################################################
    # Plot the images
    
    names = ["MRI", "FEM", "PINN"]
    images = [mri, fem_images[time_idx], pinn_images[time_idx]]

    for name, image in zip(names, images):

        if not isinstance(image, np.ndarray):
            print("Loading", image)

    for name, image in zip(names, images):

        if not isinstance(image, np.ndarray):
            image = nibabel.load(image).get_fdata()

        imageslice = np.take(image, axis=slice_ax, indices=slice_idx)
        imageslice = np.rot90(imageslice)

        imageslice = np.where(roislice, imageslice, np.nan)

        try:
            imageslice = cut_to_box(image=imageslice, mask=roislice, box_bounds=None)

        except ModuleNotFoundError:
            print("Some modulels for plotting could not be imported, will not zoom to data")


        if name != "MRI":

            print("l2 difference MRI-", name, format(np.nansum((mri[mask]- image[mask]) ** 2), ".2e") + " at t=" + format(data.measurement_times()[time_idx]/3600, ".0f") + " hours")

            if np.sum(np.isnan(image[mask])) > 0.05 * image[mask].size:
                # breakpoint()
                raise ValueError("More than 5 % nans in image, probably something wrong")
                # NOTE: Due to the fact that the mesh surface has sub-voxel resolution (the surface crosses voxels)
                # some of the (surface-) mesh coordinates will not be within the mask. 
                # in this case we can't evaluate the FEniCS function and set the voxel values to np.nan

        plt.figure()
        plt.title(name + " at t=" + format(data.measurement_times()[time_idx]/3600, ".0f") + " hours" + "\n" + "Slice along direction " + str(slice_ax) + " at voxel index " + str(slice_idx))
        plt.imshow(imageslice, vmin=vmin, vmax=vmax)
        plt.colorbar()

    plt.show()