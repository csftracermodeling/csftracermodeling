# import sys
# sys.path.insert(0, "/home/basti/2023-mri2fem-ii/code/inverseDiffusion/")

import matplotlib.pyplot as plt
import nibabel
import numpy as np

from tracerdiffusion.data import Voxel_Data
from tracerdiffusion.utils import cut_to_box
import pathlib

data_folder = pathlib.Path("./roi")
if not data_folder.exists():
    raise RuntimeError(f"Data-folder {str(data_folder)} does not exist")

mask = data_folder / "parenchyma_mask_roi.mgz"

datapath = pathlib.Path("./data/mridata3d/CONCENTRATIONS/")
if not datapath.exists():
    raise RuntimeError(f"{str(datapath)} does not exist")

if mask.suffix == ".npy":
    mask = np.load(mask)
else:
    mask = nibabel.load(mask).get_fdata().astype(bool)

# Load the first image after baseline
time_idx = 1

# Only load images up to 24 hours after baseline:
Tmax = 3600*24

data = Voxel_Data(datapath=datapath, mask=mask,
                  pixelsizes=[1, 1, 1], Tmax=Tmax)

############################################################################################################################
# Select a slice

if len(mask.shape) == 3:
    slice_idx = 133
    slice_ax = 2
    maskslice = np.take(mask, axis=slice_ax, indices=slice_idx)
    if maskslice.sum() == 0:
        print("Warning: the slice you have chosen for plotting does not intersect with the ROI!")
        print("--> Exiting script.")
        exit()
try:
    image = np.load(data.files[time_idx]) * mask

except ValueError:
    image = nibabel.load(data.files[time_idx]).get_fdata()

image *= mask

roislice = np.rot90(maskslice)

imageslice = np.take(image, axis=slice_ax, indices=slice_idx)
imageslice = np.rot90(imageslice)

imageslice = np.where(roislice, imageslice, np.nan)

try:
    imageslice = cut_to_box(image=imageslice, mask=roislice, box_bounds=None)

except ModuleNotFoundError:
    print("Some modules for plotting could not be imported, will not zoom to data")

############################################################################################################################
# Plot the image

plt.figure()
plt.title(
    f"Slice through data at t= {data.measurement_times()[time_idx]/3600:.2f} hours")
plt.imshow(imageslice, vmin=0, vmax=0.1)
plt.colorbar()

############################################################################################################################
# Evaluate NN at time t

t = data.measurement_times()[time_idx]

predimg = np.zeros_like(image)

xyz = data.voxel_center_coordinates - data.domain.dx

t_xyz = np.zeros((xyz.shape[0], xyz.shape[1] + 1))

t_xyz[:, 0] = t

t_xyz[:, 1:] = xyz

# # shape of x is (n, 4) where n is batch-size and 4 is (t,x,y,z)
# x = np.array([[0., 1., 2., 3.], [0., 1., 2., 3.]])
# print(jnp.shape(v_model(params, x)))
# print(v_model(params, x))


def NN(params, x):
    return np.sum(x, axis=1)


pred = NN(params=None, x=t_xyz)

predimg[mask] = pred


predslice = np.take(predimg, axis=slice_ax, indices=slice_idx)
predslice = np.rot90(predslice)

predslice = np.where(roislice, predslice, np.nan)

try:
    predslice = cut_to_box(image=predslice, mask=roislice, box_bounds=None)

except ModuleNotFoundError:
    print("Some modules for plotting could not be imported, will not zoom to data")


############################################################################################################################
# Plot the Network prediction

plt.figure()
plt.title(f"NN output at t={(t / 3600):.2f} hours")
plt.imshow(predslice, vmin=0, vmax=0.1)
plt.colorbar()
plt.show()
