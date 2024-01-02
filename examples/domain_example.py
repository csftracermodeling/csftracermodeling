import pathlib

import matplotlib.pyplot as plt
import nibabel
import numpy as np

from tracerdiffusion.domains import ImageDomain

data_folder = pathlib.Path("./roi12")
if not data_folder.exists():
    raise RuntimeError(f"Data-folder {str(data_folder)} does not exist")

domainmask = data_folder / "parenchyma_mask_roi.mgz"
boundarymask = data_folder / "parenchyma_mask_boundary.mgz"
datapath = data_folder / "freesurfer/CONCENTRATIONS/"

if domainmask.suffix == ".npy":
    mask = np.load(domainmask)
    boundarymask = np.load(boundarymask)
else:
    mask = nibabel.load(domainmask).get_fdata().astype(bool)
    boundarymask = nibabel.load(boundarymask).get_fdata().astype(bool)


pixelsizes = [1, 1, 1]  # in mm

# Sample from the domain
potato = ImageDomain(mask=mask, pixelsizes=pixelsizes)
samples = potato.sample(n=100)


print("Bounds:", potato.bounds())

# Sample from the boundary only
potato_boundary = ImageDomain(mask=boundarymask, pixelsizes=pixelsizes)
boundary_samples = potato_boundary.sample(n=100)


fig = plt.figure()

ax = fig.add_subplot(projection="3d")

ax.scatter(
    potato.voxel_center_coordinates[:, 0],
    potato.voxel_center_coordinates[:, 1],
    potato.voxel_center_coordinates[:, 2],
    label="voxel center coordinates",
    marker="o",
    facecolors="none",
    edgecolors="blue",
)
ax.scatter(
    samples[:, 0],
    samples[:, 1],
    samples[:, 2],
    marker="x",
    color="r",
    label="random interior samples",
)
ax.scatter(
    boundary_samples[:, 0],
    boundary_samples[:, 1],
    boundary_samples[:, 2],
    marker="x",
    color="g",
    label="boundary",
)
plt.legend()
plt.show()
