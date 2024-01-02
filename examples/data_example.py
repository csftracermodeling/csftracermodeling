import matplotlib.pyplot as plt
import nibabel
import numpy as np

from tracerdiffusion.data import Voxel_Data
from tracerdiffusion.utils import cut_to_box
import pathlib

mask = pathlib.Path("./roi12/parenchyma_mask_roi.mgz")
if not mask.is_file():
    raise RuntimeError(f"Could not find {str(mask)}")


datapath = pathlib.Path("./data/freesurfer/CONCENTRATIONS2/")
if not datapath.exists():
    raise RuntimeError(f"Could not find {str(datapath)}")

if mask.suffix == ".npy":
    mask = np.load(mask)
else:
    mask = nibabel.load(mask).get_fdata().astype(bool)

# Load the first image after baseline
time_idx = 1

# Only load images up to 24 hours after baseline:
Tmax = 24 * 6

data = Voxel_Data(datapath=datapath, mask=mask, pixelsizes=[1, 1, 1], Tmax=Tmax)


print("Bounds:", data.bounds())

# Sample randomly from voxels and times:
inputs, targets = data.sample(n=100)

# Sample at specific time only for plotting
inputs, targets = data.sample_image(n=4000, time_idx=time_idx)

# Done - I think this is all you need for the PINNs

##############################################################################################################################

# Below we plot a slice of the data:


if len(mask.shape) == 3:
    slice_idx = 133
    slice_ax = 2
    maskslice = np.take(mask, axis=slice_ax, indices=slice_idx)
    if maskslice.sum() == 0:
        print(
            "Warning: the slice you have chosen for plotting does not intersect with the ROI!"
        )
        print("--> Exiting script.")
        exit()


plt.figure()
plt.title(f"Samples from ROI at t={data.measurement_times()[time_idx]/3600:.2f} hours")
plt.scatter(inputs[:, 1], inputs[:, 2], marker="s", c=targets, s=42, vmin=0, vmax=0.1)


try:
    image = np.load(data.files[time_idx]) * mask
except ValueError:
    image = nibabel.load(data.files[time_idx]).get_fdata()


imageslice = np.take(image, axis=slice_ax, indices=slice_idx)

imageslice_roi = cut_to_box(imageslice, mask=maskslice, box_bounds=None)
imageslice_roi = np.rot90(imageslice_roi)

roislice = np.rot90(maskslice)

plt.figure()
plt.title(f"Image at t={data.measurement_times()[time_idx]/3600:.2f} hours")
plt.imshow(np.rot90(imageslice), vmin=0, vmax=0.1)


plt.figure()
plt.title("ROI")
plt.imshow(np.rot90(roislice), cmap="Reds")


plt.figure()
plt.title(f"Image cut to ROI at t={data.measurement_times()[time_idx]/3600:.2f} hours")
plt.imshow(imageslice_roi, vmin=0, vmax=0.1)

plt.show()
