
# Representing images as FEniCS functions and vice versa


## Mapping images to FEniCS functions

Scalar fields (such as the computed concentrations) given as an MRI volume can be represented as FEniCS functions similar as presented for diffusion tensor images in [1].

For convenience, we provide this functionality in a Python function `read_image` in the script `mri2fenics.py` to perform this task.
While the actual implementation contains additional data filtering, a basic implementation for reading MRI to FEniCS functions would look like:
```
def read_image(filename,functionspace):
	mri_volume = nibabel.load(filename)
	voxeldata = mri_volume.get_fdata() 
	c_data = Function(functionspace) 
	vox2ras = mri_volume.header.get_vox2ras_tkr()
	ras2vox = numpy.linalg.inv(ras2vox) 
	xyz = functionspace.tabulate_dof_coordinates()
	ijk = apply_affine(ras2vox, xyz).T
	i, j, k = numpy.rint(ijk).astype("int")
	c_data.vector()[:] = voxeldata[i, j, k]
	return c_data
```
(The CSF tracer concentration can sometimes only be computed in specific regions of the brain defined by binary voxel masks. 
Since the FreeSurfer surfaces and hence the meshes have sub-voxel resolution, vertices of a mesh describing a brain region may correspond to voxels outside the binary mask describing the same anatomical region. 
For this case, our implementation of `read_image` accepts a binary mask as `.mgz` file via the optional argument `--mask`.
Node values outside the mask will then be replaced by the median voxel values in adjacent voxels inside the mask.)

The script can be called from the command line to convert an image:
```
python ./examples/mri2fenics.py \
--mesh ./data/freesurfer/meshes/lh.xml --data ./data/freesurfer/CONCENTRATIONS/26.05.mgz
```

By default, the script will store the FEniCS function in various formats to the same location as the data, but a different path can be specified with the optional argument `--outputname`.
For example, the file `./data/freesurfer/CONCENTRATIONS/26.05.pvd` can be used for visualization in paraview.

## Mapping FEniCS functions to images

For visualizations of simulation results, it can be useful to "voxelize" a FEniCS function, i.e., fill an empty MRI volume with the values of the FEniCS function at the voxel centers.
By default, the simulation code under `scripts/forward-model/diffusion.py` stores the simulated tracer concentration at every imaging time into a file `simulation.hdf`.
We provide a Python script `fenics2mri.py` which can be called as
```
python ./examples/fenics2mri.py \
--mesh ./data/freesurfer/meshes/lh.xml \
--image ./data/freesurfer/mri/aseg.mgz \
--hdf5_file ./simulation_outputs/simulation.hdf \
--hdf5_name simulation26 \
--output ./simulation_outputs/26h.mgz
```

to store the simulated tracer at 26 h after baseline as a MRI volume.
To load the FEM prediction from the `hdf` file, we use `--hdf5_name simulation26` to assess the FEM solution stored under the key `simulation26` by the script `diffusion.py`.
For `hdf` files, the Python package [h5py](https://docs.h5py.org/en/stable/quick.html) is very useful, see the script `./examples/fenics2mri.py` for an example.

This script iterates over all voxel indices with RAS coordinates that may be in the mesh, and evaluates the FEniCS function in these points.
The script hence requires a few minutes to run for full brain meshes.
To speed up this process, the script has the optional argument `--mask` to specify in which voxels to evaluate the FEniCS function.
In our case, with `--mask ./data/freesurfer/mri/parenchyma_mask.mgz` the script only evaluates the voxel that are inside the brain parenchyma.
This reduces the runtime of the script by 30 %, but the speed up is even more significant when looking at smaller subregions of the brain.
For further technical details we refer to the documentation of the method in `./examples/fenics2mri.py`, and note that we also provide a script `./examples/voxelize-mesh.py` which can be used to obtain a binary mask in `.mgz` format from volume meshes.


# References

[1] Mardal, Kent-André, et al. Mathematical modeling of the human brain: from magnetic resonance images to finite element simulation. Springer Nature, 2022.