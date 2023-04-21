"""Create a voxel mask from FEniCS mesh
"""
import argparse
from dolfin import Mesh, HDF5File, FunctionSpace, Function
import nibabel
import pathlib
from tracerdiffusion.utils import function_to_image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True,
                        type=str, help="FEniCS mesh file")
    parser.add_argument("--image", required=True, type=str,
                        help="MRI to get the vox2ras matrices from")
    parser.add_argument("--output", required=True, type=str,
                        help="Where to store the mask. Example: ./meshmask.mgz")
    parserargs = vars(parser.parse_args())

    # Load data
    nii_img = nibabel.load(parserargs["image"])
    mesh_path = pathlib.Path(parserargs["mesh"])
    if not mesh_path.is_file():
        raise RuntimeError(f"Could not find mesh file at: {mesh_path}")

    if mesh_path.suffix == ".xml":
        brainmesh = Mesh(str(parserargs["mesh"]))
    else:
        brainmesh = Mesh()
        hdf = HDF5File(brainmesh.mpi_comm(), str(mesh_path), "r")
        hdf.read(brainmesh, "/mesh", False)
        hdf.close()

    # Setup function
    V = FunctionSpace(brainmesh, "DG", 0)
    f = Function(V)

    f.vector()[:] = 1

    output_volume, output_arry = function_to_image(
        function=f, template_image=nii_img, extrapolation_value=0)

    nibabel.save(output_volume, parserargs["output"])

    print("*" * 80)
    print("Script done, to view the result in freeview run")
    print("freeview " + parserargs["output"])
    print("*" * 80)
