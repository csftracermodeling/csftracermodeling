"""Create a voxel mask from FEniCS mesh
"""
import argparse
import dolfin
import nibabel
from tracerdiffusion.utils import function_to_image

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, type=str, help="FEniCS mesh file")
    parser.add_argument("--image", required=True, type=str, help="MRI to get the vox2ras matrices from")
    parser.add_argument("--output", required=True, type=str, help="Where to store the mask. Example: ./meshmask.mgz")
    parserargs = vars(parser.parse_args())

    # Load data
    nii_img = nibabel.load(parserargs["image"])

    if parserargs["mesh"].endswith(".xml"):
        brainmesh = dolfin.Mesh(parserargs["mesh"])
    else:
        brainmesh= dolfin.Mesh()
        hdf = dolfin.HDF5File(brainmesh.mpi_comm(), parserargs["mesh"], "r")
        hdf.read(brainmesh, "/mesh", False)
        hdf.close()

    # Setup function
    V = dolfin.FunctionSpace(brainmesh, "DG", 0) 
    f = dolfin.Function(V)

    f.vector()[:] = 1

    output_volume, output_arry = function_to_image(function=f, template_image=nii_img, extrapolation_value=0)


    nibabel.save(output_volume, parserargs["output"])


    print("*"* 80)
    print("Script done, to view the result in freeview run")
    print("freeview " + parserargs["output"])
    print("*"* 80)