import argparse

import dolfin
import h5py
import nibabel

from tracerdiffusion.utils import function_to_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, type=str, help="FEniCS mesh file")
    parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="MRI file to get transformation matrix from",
    )
    parser.add_argument(
        "--hdf5_file", required=True, type=str, help="File storing the FEniCS function"
    )
    parser.add_argument(
        "--hdf5_name",
        required=True,
        type=str,
        help="Name of function inside the HDF5 file",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="MRI file to save the function to (e.g. shear_modulus.nii)",
    )
    parser.add_argument("--function_space", type=str, default="Lagrange")
    parser.add_argument("--function_degree", type=int, default=1)
    parser.add_argument("--extrapolation_value", type=float, default=float("nan"))
    parser.add_argument(
        "--mask", type=str, help="Mask used to specify which image voxels to evaluate"
    )
    parser.add_argument(
        "--skip_value",
        type=float,
        help="Voxel value indicating that a voxel should be skipped in the mask. If unspecified, it's the same as the extrapolation value.",
    )

    parserargs = vars(parser.parse_args())
    hdf5_name = parserargs["hdf5_name"]

    f = h5py.File(parserargs["hdf5_file"])
    hdf_keys = list(f.keys())

    print("Keys in hdf file:")
    for hdfkey in hdf_keys:
        print("--", hdfkey)

    assert (
        hdf5_name in hdf_keys
    ), "only the functions listed above are stored in the hdf file"

    # Load data
    nii_img = nibabel.load(parserargs["image"])

    if parserargs["mesh"].endswith(".xml"):
        brainmesh = dolfin.Mesh(parserargs["mesh"])
    else:
        brainmesh = dolfin.Mesh()
        hdf = dolfin.HDF5File(brainmesh.mpi_comm(), parserargs["mesh"], "r")
        hdf.read(brainmesh, "/mesh", False)
        hdf.close()

    # Setup function
    V = dolfin.FunctionSpace(
        brainmesh, parserargs["function_space"], parserargs["function_degree"]
    )
    f = dolfin.Function(V)
    hdf5 = dolfin.HDF5File(brainmesh.mpi_comm(), parserargs["hdf5_file"], "r")

    if not hdf5_name.startswith("/"):
        hdf5_name = "/" + hdf5_name
    hdf5.read(f, hdf5_name)

    tracer_in_domain = dolfin.assemble(f * dolfin.dx(domain=brainmesh))
    print(f"Tracer in domain {tracer_in_domain * 1e-6:.4e} mmol")

    output_volume, output_arry = function_to_image(
        function=f,
        template_image=nii_img,
        extrapolation_value=parserargs["extrapolation_value"],
        mask=parserargs["mask"],
    )

    nibabel.save(output_volume, parserargs["output"])

    print("*" * 80)
    print("Script done, to view the result in freeview run")
    print("freeview " + parserargs["output"])
    print("*" * 80)
