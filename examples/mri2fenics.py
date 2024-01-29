import argparse
import pathlib

from dolfin import File, FunctionSpace, HDF5File, Mesh

from tracerdiffusion.data import read_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d", "--data", required=True, type=str, help="Path to image in mgz format."
    )
    parser.add_argument("-m", "--mesh", required=True, help="Path to mesh.")
    parser.add_argument(
        "--mask",
        default=None,
        type=str,
        help="Path to mask in mgz format. This is useful if mesh vertices are located in voxels where the MRI data is invalid",
    )
    parser.add_argument("--outputname", type=str, default=None)
    parser.add_argument("--functiondegree", type=int, default=1)
    parser.add_argument("--functionspace", type=str, default="Lagrange")

    parserargs = vars(parser.parse_args())

    meshfile = parserargs["mesh"]

    if meshfile.endswith(".xml"):
        brainmesh = Mesh(meshfile)
    else:
        brainmesh = Mesh()
        hdf = HDF5File(brainmesh.mpi_comm(), meshfile, "r")
        hdf.read(brainmesh, "/mesh", False)

    V = FunctionSpace(
        brainmesh, parserargs["functionspace"], parserargs["functiondegree"]
    )

    c_data_fenics = read_image(
        filename=parserargs["data"], functionspace=V, mask=parserargs["mask"]
    )

    if parserargs["outputname"] is None:
        outputname = parserargs["data"]
    else:
        outputname = parserargs["outputname"]

    outputname = pathlib.Path(outputname).absolute()

    File(str(outputname.with_suffix(".pvd"))) << c_data_fenics

    hdf5file = HDF5File(V.mesh().mpi_comm(), str(outputname.with_suffix(".hdf5")), "w")
    hdf5file.write(V.mesh(), "mesh")
    hdf5file.write(c_data_fenics, "c")
    hdf5file.close()

    print("*" * 80)
    print("Script done, to view the result in paraview run")
    print("paraview " + outputname + ".pvd")
    print("*" * 80)
