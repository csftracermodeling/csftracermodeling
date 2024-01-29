from fenics import *
import pathlib
import os
import numpy as np
import argparse

from tracerdiffusion.data import FEniCS_Data
from tracerdiffusion.model import Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        help="Path to folder containing concentrations in mgz format. Assuming that imaging time is contained as 'HH.MM.mgz' in filename.",
    )
    parser.add_argument(
        "--mesh",  # default="${WORKDIR}/meshes/lh.xml",
        help="path to mesh",
    )
    parser.add_argument("--outfolder", default="./simulation_outputs/")
    parserargs = vars(parser.parse_args())

    outfolder = pathlib.Path(parserargs["outfolder"])

    os.makedirs(outfolder, exist_ok=True)

    datapath = pathlib.Path(parserargs["data"])  # )

    meshpath = parserargs["mesh"]  # ""

    assert os.path.isfile(meshpath)

    if meshpath.endswith("xml"):
        brainmesh = Mesh(meshpath)

        dx_SD = None
    else:
        try:
            brainmesh = Mesh()
            hdf = HDF5File(brainmesh.mpi_comm(), meshpath, "r")
            hdf.read(brainmesh, "/mesh", False)
            subdomains = MeshFunction("size_t", brainmesh, brainmesh.topology().dim())
            hdf.read(subdomains, "/subdomains")

            # GRAY = 1. WHITE = 2. BRAIN STEM = 3.
            dx_SD = Measure("dx")(domain=brainmesh, subdomain_data=subdomains)
        except:
            print("No subdomains found")
            dx_SD = None

    V = FunctionSpace(brainmesh, "Lagrange", 1)
    mean_diffusivitySpace = FunctionSpace(brainmesh, "DG", 0)

    print(
        "Using D=1e-3 mm^2/s for water diffusion (cf. Valnes et al Scientific Reports 2020)"
    )
    mean_water_diffusivity = Constant(1e-3)

    # simulate for up to 3 days after first image
    tmax = 51

    mris = FEniCS_Data(datapath=datapath, function_space=V, Tmax=tmax)

    mris.dump_pvd(vtkpath=str(outfolder / "data.pvd"))

    diffusion_model = Model(
        dt=1800, V=V, mris=mris, dx_SD=dx_SD, outfolder=outfolder, verbosity=0
    )

    diffusion_model.forward(water_diffusivity=mean_water_diffusivity)

    mismatch = diffusion_model.L2_error

    print("Simulation done, mismatch=", format(mismatch, ".2f"))

    diffusion_model.save_predictions()
