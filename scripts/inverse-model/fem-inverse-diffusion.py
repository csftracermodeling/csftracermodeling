from fenics import *

# Setting config.inverse = True will also import dolfin-adjoint in model.py.
# This is necessary since we want to use model.py for both forward and inverse simulations,
# and the forward simulations are faster if dolfin-adjoint does not keep track of the results.
# Hence, config.inverse = False by default such that the forward model can be run as efficiently as possible
import tracerdiffusion.config as config
config.inverse = True

from tracerdiffusion.model import Model

from fenics_adjoint import *
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy

import pathlib
import os
import shutil
import numpy
import argparse
from tracerdiffusion.data import FEniCS_Data
import json
import nibabel







def iter_cb(params):

    D_during_optim.append(params[0])

    D_file.write("," + str(params[0] * diffusion_model.scale_diffusion_gad))


    rf_j_d_numpy = ReducedFunctionalNumPy(Jhat)
    j_d_i = rf_j_d_numpy(params)

    J_file.write("," + str(j_d_i))

    if len(params) == 1:
        
        print("Alpha=", format(params[0], ".2e"))
    else:
        print("Alpha=", format(params[0], ".2e"), "r=", format(params[-1], ".2e"))
        r_during_optim.append(params[1])
        
        r_file.write("," + str(params[1]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data/mridata3d/CONCENTRATIONS/",
                        help="""Path to folder containing concentrations in mgz format. Assuming that imaging time is contained as YYYYMMDD_HHMMSS' in filename."""
                        )
    parser.add_argument("--mesh", default="./roi12/parenchyma_mask_roi12.xml", help="path to mesh as xml file.")
    parser.add_argument("--mask", default="./roi12/parenchyma_mask_roi.mgz",
                    help="path to mask from which mesh was made.")
    
    parser.add_argument("--outfolder", default="./simulation_outputs/")
    parser.add_argument("--lbfgs_iters", default=42)
    parser.add_argument("--dt", default=3600, type=float, help="timestep size")
    parser.add_argument("--taylortest", default=False, action="store_true", help="Run Taylor test to check if gradients are correct.")

    
    parserargs = vars(parser.parse_args())


    outfolder = pathlib.Path(parserargs["outfolder"])
    if outfolder.is_dir():
        print("Deleting existing outputfolder", outfolder)
        shutil.rmtree(outfolder)

    os.makedirs(outfolder, exist_ok=True)

    datapath = pathlib.Path(parserargs["data"])
    
    meshpath =  parserargs["mesh"]

    assert os.path.isfile(meshpath)

    brainmesh = Mesh(meshpath)

    print("Some info on your mesh:")
    print("(hmin, hmax) = (", brainmesh.hmin(), brainmesh.hmax() ,")")
    print("Number of cells =", format(brainmesh.cells().shape[0], ".1e"))
    print("Number of vertices =", format(brainmesh.coordinates().shape[0], ".1e"))
    print("MeshQuality.radius_ratio_min_max=", MeshQuality.radius_ratio_min_max(brainmesh))

    # exit()

    assert min(MeshQuality.radius_ratio_min_max(brainmesh)) > 1e-6, "Mesh contains degenerated cells"

    V = FunctionSpace(brainmesh, "CG", 1)

    # md = (0.00092582074369026 + 0.000867643624860488) / 2

    # mean_water_diffusivity = Constant(md)

    # simulate for up to 4 days after first image
    tmax = 3 * 24

    d_init = 1e-3
    mean_water_diffusivity = Constant(d_init)

    
    if V.mesh().topology().dim() == 3:
        slice_params = {}
    else:
        # Load slice params (slice normal and offset) to load MRI data to slice mesh
        with open(pathlib.Path(meshpath).parent / 'sliceparams.json') as data_file:    
            slice_params = json.load(data_file)

    if parserargs["mask"] is not None:
        mask = nibabel.load(parserargs["mask"]).get_fdata()
    else:
        mask = None

    mris = FEniCS_Data(function_space=V, datapath=datapath, Tmax=tmax, mask=mask)

    mris.dump_pvd(vtkpath=str(outfolder / "data.pvd"))

    # alpha = Constant(1)
    
    r_init = 1e-5
    reaction_rate = Constant(r_init)


    diffusion_model = Model(dt=parserargs["dt"], V=V, mris=mris, outfolder=outfolder)

    diffusion_model.forward(water_diffusivity=mean_water_diffusivity, r=reaction_rate, taylortest=parserargs["taylortest"])

    L2_mismatch = diffusion_model.return_value()

    initial_missmatch = float(L2_mismatch)
    print("Simulation done, mismatch=", format(L2_mismatch, ".2f"))
    diffusion_model.save_predictions(name="init")

    ctrls = [Control(mean_water_diffusivity)]
    bounds = [[0], [1e-2]]
    c0 = [mean_water_diffusivity]

    ctrls.append(Control(reaction_rate))
    bounds = [[0, 0], [1e-2, 1e-5]]
    c0 = [mean_water_diffusivity, reaction_rate]

    print("Creating ReducedFunctional")
    Jhat = ReducedFunctional(L2_mismatch, ctrls)

    jhat0 = Jhat(c0)


    J_file = open(outfolder / 'J.txt', 'w')
    J_file.write(str(jhat0))
    J_file.close()

    J_file = open(outfolder / 'J.txt', 'a')

    print("jhat0", jhat0)

    assert numpy.allclose(initial_missmatch, jhat0)

    print("Evaluating reduced functional done")


    if parserargs["taylortest"]:
        conv_rate = taylor_test(J=Jhat, m=[Constant(1e-2), Constant(1e-5)], h=[Constant(0), Constant(1e-5)])
        print(conv_rate)
        conv_rate = taylor_test(J=Jhat, m=[Constant(1e-2), Constant(1e-5)], h=[Constant(1e-2), Constant(0)])
        print(conv_rate)
        print("convergence test done, exiting")

        exit()

    D_during_optim = []

    D_file = open(outfolder / 'D.txt', 'w')
    D_file.write(str(d_init * diffusion_model.scale_diffusion_gad))
    D_file.close()

    D_file = open(outfolder / 'D.txt', 'a')


    r_during_optim = []

    r_file = open(outfolder / 'r.txt', 'w')
    r_file.write(str(r_init))
    r_file.close()

    r_file = open(outfolder / 'r.txt', 'a')
    

    opt_ctrls = minimize(Jhat, method="L-BFGS-B", callback=iter_cb, bounds=bounds,
                            options={"disp": True, "maxiter": int(parserargs["lbfgs_iters"])})
    # "ftol": 1e-12, "gtol": 1e-6

    if not isinstance(opt_ctrls, list):
        opt_ctrls = [opt_ctrls]

    vals = numpy.zeros(1)
    vals[0] = assemble(opt_ctrls[0]*dx(domain=brainmesh))/assemble(1*dx(domain=brainmesh))
    print("After optim. D (water)      =", format(vals[0], ".2e"))
    print("After optim. D (gadobutrol) =", format(vals[0] * diffusion_model.scale_diffusion_gad, ".2e"))

    final_D = Constant(D_during_optim[-1])
    final_r = None

    print("final_D", D_during_optim[-1])




        
    vals = numpy.zeros(1)
    vals[0] = assemble(opt_ctrls[1]*dx(domain=brainmesh))/assemble(1*dx(domain=brainmesh))
    print("After optim. r =", format(vals[0], ".2e"))

    final_r = r_during_optim[-1]

    print("final_r", final_r)

    # diffusion_model = Model(dt=parserargs["dt"], V=V, mris=mris, outfolder=outfolder, mean_water_diffusivity=final_D)
    L2_mismatch = diffusion_model.forward(water_diffusivity=final_D, r=final_r, taylortest=parserargs["taylortest"])

    print("Simulation done, mismatch=", format(L2_mismatch, ".2f"))

    # assert numpy.allclose(mismatch, J)

    print("Final / Initial Mismatch =", format(float(L2_mismatch) / initial_missmatch, ".2f"))

    if float(L2_mismatch) / initial_missmatch > 1:
        raise ValueError("Optimizaiton did not improve ?!")

    diffusion_model.save_predictions(name="final")