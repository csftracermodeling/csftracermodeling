import argparse
import os
import numpy as np
import pathlib
import nibabel

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outfolder", required=True,
                        help="Path to hdf folder storing the simulation results. " \
                        + "Assumes that the folders are named like outfolder/3600_12/ for timestep 3600 s and mesh resolution 12" \
                        + ". Further, it assumes that the PINN results are in folders like for example outfolder/pinn1e2/ for pde weight 1e2 "
                        )
    parser.add_argument("--datafolder", default="./data/freesurfer/CONCENTRATIONS/",
                        help="""Path to hdf folder storing the input data, needed to convert to mgz"""
                        )
    parser.add_argument("--mask", default="./roi12/parenchyma_mask_roi.mgz",
                    help="""Path to mask mgz file in which the error will be computed"""
                    )


    parserargs = vars(parser.parse_args())

    outfolder = pathlib.Path(parserargs["outfolder"])
    datafolder = pathlib.Path(parserargs["datafolder"])

    mask = nibabel.load(parserargs["mask"]).get_fdata().astype(int)

    data = {}


    for file in os.listdir(datafolder):

        if not str(file).endswith("mgz"):
            continue

        print(file)

        conc = nibabel.load(datafolder / file).get_fdata()

        conc = mask * conc

        t = int(np.round(float(file.replace(".mgz", ""))))

        if t > 55:
            continue

        data[t] = conc



    for subfolder in sorted(os.listdir(outfolder), key=lambda x: "pinn" in x):

        print(outfolder, subfolder)

        try:
            D = np.genfromtxt(outfolder / subfolder / "D.txt", delimiter=",")[-1]
            if np.isnan(D):
                D = np.genfromtxt(outfolder / subfolder / "D.txt", delimiter=",")[-2]


            # pinn code works with time units of hours
            if "pinn" in str(subfolder):
                D = D / 3600


            D *= 1e4

            r = np.genfromtxt(outfolder / subfolder / "r.txt", delimiter=",")[-1]
            if np.isnan(r):
                r = np.genfromtxt(outfolder / subfolder / "r.txt", delimiter=",")[-2]

            # pinn code works with time units of hours
            if "pinn" in str(subfolder):
                r = r / 3600

            r *= 1e6


        except (FileNotFoundError, IndexError):
            print(outfolder, subfolder, "not done?")
            print("continue")
            continue

        #### Load data and predictions

        error = 0
        norm = 0

        for key, conc_mri in data.items():
            
            conc_pred = None
            
            # Depending on the time step size, the predictions are stored at +- 1 h compared to imaging time (round-off)
            # Hence we check if an prediction image is available one hour before or after the mri
            for dt in [0, -1, 1]:
                try:
                    conc_pred = nibabel.load(outfolder / subfolder / (str(key + dt) + "h.mgz")).get_fdata()

                    conc_pred = mask * conc_pred
                    
                    break
                
                except FileNotFoundError:
                    continue

            if conc_pred is None:

                print(outfolder, subfolder, "has no images? Either not done or you have to run postprocess script")
                print("continue")
                continue

            # L2 errors:
            # TODO
            e = np.nansum((conc_mri-conc_pred)) ** 2
            n = np.nansum((conc_mri)) ** 2
            
            error += e
            norm += n
        l2_error = (error / norm) ** (1 / 2)
        
        print(format(D, ".2f"), "(1e4 mm^2 / s)", format(r, ".2f"), "(1e-6 1 / s)", "L2 error", format(l2_error, ".3e"))