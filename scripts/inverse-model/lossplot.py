import argparse
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 30})

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outfolder", required=True,
                        help="""Path to hdf folder storing the simulation results"""
                        )

    parser.add_argument("--mask", required=True,
                    help="path to mask from which mesh was made, e.g., ./roi12/parenchyma_mask_roi.mgz")
    

    parserargs = vars(parser.parse_args())

    parserargs["outfolder"] = pathlib.Path(parserargs["outfolder"])

    files = ["J.txt", "D.txt", "r.txt"]
    labels = ["$L^2$-error", "$D \,(10^{-4}$ mm$^2$/s)", "$r \, (10^{-6}$/s)"]
    scales = [1, 1e4, 1e6]


    fs = None # 30
    dpi = None # 300


    for file, label, scale in zip(files, labels, scales):
        


        plt.figure()
        history = np.genfromtxt(parserargs["outfolder"] / file, delimiter=",")
        if file == "D.txt":
            print("D final ", format(history[-1] * scale), "1e-4 mm^2/s")
            print("D final ", format(history[-1] * 3600), "mm^2/h")
        if file == "r.txt":
            print("r final ", format(history[-1] * scale), "1e-5 /s")
            print("r final ", format(history[-1] * 3600), "/h")

        plt.plot(np.array(range(history.size)), history * scale)
        plt.xlabel("Iteration", fontsize=fs)
        plt.ylabel(label, fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.tight_layout()
        plt.savefig(str(parserargs["outfolder"] / file).replace(".txt", ".png"), dpi=dpi)
    
    os.makedirs(parserargs["outfolder"], exist_ok=True)
