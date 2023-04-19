import nibabel
import pathlib
import numpy as np
import itertools
import os
from tracerdiffusion.data import MRI_Data
import matplotlib.pyplot as plt
import SVMTK as svmtk
import shutil
from tracerdiffusion.utils import find_interior_boundary
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--concfolder", type=str, default=20, help="Resolution of the volume mesh for the ROI")
    parser.add_argument("--roi", type=str, default=20, help="Resolution of the volume mesh for the ROI")
    parserargs = vars(parser.parse_args())

    # parserargs["concfolder"] "/home/basti/Dropbox (UiO)/brainbook2data/freesurfer/CONCENTRATIONS/"
    concfolder = parserargs["concfolder"]
    concfolder = pathlib.Path(concfolder)

    os.chdir(concfolder)

    mris = MRI_Data(datapath="./")

    roi = roi.astype(bool)

    for key, filename in mris.time_filename_mapping.items():

        conc = nibabel.load(filename=filename).get_fdata()[roi].sum()

        plt.plot(float(key) / 3600, conc, linewidth=0, marker="o", markersize=5)

    plt.show()