import argparse
import os
import nibabel
import numpy as np
import pathlib
import json
import matplotlib.pyplot as plt
from tracerdiffusion.data import Voxel_Data
import pickle
from tracerdiffusion.utils import cut_to_box


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imagefolder", required=True,
                        help="""Path to folder containing images"""
                        )
    parser.add_argument("-m", "--mask", required=True,
                        help="""Path to output folder storing the FEM results"""
                        )
    parser.add_argument("-o", "--outputfolder",
                        help="""Path to folder where the masked images are stored. Will be imagefolder_MASKED by default"""
                        )
    
    parser.add_argument("--overwrite", action="store_true", default=False, 
                        help="Overwrite --outputfolder")
    

    parserargs = vars(parser.parse_args())

    imagefolder = pathlib.Path(parserargs["imagefolder"])

    if parserargs["outputfolder"] is None:
        outputfolder = imagefolder.parent / (imagefolder.name + "_MASKED")
    else:
        outputfolder = pathlib.Path(parserargs["outputfolder"])

    if os.path.isdir(outputfolder) and len(os.listdir(outputfolder)) > 1:
        print("--outputfolder", str(outputfolder), "exists, exiting script. Use --overwrite to re-do the masking.")

        if not parserargs["overwrite"]:
           exit()

    os.makedirs(outputfolder)

    mask = pathlib.Path(parserargs["mask"])

    mask = nibabel.load(mask)

    affine = mask.affine

    mask = mask.get_fdata()

    for imagefile in os.listdir(imagefolder):

        if not str(imagefile).endswith(".mgz"):
            continue

        imagedata = nibabel.load(imagefolder / imagefile).get_fdata()

        outputfile = str(outputfolder / imagefile)

        print("Storing", outputfile)

        nibabel.save(nibabel.Nifti1Image(imagedata * mask, affine), outputfile)