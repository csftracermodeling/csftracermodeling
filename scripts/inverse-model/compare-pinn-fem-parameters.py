import argparse
import json
import pathlib

import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pinnfolder", required=True,
                        help="""Path to output folder storing the PINN results"""
                        )
    parser.add_argument("--femfolder", required=True,
                        help="""Path to output folder storing the FEM results"""
                        )
    

    parserargs = vars(parser.parse_args())

    pinnfolder = pathlib.Path(parserargs["pinnfolder"])
    femfolder= pathlib.Path(parserargs["femfolder"])

    with open(pinnfolder / 'hyperparameters.json') as data_file:    
        pinn_hyperparameters = json.load(data_file)


    print("*"*100)
    print("*"*100)

    for qty, label in zip(["D.txt", "r.txt"],["Final D (mm^2 / s):", "Final r (1 / s):"]):

        pinnhistory = np.genfromtxt(pinnfolder / qty, delimiter=",")

        if np.isnan(pinnhistory[-1]):
            pinnhistory = pinnhistory[:-1]

        assert np.sum(np.isnan(pinnhistory)) == 0

        femhistory = np.genfromtxt(femfolder / qty, delimiter=",")

        print(label)
        print("PINN", format(pinnhistory[-1], ".2e"))
        print("FEM ", format(femhistory[-1], ".2e"))
        print()

    print("*"*100)
    print("*"*100)
