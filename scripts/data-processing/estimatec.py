import os
import pathlib
import argparse
import numpy
import nibabel
import warnings

from tracerdiffusion.data import get_delta_t

# Treshold T1 values to range [TMIN, TMAX]
TMIN, TMAX = 0.2, 4.5 # seconds
MAX_ALLOWED_INCREASE = 400 # in percent, used as a safety to check if the input data is normalized

def concentration_from_T1(T_1_t: numpy.ndarray,T_1_0: numpy.ndarray, r_1: float = 3.2) -> numpy.ndarray:

    """
    Compute tracer concentration c in mmol/L in every voxel from T1 Maps as

    c = 1 / r_1 * (1 / T_1_t - 1 / T_1_0)

    The formula is based on [1].

    Keyword Arguments:

    - T_1_t: T_1-map at time t after contrast injection as np.array
    - T_1_0: baseline T1-map with voxel values in units of seconds as np.ndarray
    - r_1: Relaxitvity of the CSF tracer in units of L / (mmol * s). Default=3.2 (gadobutrol in water at 37 Celsius and 3T field strenght [2])

    Returns:
        numpy.ndarray: concentration array with same shape as T_1_t, T_1_0


    [1] Valnes, Lars Magnus et al.
    "Supplementary Information for "Apparent diffusion
    coefficient estimates based on 24 hours tracer
    movement support glymphatic transport in human cerebral cortex",
    Scientific Reports (2020)

    [2] Rohrer, Martin et al.
    "Comparison of magnetic properties of MRI contrast media solutions at different magnetic field strengths",
    Investigative radiology (2005)
    """
    # Equation (S3) in [1]
    concentration = 1 / r_1 * (1 / T_1_t - 1 / T_1_0)

    return concentration    


def signal_to_T1(S_t: numpy.ndarray, S_0: numpy.ndarray, T_1_0: numpy.ndarray, b:float = 1.48087682, T_min: float = TMIN, T_max: float = TMAX) -> numpy.ndarray:

    """
    Estimate T1 in every image voxel of a normalized T1-weighted images via the equation

    T_1 = f^{-1}( S_t / S_0 * f(T_1_0) )

    We use an approximation f(T) = A exp(- B * T) to the function f defined by equation (S11) in [1].
    For the MPRAGE parameters in [1] a fit yields B = 1.48087682 / s for T \in [0.2 s, 4.5 s].

    For details on the normalization (every image should be divided by the MRI signal in a reference ROI), see the supplementary materials in [2].


    [1] Valnes, Lars Magnus et al.
    Supplementary Information for "Apparent diffusion
    coefficient estimates based on 24 hours tracer
    movement support glymphatic transport in human cerebral cortex",
    Scientific Reports (2020)
    [2] PK Eide et al. "Sleep deprivation impairs molecular clearance from the human brain",
    Brain (2021)


    Keyword Arguments:

    - S_t: normalized T1-weighted image/signal after contrast injection as np.array
    - S_t: normalized T1-weighted baseline image/signal as np.array
    - T_1_0: baseline T1-map with voxel values in units of seconds as np.ndarray
    - b: Fit parameter to the function f defined by equation (S11) in [1], unit 1/s, default = 1.48087682
    - T_min, lower threshold for estimated T_1 in s, everything below is assumed to be noise. default = 0.2
    - T_max, upper threshold for estimated T_1 in s, everything above is assumed to be noise. default = 4.5

    Returns:
        numpy.ndarray: T1-map array with same shape as S_t, S_0, T_1_0
    """

    if numpy.max(S_t[~numpy.isnan(S_t)]) > MAX_ALLOWED_INCREASE:
        answer = None
        while answer not in ["y", "n"]:
            answer = input("Maximum increase larger than 400 %. Did you normalize the T1-weighted images? Type 'y' to continue, 'n' to abort.").lower()
        
        if answer == "y":
            pass
        else:
            print("'n' was entered, Exiting script.")
            exit()



    if numpy.max(T_1_0) > T_max:
        warnings.warn("Warning: Maximum value in T1 Map is, " + format(numpy.max(T_1_0)) +",T1 values should be given in units of seconds")

    # Rearrange (S13) in [1] with f(T) = A exp(-B*T) to obtain:
    T_1 = T_1_0 - numpy.log(S_t / S_0) / b

    # Threshold unreasably high / low T1 values (noise)
    T_1 = numpy.where(T_1 < T_min, T_min, T_1)
    T_1 = numpy.where(T_1 > T_max, T_max, T_1)

    return T_1    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputfolder", required=True, type=str, 
        help="path to folder in which the normalized images are stored.")
    parser.add_argument("--exportfolder", required=True, type=str, 
        help="path to export concentration images. will be stored. Will be created if it does not exist.")
    parser.add_argument('--t1map', type=str, default=None, 
                        help=("T1 Map as .mgz file."
                              "If None, it is assumed that the images in --inputfolder are T1 maps."))
    parser.add_argument('--mask', type=str, default=None, help="Path to mask for the brain")
    parser.add_argument('--baseline', type=str, default=None, help="Manually specify path to baseline in case sorting fails.")
    
    parserargs = vars(parser.parse_args())
    
    inputfolder = pathlib.Path(parserargs["inputfolder"])
    exportfolder = pathlib.Path(parserargs["exportfolder"])
    
    os.makedirs(exportfolder, exist_ok=True)

    images = sorted([inputfolder / f for f in os.listdir(parserargs["inputfolder"]) if f.endswith(".mgz") and "template" not in f])

    if parserargs["baseline"] is not None:
        baseline_file = pathlib.Path(parserargs["baseline"])
        assert baseline_file.is_file()
    else:
        baseline_file = images[0]

    print("")
    print("Loading baseline image", baseline_file.name)
    print("The other images are (sorted):")
    for im in images:
        print("--", im.name)


    if parserargs["baseline"] is None:
        answer = None
        while answer not in ["y", "n"]:
            answer = input("Is the baseline correct? Type either of [y, n]:").lower()
        
        if not answer == "y":
            print("Manually set baseline with the optional --baseline argument. Exiting script.")
            exit()

    baseline_img = nibabel.load(baseline_file)
    affine = baseline_img.affine
    baseline = baseline_img.get_fdata()


    if parserargs["mask"] is not None:
        mask = nibabel.load(parserargs["mask"])
        
        assert numpy.allclose(affine, mask.affine), "Affine transformations differ, are you sure the baseline and T1 Map are registered properly?"

        mask = mask.get_fdata().astype(bool)

        baseline *= mask


    if parserargs["t1map"] is not None:
        baseline_t1_map = nibabel.load(parserargs["t1map"])
        t1_map_affine = baseline_t1_map.affine
        baseline_t1_map = baseline_t1_map.get_fdata()
        assert numpy.allclose(affine, t1_map_affine), "Affine transformations differ, are you sure the images are registered properly?"

        if baseline_t1_map.max() > 100:
            print("Assuming T1 Map is given in milliseconds, converting to seconds")
            baseline_t1_map /= 1e3

        # Treshold extreme values (probably noise / artefacts):
        baseline_t1_map = numpy.where(baseline_t1_map < TMIN, TMIN, baseline_t1_map)
        baseline_t1_map = numpy.where(baseline_t1_map > TMAX, TMAX, baseline_t1_map)


        # Check if the images are normalized by checking the max value in baseline (should be of the order of 1)

        if numpy.nanmax(baseline) > 100:
            answer = None
            while answer not in ["y", "n"]:
                print("Maximum voxel value in baseline is > 10. This is suspicious, they should be ~ 1 after normalization")
                answer = input("Did you normalize the T1-weighted images using a proper ROI ? Type either of [y, n]:").lower()

            if not answer == "y":
                print("Normalize the T1-weighted images.")
                print("Exiting script.")
                exit() 

    else:
        print("*"*100)
        print("Argument --t1map not specified, assuming the image images in --inputfolder are T1 maps (not T1-weighted images!)")

        answer = None
        while answer not in ["y", "n"]:
            answer = input("Is this correct? Type either of [y, n]:").lower()

        if not answer == "y":
            print("Specify path to T1 map with --t1map or")
            print("Normalize the images and ")
            print("run scripts/data-processing/make_brainmask.py first to generate synthetic T1 Map.")
            print("Exiting script.")
            exit()

        print("*"*100)
        baseline_t1_map = baseline





    for imagepath in images:
        
        assert imagepath.is_file()

        print("Converting", imagepath)
        
        image = nibabel.load(imagepath)
        
        assert numpy.allclose(affine, image.affine), "Affine transformations differ, are you sure the images are registered properly?"
        
        image = image.get_fdata()

        if parserargs["mask"] is not None:
            image = numpy.where(mask, image, numpy.nan)

        if parserargs["t1map"] is not None:
            T_1_t = signal_to_T1(S_t=image, S_0=baseline, T_1_0=baseline_t1_map)
        else:
            T_1_t = image

        concentration = concentration_from_T1(T_1_t=T_1_t, T_1_0=baseline_t1_map)

        time_after_baseline = get_delta_t(file1=baseline_file.stem, file2=imagepath.stem, name_format='%Y%m%d_%H%M%S')

        hours, remainder = divmod(time_after_baseline / 3600, 1)

        minutes = remainder * 60

        assert minutes < 60
        assert minutes >= 0
        assert hours >= 0

        filename = format(hours + minutes / 100, ".2f") + ".mgz"

        print("Storing to", str(exportfolder / filename))
        nibabel.save(nibabel.Nifti1Image(concentration.astype(float), affine), str(exportfolder / filename))