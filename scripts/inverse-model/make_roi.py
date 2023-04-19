import nibabel
import pathlib
import numpy as np
import itertools
import os
import SVMTK as svmtk
import shutil
from tracerdiffusion.utils import find_interior_boundary
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--maskfile", default="./data/mridata3d/mri/parenchyma_mask.mgz", help="Path to parenchyma_mask.mgz")
    parser.add_argument("--resolution", type=int, default=20, help="Resolution of the volume mesh for the ROI")
    parser.add_argument("-R", "--radius", type=int, default=10, help="ROI radius in number of voxels (integer)")
    parser.add_argument("-i", "--center_i", type=int, default=146, help="roi center in first direction")
    parser.add_argument("-j", "--center_j", type=int, default=100, help="roi center in first direction")
    parser.add_argument("-k", "--center_k", type=int, default=133, help="roi center in first direction")
    parserargs = vars(parser.parse_args())

    meshresolution = parserargs["resolution"]
    
    outfolder = pathlib.Path(os.getcwd()) / ("roi" + str(meshresolution) + "/")

    if outfolder.is_dir():

        answer = None
        while answer not in ["y", "n"]:
            answer = input("Outputfolder " + str(outfolder) + "  exists, delete and create files from scratch? [y, n]:").lower()

        if not answer == "y":
            print("Not deleting outputfolder" + str(outfolder) + " (not recommended).")
        else:
            print("Deleting existing outputfolder", outfolder)
            shutil.rmtree(outfolder)

    os.makedirs(outfolder, exist_ok=True)

    maskfile = pathlib.Path(parserargs["maskfile"])

    os.chdir(maskfile.parent)

    maskfile = maskfile.name

    mask_volume = nibabel.load(maskfile)

    mask = mask_volume.get_fdata()


    """
    Create a ROI as intersection of sphere centered at (i_cursor, j_cursor, k_cursor) and brainmask
    """

    i_cursor, j_cursor, k_cursor = parserargs["center_i"], parserargs["center_j"], parserargs["center_k"] 

    radius = parserargs["radius"]

    roi = np.zeros_like(mask)

    for di, dj, dk in itertools.product(range(-radius, radius + 1), repeat=3):
        if abs(di) ** 2 + abs(dj) ** 2 + abs(dk) ** 2 <= radius ** 2:
            roi[i_cursor + di, j_cursor + dj, k_cursor + dk] = 1.

    roi = roi * mask

    print("number of voxels in ROI:", roi.sum())

    roifile = str(outfolder / maskfile.replace(".mgz", "_roi.mgz"))
    roisurffile = str(outfolder / maskfile.replace(".mgz", "_roi.stl"))

    nibabel.save(nibabel.Nifti1Image(roi, mask_volume.affine) , roifile)


    """
    Get a binary mask for the boundary.
    This can be useful for PINNs if you want to enforce boundary conditions without relying on meshes.
    """
    
    try:
        # find_interior_boundary() relies on the following packages, try to imprt them
        from scipy.ndimage import convolve
        from skimage.measure import marching_cubes
        import cc3d

        boundary = find_interior_boundary(roi)
        boundaryfile = str(outfolder / maskfile.replace(".mgz", "_boundary.mgz"))
        nibabel.save(nibabel.Nifti1Image(boundary, mask_volume.affine) , boundaryfile)
    except ModuleNotFoundError:
        pass


    """
    Create a surface mesh from the ROI using FreeSurfer
    """

    os.system("mri_binarize --i " + roifile + " --match 1 --surf-smooth 3 --surf " + roisurffile)


    """
    Create a volume mesh from the boundary mesh using SVMTK
    """

    surface = svmtk.Surface(roisurffile)

    surface.fill_holes()

    surface.smooth_taubin(2)
    surface.isotropic_remeshing(1, 5, False)
    surface.collapse_edges(0.5*1)
    surface.fill_holes()

    domain = svmtk.Domain(surface)

    domain.create_mesh(meshresolution)

    outfile = roisurffile.replace(".stl", str(meshresolution) + ".mesh")

    # save the volume mesh as .mesh file
    domain.save(outfile)

    # Convert mesh to FEniCS format (.xml) and paraview-friendly format for visualization (.xdmf)
    os.system("meshio-convert " + outfile + " " + outfile.replace(".mesh", ".xml"))
    os.system("meshio-convert " + outfile + " " + outfile.replace(".mesh", ".xdmf"))


    """
    Print some info over the mesh using FEniCS
    """

    try:
        from fenics import *
        roimesh = Mesh(outfile.replace(".mesh", ".xml"))

        print("Some info on your mesh:")
        print("(hmin, hmax) = (", roimesh.hmin(), roimesh.hmax() ,")")
        print("Number of cells =", format(roimesh.cells().shape[0], ".1e"))
        print("Number of vertices =", format(roimesh.coordinates().shape[0], ".1e"))
        print("MeshQuality.radius_ratio_min_max=", MeshQuality.radius_ratio_min_max(roimesh))

        assert min(MeshQuality.radius_ratio_min_max(roimesh)) > 1e-6, "Mesh contains degenerated cells"
    except ModuleNotFoundError:
        print()
        print("FEniCS not installed, could not print info about mesh")
        print()


