import os
import pathlib
import argparse
import numpy
import nibabel
   
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--aseg', required=True, type=str, 
    help="FreeSurfer segmentation file aseg.mgz")
parser.add_argument('--maskname', default=None, type=str, 
    help="Filepath where brain mask should be stored. Will be stored as parenchyma_only.mgz into the folder of aseg.mgz by default.")

parser.add_argument('--t1', default=None, type=float, 
    help="Set the voxel valuels inside mask to t1 (create synthetic T1 map). Note: the other scripts use units of seconds")
parser.add_argument('--t1mapname', default=None, type=str, 
    help="Filepath where the synthetic T1 map should be stored.Will be stored as synthetic_T1_map.mgz into the folder of aseg.mgz by default.")


parserargs = parser.parse_args()
parserargs = vars(parserargs)


# From the FreeSurfer labels
# https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
# we set the mask to 0 for those voxels labeled the CSF:
csf_labels = [4, 5, 14, 15, 24, 43, 44]

aseg = nibabel.load(parserargs["aseg"])
affine = aseg.affine

aseg = aseg.get_fdata().astype(int)

csf_mask = numpy.zeros(tuple(aseg.shape), dtype=bool)

for csf_label in csf_labels:
    csf_mask += (aseg == csf_label) 

brainmask = (aseg > 0) * (~ csf_mask)

if parserargs["maskname"] is None:
    outfile = pathlib.Path(parserargs["aseg"]).parent / "parenchyma_only.mgz"
else:
    outfile = parserargs["maskname"]

if not pathlib.Path(parserargs["maskname"]).parent.is_dir():
    os.makedirs(pathlib.Path(parserargs["maskname"]).parent, exist_ok=True)

nibabel.save(nibabel.Nifti1Image((brainmask).astype(float), affine), outfile)

print("Created mask, to view run")
print("freeview ", parserargs["aseg"], " ", outfile)

if parserargs["t1"] is not None:
    if parserargs["t1mapname"] is None:
        outfile = pathlib.Path(parserargs["aseg"]).parent / "synthetic_T1_map.mgz"
    else:
        outfile = parserargs["t1mapname"]

    if not pathlib.Path(parserargs["t1mapname"]).parent.is_dir():
        os.makedirs(pathlib.Path(parserargs["t1mapname"]).parent, exist_ok=True)
        
    nibabel.save(nibabel.Nifti1Image((brainmask * parserargs["t1"]).astype(float), affine), outfile)