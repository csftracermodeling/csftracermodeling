#!/bin/bash
set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

echo "FreeSurfer configuration is required to run this script" 
if [ ! -z "${FREESURFER_HOME}" ];
then
   echo "** FreeSurfer found"  
else
   echo "FreeSurfer not found" 
   exit
fi

echo "Checking if path to mri2fem2 dataset is set" 
if  [ ! -z "${WORKDIR}" ]; 
then
   echo "** mri2fem2 dataset found"
else
   echo "mri2fem2 dataset not found"
   echo "Run setup in mri2fem2-dataset folder" 
   echo "source Setup_mri2fem2_dataset.sh" 
   exit
fi

# Input and output filenames
input=${WORKDIR}/mri/wmparc.mgz

mkdir -pv ${WORKDIR}/surf/
output=${WORKDIR}/surf/ventricles.stl


# Also match the 4th ventricle and aqueduct?
include_fourth_and_aqueduct=true
if [ "$include_fourth_and_aqueduct" == true ]; then
    matchval="15"
else
    matchval="1"
fi
num_smoothing=3

# Other parameters
postprocess=true
num_closing=2
V_min=100

if [ "$postprocess" == true ]; then
    mri_binarize --i $input --ventricles \
	         --o "tmp.mgz"
    
    mri_volcluster --in "tmp.mgz" \
	           --thmin 1 \
	           --minsize $V_min \
	           --ocn "tmp-ocn.mgz"
    
    mri_binarize --i "tmp-ocn.mgz" \
	         --match 1 \
	         --o "tmp.mgz"
    
    mri_morphology "tmp.mgz" \
	           close $num_closing "tmp.mgz"
    
    mri_binarize --i "tmp.mgz" \
	         --match 1 \
	         --surf-smooth $num_smoothing \
	         --surf $output
    

else
    mri_binarize --i $input --ventricles \
        --match $matchval \
        --surf-smooth $num_smoothing \
        --surf $output
fi

rm tmp.mgz
rm tmp-ocn.mgz
rm tmp-ocn.lut

mris_convert ${WORKDIR}/surf/lh.pial ${WORKDIR}/surf/lh.pial.stl


echo "**********************************************************************************************************"
echo "Script done, run"
echo "freeview ${WORKDIR}/mri/aseg.mgz -f ${WORKDIR}/surf/ventricles.stl"
echo "to inspect the ventricle surface"
echo "**********************************************************************************************************"
