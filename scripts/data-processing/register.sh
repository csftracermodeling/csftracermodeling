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


INPUTFOLDER=${WORKDIR}/PREREG
OUTPUTFOLDER=${WORKDIR}/RESAMPLED
mkdir -pv ${OUTPUTFOLDER}



# Resample the images to 256x256x256 voxels (FreeSurfer standard) 
for inputfile in ${INPUTFOLDER}/*.mgz; do
    # Here, we use the shell command 'basename' to extract the filename from the variable 'inputfile' which contains the full path to the image. 
    filename=$(basename $inputfile)
    # Resample to 256 x 256 x 256
    # cf. https://surfer.nmr.mgh.harvard.edu/fswiki/mri_convert
    mri_convert --conform -odt float  ${inputfile} ${OUTPUTFOLDER}/${filename}
    echo "Resampled" $filename
done

# create a list of the files in OUTPUTFOLDER
RESAMPLED_IMAGES=$(find ${OUTPUTFOLDER}/*.mgz )

# We register the resampled images to limit computational effort and memory requirements:
INPUTFOLDER=${WORKDIR}/RESAMPLED

OUTPUTFOLDER=${WORKDIR}/REGISTERED
mkdir -pv ${OUTPUTFOLDER}
TEMPLATE=${OUTPUTFOLDER}/template.mgz



# create registration template
# cf. https://surfer.nmr.mgh.harvard.edu/fswiki/mri_robust_template
# this takes some time so we check if it has been done before:
if [ ! -f ${TEMPLATE} ]; then
    mri_robust_template --mov ${RESAMPLED_IMAGES} --average 1 --template ${TEMPLATE} --satit --inittp 1 --fixtp  --maxit 10
else
    echo "** Template" ${TEMPLATE} "exists, continuing"
fi



# Register the images to template.mgz
# cf. https://surfer.nmr.mgh.harvard.edu/fswiki/mri_robust_register

mkdir -pv ${WORKDIR}/LTA/

for inputfile in ${INPUTFOLDER}/*.mgz; do

    filename=$(basename $inputfile )

    mri_robust_register --mov ${inputfile} --dst ${TEMPLATE} --lta ${WORKDIR}/LTA/${filename}.lta \
    --mapmov ${OUTPUTFOLDER}/${filename} --iscale --satit --maxit 10
    
    echo "Done with" $filename

done

freeview ${OUTPUTFOLDER}/*