#!/bin/bash

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

set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error


bash ./scripts/mesh-generation/extract-ventricles.sh
python ./scripts/mesh-generation/make_mesh.py