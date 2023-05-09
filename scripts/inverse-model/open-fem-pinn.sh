#!/bin/bash
# set -o errexit # Exit the script on any error
# set -o nounset # Treat any unset variables as an error

echo "FreeSurfer configuration is required to run this script" 
if [ ! -z "${FREESURFER_HOME}" ];
then
   echo "** FreeSurfer found"  
else
   echo "FreeSurfer not found" 
   exit
fi

# Run 
# conda activate 
# in terminal before
python scripts/data-processing/mask-mri.py \
--imagefolder data/freesurfer/CONCENTRATIONS/ \
--mask roi12/parenchyma_mask_roi.mgz \
--outputfolder data/freesurfer/MASKED_CONCENTRATIONS/

BASELINE_IMAGE=./data/freesurfer/REGISTERED/20230213_073508.mgz

DATA7H=data/freesurfer/MASKED_CONCENTRATIONS/6.56
DATA50H=data/freesurfer/MASKED_CONCENTRATIONS/50.39

FEMPATH=./simulation_outputs/3600_roi12

FEM7h=${FEMPATH}/7h
FEM50h=${FEMPATH}/50h

PINNPATH=./simulation_outputs/pinn1e-3
PINN7h=${PINNPATH}/7h
PINN50h=${PINNPATH}/50h


for IMAGE in $DATA7H $DATA50H $FEM7h $FEM50h $PINN7h $PINN50h
do
echo $IMAGE
freeview --slice 146 100 133 ${BASELINE_IMAGE} --colormap Jet -v ${IMAGE}.mgz --zoom 4  \
--viewport axial --layout 1 --quit --screenshot ${IMAGE}.png 2 true
done