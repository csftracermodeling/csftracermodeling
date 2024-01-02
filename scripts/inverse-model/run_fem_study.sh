#!/bin/bash
set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

## Run a systematic study: solve the inverse problem with different mesh sizes and time steps
## After that post-process the result to compute errors and visualization

for DT in 450 900 1800 3600; do
    for ROI in 12 20 28; do
    if [ ! -f roi${ROI}/parenchyma_mask_roi${ROI}.xml ]; then
        python scripts/inverse-model/make_roi.py \
        --maskfile data/freesurfer/mri/parenchyma_mask.mgz \
        --resolution ${ROI}
    fi
    python scripts/inverse-model/fem-inverse-diffusion.py --data data/freesurfer/CONCENTRATIONS/ \
    --mesh roi${ROI}/parenchyma_mask_roi${ROI}.xml --mask roi${ROI}/parenchyma_mask_roi.mgz --dt ${DT} \
    --outfolder simulation_outputs/${DT}_${ROI}/ > ${DT}_${ROI}.txt
    done
done

for DT in 450 900 1800 3600; do
    for ROI in 12 20 28; do
    python scripts/inverse-model/fem-postprocess.py --hdffile simulation_outputs/${DT}_${ROI}/final.hdf \
    --moviefile simulation_outputs/${DT}_${ROI}/movie.hdf --mask roi${ROI}/parenchyma_mask_roi.mgz > ${DT}_${ROI}_post.txt
    done
done
