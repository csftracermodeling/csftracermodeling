# Creating a region of interest (ROI)
Run
```
$ python ./scripts/inverse-model/roi.py \
    --maskfile ./data/freesurfer/mri/parenchyma_mask.mgz \
    --resolution 12
```
where `./data/freesurfer/mri/parenchyma_mask.mgz` is an MRI volume where all brain parenchyma voxels are labelled with 1, and all other voxels with 0. The `resolution` parameter determines the resolution of the mesh created. 

# Inverse modeling

## For FEM scripts based on FEniCS
Run
```
$ conda activate diffusion-fenics-env \
$ python scripts/inverse-model/fem-inverse-diffusion.py \
    --data data/freesurfer/CONCENTRATIONS/ \
    --mesh data/roi12/parenchyma_mask_roi12.xml \
    --mask data/roi12/parenchyma_mask_roi.mgz
```

## Physics-informed neural networks

For Jax PINN scripts, run
```
$ conda activate jax
$ python scripts/inverse-model/pinn-inverse-diffusion.py
```
This script stores output to `./pinn_outputs/`.


To investigate the loss and parameters after training, in a second terminal you can run
```
$ conda activate jax
$ python scripts/inverse-model/pinn-postprocess.py \
--outfolder pinn_outputs --datapath data/freesurfer/CONCENTRATIONS/ \
--mask data/roi12/parenchyma_mask_roi.mgz
```
This will also plot the prediction and store the prediction at all time points as MRI.


# Compare FEM and PINN results
Assuming you have stored some PINN outputs under `pinn_outputs` and FEM output under `simulation_outputs/3600_12/` (for a time step of 3600 s and a mesh resolution parameter 12), you can run
```
$ python scripts/inverse-model/compare-pinn-fem-parameters.py \
--pinnfolder pinn_outputs/ --femfolder simulation_outputs/3600_12/
```
and
```
python scripts/inverse-model/compare-pinn-fem-prediction.py \
--pinnfolder pinn_outputs/ --femfolder simulation_outputs/3600_12/ \
--datapath data/freesurfer/CONCENTRATIONS/ \
--mask data/roi12/parenchyma_mask_roi.mgz
```
to get some information about the results.


# Examples

```
$ python examples/data_example.py
$ python examples/domain_example.py
```

