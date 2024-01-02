# Setup

For FEniCS scripts, run in terminal:

```
git clone https://github.com/bzapf/inverseDiffusion.git
cd inverseDiffusion
conda env create -f fenics-env.yml
conda activate diffusion-fenics-env
pip install -e .
export WORKDIR=./data/freesurfer/
```



For Jax PINN scripts, run
```
$ python scripts/inverse-model/pinn-inverse-diffusion.py
```
This script stores output to './pinn_outputs/'.

To investigate the loss and parameters after training, in a second terminal you can run
```
$ python scripts/inverse-model/pinn-postprocess.py \
--outfolder pinn_outputs --datapath data/freesurfer/CONCENTRATIONS/ \
--mask data/roi12/parenchyma_mask_roi.mgz
```
This will also plot the prediction and store the prediction at all time points as MRI.


# Compare FEM and PINN results
```
$ python scripts/inverse-model/compare-pinn-fem.py --pinnfolder pinn_outputs/ --femfolder simulation_outputs_450_roi20/
```



# Examples

```
$ python examples/data_example.py
$ python examples/domain_example.py
```

