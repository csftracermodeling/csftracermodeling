# Requirements

The conda environment `diffusion-fenics-env` needs to installed and activated, [see here for instructions.](https://github.com/bzapf/tracerdiffusion/tree/main)

# Running a simulation
Then call the script from the top level of this repository as
```
python ./scripts/forward-model/diffusion.py \
  --data ./data/freesurfer/CONCENTRATIONS/ --mesh ./data/freesurfer/meshes/lh.xml
```
