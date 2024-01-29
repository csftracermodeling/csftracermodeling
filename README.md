# Getting started
Download the code: 
```
git clone https://github.com/bzapf/tracerdiffusion.git
```

## Setup

# Data processing and simulations (FEniCS).
For data processing, installing FreeSurfer is required.
If you only want to run simulations, you do not necessarily need to install FreeSurfer. 
You can instead download the processed data from **???**.

In both cases, install the necessary Python packages by running the following commands in the terminal:

```bash
git clone https://github.com/bzapf/tracerdiffusion.git
cd tracerdiffusion
conda env create -f fenics-env.yml
conda activate diffusion-fenics-env
python3 -m pip install -e .
export WORKDIR=./data/freesurfer/
```

For jax/PINNs scripts, run in terminal:

```bash
git clone https://github.com/bzapf/tracerdiffusion.git
cd tracerdiffusion
conda env create -f jax.yml
conda activate jax
python3 -m pip install -e .
```


# Requirements

## For data processing

- FreeSurfer needs to be installed
- the conda environment `diffusion-fenics-env` needs to be created and the necessary packages installed (see below)
- The following folders/files are needed and can be obtained from https://github.com/bzapf/rawdata and should be put into the top level under `/data/` in your cloned repository as:
```
├──data
    ├──freesurfer
        ├── PREREG
        |   ├── 20220230_070707.mgz
        |   ├── 20220230_080808.mgz
        |   ├── ...
        ├── mri
        │   ├── aseg.mgz
```

## For inverse problems (PINN and FEM)

Assumes that MRI have been processed and concentration estimates computed. 
Also, a mesh is assumed and a region of interest (ROI).
Alternatively, this data can be obtained from https://github.com/bzapf/concentrationdata
and should be put into the top level in your cloned repository under `/data/` as:
```
├──data
    ├──freesurfer
        ├── CONCENTRATIONS
        |   ├── 0.00.mgz
        |   ├── 06.25.mgz
        |   ├── ...
        ├── mri
        │   ├── parenchyma_mask.mgz
        ├── meshes
        │   ├── lh.xml
    ├──roi12
        ├──parenchyma_mask_roi.mgz
        ├──parenchyma_mask_boundary.mgz
        ├──parenchyma_mask_roi12.xml
```

# Running the scripts

Note that there are README files in the subfolders of `/scripts/`.