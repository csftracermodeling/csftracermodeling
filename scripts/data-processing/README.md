## Requirements

- FreeSurfer needs to be installed.

- FreeSurfer needs to have been run. The FreeSurfer output folder is assumed to be found under ${WORKDIR}. To set this variable, run  `export WORKDIR=./data/freesurfer/` from the top of this package.

- The Python scripts require `numpy` and `nibabel`.

- T1 Maps or T1-weighted images (depending on what is available) have been extracted from Dicom files (Cf. Book Volume 1) and put under `INPUTFOLDER=${WORKDIR}/PREREG` in .mgz format

- git needs to be installed if you want to clone this repository ("download the code")




The following folders/files are needed:
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


## Getting started

```
git clone https://github.com/bzapf/tracerdiffusion
```

## Usage

```
$ cd tracerdiffusion
```

Set path to data for simplicity
```
$ export WORKDIR=./data/freesurfer/
```

### Registration pipeline

In terminal, run

```
bash register.sh
```



### Normalization

We next normalize the (registered) T1 weighted images (divide by median signal in a reference ROI to account for scanner variability over time).
See [2] (Supplementary Material) for details.

```
python ./scripts/data-processing/normalize.py \
--inputfolder ./data/freesurfer/REGISTERED/ \
--exportfolder ./data/freesurfer/NORMALIZED_CONFORM/ \
--refroi ./data/freesurfer/mri/refroi.mgz
```

### Create brain mask and synthetic T1 map

Create an image with voxel value 1 s everywhere inside the parenchyma:

```
python ./scripts/data-processing/make_brainmask.py \
--aseg ./data/freesurfer/mri/aseg.mgz \
--maskname ./data/freesurfer/mri/parenchyma_mask.mgz \
--t1 1 --t1mapname ./data/freesurfer/mri/synth_T1_map.mgz
```


### Tracer estimation

To estimate tracer concentrations as in [1] (Supplementary Material), first create a synthetic T1 Map and run

```
python ./scripts/data-processing/estimatec.py \
--inputfolder ./data/freesurfer/NORMALIZED_CONFORM/ \
--exportfolder ./data/freesurfer/CONCENTRATIONS/ \
--t1map ./data/freesurfer/mri/synth_T1_map.mgz \
--mask ./data/freesurfer/mri/parenchyma_mask.mgz
```

Note that without the flagged argument `--t1map`, it is assumed that the images in the directory specified with `--inputfolder` are T_1-maps.
With `--mask` we provide a binary mask for the brain tissue, and the concentration is computed only in the masked voxels. 
This is useful for visualizations.
Further, specifying a mask can be necessary if the T1 values are subject to large uncertainties in some regions of the brain, and quantifying tracer within these regions is not feasible (For the T1--map under consideration here, an example are the CSF compartments. 
The used T1 mapping sequence, 'MOLLI5(3)3', was calibrated to yield accurate T1--values in the brain, yielding less reliable values in the CSF.).


If a T1 map is available (same resolution as T1 weighted images and registered to T1 weighted images), replace the argument given to
`--t1map` with the path to the correct T1 Map.

## References

[1] Valnes, Lars Magnus et al. "Supplementary Information for "Apparent diffusion coefficient estimates based on 24 hours tracer movement support glymphatic transport in human cerebral cortex", Scientific Reports (2020)

[2] PK Eide et al. "Sleep deprivation impairs molecular clearance from the human brain", Brain (2021)
