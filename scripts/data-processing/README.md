## Requirements

- FreeSurfer needs to be installed.

- FreeSurfer needs to have been run. The FreeSurfer output folder is assumed to be found under ${WORKDIR}. To set this variable, run  `export WORKDIR=./data/freesurfer/` from the top of this package.

- The Python scripts require `numpy` and `nibabel`.

- T1 Maps or T1-weighted images (depending on what is available) have been extracted from Dicom files (Cf. Book Volume 1) and put under `INPUTFOLDER=${WORKDIR}/PREREG` in .mgz format

- git needs to be installed if you want to clone this repository ("download the code")




The following folders/files are needed:
```

${WORKDIR}
├── PREREG
|   ├── 20230213_073508.mgz
|   ├── 20230213_143049.mgz
|   ├── 20230214_094033.mgz
|   ├── 20230215_101347.mgz
|   ├── 20230216_080616.mgz
├── mri
│   ├── aseg.mgz
```

## Getting started

```
git clone https://github.com/bzapf/inverseDiffusion
```

## Usage

```
$ cd inverseDiffusion
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
python ./scripts/data-processing/normalize.py --inputfolder ${WORKDIR}/REGISTERED/ \
--exportfolder ${WORKDIR}/NORMALIZED_CONFORM/ \
--refroi ${WORKDIR}/mri/refroi.mgz
```

### Create brain mask and synthetic T1 map

Create an image with voxel value 1 s everywhere inside the parenchyma:

```
python ./scripts/data-processing/make_brainmask.py --aseg ${WORKDIR}/mri/aseg.mgz --t1 1 \ 
--maskname ${WORKDIR}/mri/parenchyma_only.mgz \
--t1mapname ${WORKDIR}/mri/synthetic_T1_map.mgz \
```


### Tracer estimation

To estimate tracer concentrations as in [1] (Supplementary Material),

```
python ./scripts/data-processing/estimatec.py --inputfolder ${WORKDIR}/NORMALIZED_CONFORM/ \
--exportfolder ${WORKDIR}/CONCS_constT1/ --t1map ${WORKDIR}/mri/synthetic_T1_map.mgz
```
will create a binary mask for the parenchyma under  `${WORKDIR}/mri/parenchyma_only.mgz`.


We can also map out everything outside the brain to get a cleaner image:
```
python ./scripts/data-processing/estimatec.py --inputfolder ${WORKDIR}/NORMALIZED_CONFORM/ \
--exportfolder ${WORKDIR}/CONCENTRATION/ --t1map ${WORKDIR}/mri/synthetic_T1_map.mgz \
--mask ${WORKDIR}/mri/parenchyma_only.mgz
```



If a T1 map is available (same resolution as T1 weighted images and registered to T1 weighted images)
```
python ./scripts/data-processing/estimatec.py --inputfolder ${WORKDIR}/NORMALIZED_CONFORM/ \
--exportfolder ${WORKDIR}/CONCS_T1MAP/ --t1map ${WORKDIR}/<path_to_your_T1_Map> \
--mask ${WORKDIR}/mri/parenchyma_only.mgz
```

## References

[1] Valnes, Lars Magnus et al. "Supplementary Information for "Apparent diffusion coefficient estimates based on 24 hours tracer movement support glymphatic transport in human cerebral cortex", Scientific Reports (2020)

[2] PK Eide et al. "Sleep deprivation impairs molecular clearance from the human brain", Brain (2021)
