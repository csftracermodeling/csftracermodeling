# Setup

For FEniCS scripts and data processing, run in terminal:

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
