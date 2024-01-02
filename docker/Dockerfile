# source: https://github.com/jorgenriseth/multicompartment-solute-transport/blob/main/docker/Dockerfile

FROM condaforge/mambaforge 

ENV DEBIAN_FRONTEND=noninteractive

# Install ssh (missing dependency to run conda envs)
RUN apt-get update && apt-get install -y ssh build-essential \
    git \
    bzip2 \
    cmake \
    curl \
    g++ \
    libboost-all-dev \
    libeigen3-dev \
    libgmp3-dev \
    libmpfr-dev \
    xz-utils \
    zlib1g-dev \
    python3-pip

# Downlod CGAL and pybind
RUN mkdir external && cd external && \
    git clone https://github.com/pybind/pybind11.git --branch=v2.6.2 && \
    git clone https://github.com/CGAL/cgal.git --branch v5.3.1

RUN pip3 install pip setuptools --upgrade

RUN python3 -m pip install numpy

RUN git clone --recursive https://github.com/SVMTK/SVMTK
RUN cd SVMTK && python3 -m pip install . && cd ..

RUN python3 -c "import SVMTK"


# Upgrade mamba
RUN mamba upgrade -y mamba

# Copy code into docker env
RUN mkdir ./tracerdiffusion/
RUN mkdir ./tracerdiffusion/src/
RUN mkdir ./tracerdiffusion/src/tracerdiffusion/

COPY src/tracerdiffusion/* ./tracerdiffusion/src/tracerdiffusion/
COPY fenics-env.yml ./tracerdiffusion/
COPY pyproject.toml ./tracerdiffusion/

# Update environment file with new environment name
RUN mamba env update --file ./tracerdiffusion/fenics-env.yml --name fenics
SHELL ["mamba", "run", "-n", "fenics", "/bin/bash", "-c"]

RUN cd tracerdiffusion && python3 -m pip install -e . && cd ..

# Test dependencies
RUN python3 -c "import dolfin; print(dolfin.__version__); import h5py; print(h5py.__version__)"
RUN python3 -c "import dolfin_adjoint; print(dolfin_adjoint.__version__); import h5py; print(h5py.__version__)"
RUN python3 -c "import SVMTK"
RUN python3 -c "import tracerdiffusion"

RUN echo "source activate fenics" > ~/.bashrc

