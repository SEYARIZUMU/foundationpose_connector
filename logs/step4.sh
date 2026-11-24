#!/bin/bash
# step 4: PyTorch3D and nvdiffrast from source, as in the official readme (CUDA_HOME = the env)
set -e
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=8 MAKEFLAGS=-j8 CMAKE_BUILD_PARALLEL_LEVEL=8 NINJAFLAGS=-j8
export FORCE_CUDA=1
echo "CC=$CC CXX=$CXX CUDA_HOME=$CUDA_HOME nvcc=$(which nvcc) python=$(which python)"
date
nice -n 15 python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
date
nice -n 15 python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
date
echo STEP4_DONE
