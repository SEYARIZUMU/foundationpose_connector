#!/bin/bash
# step 5: remaining Python packages and the C++ extension, as in the official readme
set -e
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL=8 MAKEFLAGS=-j8
cd "$HOME/foundationpose/FoundationPose"
date
nice -n 15 python -m pip install -r requirements.txt
date
# the script asks for -j$(nproc); limit the cores it sees to 8
nice -n 15 taskset -c 0-7 bash build_all_conda.sh
date
python check_env.py || echo "CHECK_ENV_EXIT=$?"
echo STEP5_DONE
