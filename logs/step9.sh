#!/bin/bash
set -e
export CUDA_HOME="$CONDA_PREFIX"; export PATH="$CUDA_HOME/bin:$PATH"
cd "$HOME/foundationpose/work"; python run_fp_sim.py; echo STEP9_DONE
