#!/bin/bash
set -e
export CUDA_HOME="$CONDA_PREFIX"; export PATH="$CUDA_HOME/bin:$PATH"
cd "$HOME/foundationpose/work"
date; python run_fp.py "$1"; date; echo STEP8_DONE
