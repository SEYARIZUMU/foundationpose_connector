#!/bin/bash
# step 7: the official demo (mustard bottle), without opening a window; images are saved
set -e
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
# a virtual display (xvfb, already on this PC): the demo opens a window, nothing appears on the desktop
cd "$HOME/foundationpose/FoundationPose"
date
xvfb-run -a python run_demo.py --debug 2
date
echo STEP7_DONE
