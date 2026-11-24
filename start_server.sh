#!/bin/bash
# FoundationPose service for garment_connector (127.0.0.1:47993). Ctrl+C to stop.
# Runs fp_server.py inside the conda env, with the ROS variables removed so that the ROS
# Python packages are not mixed in. Nothing outside ~/foundationpose is touched.
FP=$HOME/foundationpose
cd "$FP/work" || exit 1
exec env -u PYTHONPATH -u LD_LIBRARY_PATH -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH \
  CONDARC="$FP/condarc" XDG_CACHE_HOME="$FP/cache" PYTHONNOUSERSITE=1 \
  CUDA_HOME="$FP/miniforge3/envs/foundationpose" \
  "$FP/miniforge3/bin/conda" run --no-capture-output -n foundationpose python fp_server.py "$@"
