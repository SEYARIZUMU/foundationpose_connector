#!/bin/bash
# FoundationPose service for garment_connector (127.0.0.1:47993), fast version: the searches
# start only from hypotheses whose mating axis is within FP_CONE deg of the tool axis
# (fp_server_fast.py, 2026-11-27; 0.8 s -> ~0.1 s per answer). Ctrl+C to stop.
# FP_CONE defaults to 45 (as relative.fp.max_axis_deg); e.g. FP_CONE=35 ./start_server_fast.sh
# Runs inside the conda env, with the ROS variables removed so that the ROS Python packages
# are not mixed in. Nothing outside ~/foundationpose is touched.
FP=$HOME/foundationpose
cd "$FP/work" || exit 1
# what it prints (the time of every answer) is also kept in $FP/logs
LOG="$FP/logs/fp_server_fast_$(date +%Y%m%d_%H%M%S).log"
echo "log: $LOG"
exec > >(tee -a "$LOG") 2>&1
exec env -u PYTHONPATH -u LD_LIBRARY_PATH -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH \
  CONDARC="$FP/condarc" XDG_CACHE_HOME="$FP/cache" PYTHONNOUSERSITE=1 \
  CUDA_HOME="$FP/miniforge3/envs/foundationpose" \
  FP_CONE="${FP_CONE:-45}" \
  "$FP/miniforge3/bin/conda" run --no-capture-output -n foundationpose python fp_server_fast.py "$@"
