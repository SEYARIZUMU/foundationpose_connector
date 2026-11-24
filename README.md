# foundationpose_connector

FoundationPose service for garment_connector (connector mating on the real robot).

- `start_server.sh`: starts `work/fp_server.py` (127.0.0.1:47993) in the conda env.
- `work/fp_server.py`: partner connector poses for `relative.fp`; `work/latch_depth.py`: latch side from the outline where the two half-turn poses differ.
- `work/*` others: offline runs on recorded / simulated frames.
- `logs/step*.sh`: the install steps.

Not in the repository: the FoundationPose clone, the conda env (miniforge3/), the cache, recorded frames, and the Molex meshes (`work/46992-0620_m.obj`, `work/46993-0620_m.obj`). Setup: garment_connector/docs/foundationpose_mating.md.
