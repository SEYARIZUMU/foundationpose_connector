"""FoundationPose on the simulated wrist frames; meshes = the boxes the simulation renders (face frame)."""
import glob, os, sys, time
import numpy as np, trimesh, yaml
sys.path.insert(0, os.path.expanduser('~/foundationpose/FoundationPose'))
from estimater import *
W = os.path.dirname(os.path.abspath(__file__))
geo = yaml.safe_load(open(os.path.expanduser('~/garment_ws/src/garment_research_ros2/garment_connector/config/connector_geometry.yaml')))
set_seed(0)
scorer, refiner, glctx = ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()
ests = {}
for side, shape in (('left', 'plug'), ('right', 'receptacle')):
    parts = []
    for b in geo[shape]['boxes']:
        m = trimesh.creation.box(extents=b['size']); m.apply_translation(b['center']); parts.append(m)
    mesh = trimesh.util.concatenate(parts); mesh.export(f'{W}/sim_{shape}_boxes.obj')
    print(side, shape, 'extent mm', (mesh.extents * 1000).round(1), 'boxes', len(parts))
    ests[side] = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=f'{W}/sim_out/debug', debug=0, glctx=glctx)
for f in sorted(glob.glob(f'{W}/sim_inputs/*.npz')):
    d = np.load(f); name = os.path.basename(f)[:-4]; t0 = time.time()
    pose = ests[str(d['held'])].register(K=d['K'], rgb=d['color'], depth=d['depth'], ob_mask=d['mask'], iteration=5)
    np.save(f'{W}/sim_out/{name}_pose.npy', pose); print(f'{name}: {time.time() - t0:.1f} s', flush=True)
print('RUN_DONE')
