"""FoundationPose on the exported connector frames (run inside the foundationpose env)."""
import glob, os, sys, time
import numpy as np, trimesh, cv2, imageio
sys.path.insert(0, os.path.expanduser('~/foundationpose/FoundationPose'))
from estimater import *          # FoundationPose, ScorePredictor, PoseRefinePredictor, dr, set_seed, draw_*
W = os.path.dirname(os.path.abspath(__file__))
CAD = {'left': '46993-0620.stl', 'right': '46992-0620.stl'}      # connector held by that arm
meshes, ests = {}, {}
set_seed(0)
scorer, refiner, glctx = ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()
for side, f in CAD.items():
    m = trimesh.load(os.path.expanduser('~/connector_cad/' + f), force='mesh')
    m.apply_scale(0.001)                       # the CAD is in mm
    m.merge_vertices()
    m.export(f'{W}/{f[:-4]}_m.obj')
    meshes[side] = m
    print(side, f, 'extent mm', (m.extents * 1000).round(1), 'vertices', len(m.vertices))
    ests[side] = FoundationPose(model_pts=m.vertices, model_normals=m.vertex_normals, mesh=m, scorer=scorer, refiner=refiner, debug_dir=f'{W}/out/debug', debug=0, glctx=glctx)
files = sorted(glob.glob(f'{W}/inputs/*.npz')); only = sys.argv[1] if len(sys.argv) > 1 else ''
files = [f for f in files if only in os.path.basename(f)]
for n, f in enumerate(files):
    d = np.load(f); side = str(d['held']); name = os.path.basename(f)[:-4]
    t0 = time.time()
    pose = ests[side].register(K=d['K'], rgb=d['color'], depth=d['depth'], ob_mask=d['mask'], iteration=5)
    dt = time.time() - t0
    np.save(f'{W}/out/{name}_pose.npy', pose)
    if n % 6 == 0:
        m = meshes[side]; to_origin, ext = trimesh.bounds.oriented_bounds(m); bbox = np.stack([-ext / 2, ext / 2], 0).reshape(2, 3)
        cp = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(d['K'], img=d['color'], ob_in_cam=cp, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=cp, scale=0.02, K=d['K'], thickness=1, transparency=0, is_input_rgb=True)
        ys, xs = np.nonzero(d['mask']); y0, x0 = max(0, int(ys.mean()) - 60), max(0, int(xs.mean()) - 70)
        imageio.imwrite(f'{W}/out/{name}_vis.png', cv2.resize(vis[y0:y0 + 120, x0:x0 + 140], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
    print(f'{name}: {dt:.1f} s, mask {int(d["mask"].sum())} px, distance {np.linalg.norm(pose[:3, 3]) * 1000:.0f} mm', flush=True)
print('RUN_DONE')
