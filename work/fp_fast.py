"""FoundationPose with the speed-ups (2026-11-26), shared by fp_server_fast.py and fp_bench.py.

Options (Opts), each off / as before by default:
  iteration  refinement passes per hypothesis (fp_server.py: 5)
  cone_deg   keep only the hypotheses whose mating axis (CAD +z = face +z) is within this
             angle of the nominal axis `axis` (the tool axis in the camera; the node drops
             answers more than relative.fp.max_axis_deg = 45 off it anyway). All turns about
             the axis are kept, so the roll vote is unchanged. 0 or no axis: all 252.

track(): FoundationPose's tracking (track_one), refining only the given start pose: for after
the latch side is decided.

Runs inside the FoundationPose environment.
"""
import os, sys
from dataclasses import dataclass
import numpy as np, torch, trimesh
sys.path.insert(0, os.path.expanduser('~/foundationpose/FoundationPose'))
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, dr, set_seed

# CAD [mm]: +z mating direction, +y latch. Face frame: x = latch, origin = center of the face.
CAD = {'left': ('~/connector_cad/46993-0620.stl', (0.0, 0.0058, 0.0)),
       'right': ('~/connector_cad/46992-0620.stl', (0.0042, 0.0, 0.0089))}
RZ90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Opts:
    iteration: int = 5
    cone_deg: float = 0.0
    track_iter: int = 4      # refinement passes of track() (2: 25 ms, 4: 34 ms on the RTX 4090)

    @staticmethod
    def from_env():
        return Opts(iteration=int(os.environ.get('FP_ITER', 5)),
                    cone_deg=float(os.environ.get('FP_CONE', 0.0)),
                    track_iter=int(os.environ.get('FP_TRACK_ITER', 4)))


GEOMETRY = '~/garment_ws/src/garment_research_ros2/garment_connector/config/connector_geometry.yaml'
SIM = {'left': 'plug', 'right': 'receptacle'}      # what each arm holds in Gazebo


def sim_mesh(shape):
    """The boxes Gazebo renders for `shape` (connector_geometry.yaml), in the face frame."""
    import yaml
    geo = yaml.safe_load(open(os.path.expanduser(GEOMETRY)))
    parts = []
    for b in geo[shape]['boxes']:
        m = trimesh.creation.box(extents=b['size'])
        m.apply_translation(b['center'])
        parts.append(m)
    return trimesh.util.concatenate(parts)


def load(meshes=None):
    """Estimators for both connectors, mesh_from_face per side, the rasterizer context.
    meshes: 'real' (the Molex CAD) or 'sim' (Gazebo's boxes, run_fp_sim.py); default from the
    environment FP_MESH, else real."""
    meshes = meshes or os.environ.get('FP_MESH', 'real')
    set_seed(0)
    scorer, refiner, glctx = ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()
    est, mesh_from_face = {}, {}
    for side, (path, center) in CAD.items():
        if meshes == 'sim':
            m = sim_mesh(SIM[side])
            est[side] = FoundationPose(model_pts=m.vertices, model_normals=m.vertex_normals,
                                       mesh=m, scorer=scorer, refiner=refiner, debug=0,
                                       glctx=glctx, debug_dir='/tmp/unused_fp_debug')
            mesh_from_face[side] = np.eye(4)
            continue
        m = trimesh.load(os.path.expanduser(path), force='mesh')
        m.apply_scale(0.001)
        m.merge_vertices()
        est[side] = FoundationPose(model_pts=m.vertices, model_normals=m.vertex_normals, mesh=m,
                                   scorer=scorer, refiner=refiner, debug=0, glctx=glctx,
                                   debug_dir='/tmp/unused_fp_debug')
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = RZ90, center        # p_cad = Rz(90) p_face + center
        mesh_from_face[side] = t
    return est, mesh_from_face, glctx


def estimate(e, color, depth, mask, K, opts, axis=None):
    """Pose of the CAD mesh in the camera (as FoundationPose.register returns it).
    axis: nominal mating axis in the camera (3,), for opts.cone_deg."""
    full = e.rot_grid
    if opts.cone_deg > 0 and axis is not None:
        a = torch.as_tensor(np.asarray(axis, float) / np.linalg.norm(axis), device='cuda',
                            dtype=torch.float)
        keep = full[:, :3, 2] @ a >= np.cos(np.radians(opts.cone_deg))
        if keep.any():
            e.rot_grid = full[keep]
    try:
        return e.register(K=K.astype(float), rgb=color, depth=depth.astype(np.float32),
                          ob_mask=mask.astype(bool), iteration=opts.iteration)
    finally:
        e.rot_grid = full


def track(e, color, depth, K, init, iteration=2):
    """Pose of the CAD mesh in the camera (as estimate gives it) refined from `init` (the same
    kind of pose, e.g. the last answer moved with the arms) by FoundationPose's tracking: one
    hypothesis, no turn scores."""
    tf = e.get_tf_to_centered_mesh()
    e.pose_last = torch.as_tensor(np.asarray(init, float), device='cuda',
                                  dtype=torch.float) @ torch.linalg.inv(tf)
    return e.track_one(rgb=color, depth=depth.astype(np.float32), K=K.astype(float),
                       iteration=iteration)
