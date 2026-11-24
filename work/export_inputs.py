"""Export recorded wrist-camera frames as FoundationPose inputs (run with the ROS Python).

For each frame: color, depth [m], the mask of the PARTNER connector (as the control node finds
it), intrinsics, both tool poses, and the pose the current model fit gives (for comparison).
The recordings are only read."""
import json, sys
import numpy as np, cv2
from scipy.spatial.transform import Rotation
from garment_connector import multiview_eval as ev, model_fit as mf, multiview as mv
from garment_connector.vs_control import Intrinsics
D, tag, NF, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
base_cfg = ev.load_config('connector_mating.yaml'); real = ev.load_config('connector_mating_real.yaml'); low = ev.load_config('connector_mating_real_lowres.yaml'); rel = real['relative']
cfg_fit = dict(min_inlier_frac=rel['model_min_inliers'], max_rms=rel['model_max_rms'], wide_start=True)
intr = json.load(open(D + '/intrinsics.json'))
n = 0
for cam in ('left', 'right'):
    held = ev.OTHER[cam]; c = ev.LETTER[cam]
    k = Intrinsics(fx=intr[c][0], fy=intr[c][1], cx=intr[c][2], cy=intr[c][3])
    ce = (Rotation.from_quat(base_cfg[cam]['cam_in_ee_quat']).as_matrix(), np.array(base_cfg[cam]['cam_in_ee_xyz'], float))
    model = ev.make_model(base_cfg, real, held); det_own, det_partner = ev.make_detectors(real, low, cam)
    fr = ev.load_frames(D, c)
    for i, (t, d) in enumerate(fr[::max(1, len(fr) // NF)]):
        th = ev.tool_pose(d[f'pose_{held}']); tc = ev.tool_pose(d[f'pose_{cam}']); bc = mv.camera_to_base(tc, *ce)
        near, far = ev.split_masks(d, k, ce, rel)
        own = det_own.detect(d['color'], d['depth'], None, exclude=far)
        det = det_partner.detect(d['color'], d['depth'], None, exclude=cv2.dilate(own.mask, np.ones((9, 9), np.uint8)) if own.ok else near)
        if not det.ok: continue
        fp = mf.fit_face_pose(det, d['depth'], 0.0001, k, model, np.linalg.inv(bc) @ ev.grasp_face(th, rel), info={}, rolls_deg=(0.0,), latch_both_fits=True, **cfg_fit)
        fit = np.eye(4) * np.nan
        if fp is not None: fit = np.eye(4); fit[:3, :3] = fp.rotation; fit[:3, 3] = fp.position
        np.savez_compressed(f'{out}/{tag}_{c}_{i:03d}.npz', color=d['color'], depth=d['depth'].astype(np.float32) * 1e-4, mask=det.mask > 0,
                            K=np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1.0]]), t=t, held=held, cam=cam,
                            tool_held=th, tool_cam=tc, cam_in_base=bc, fit=fit)
        n += 1
print('exported', n, 'frames to', out)
