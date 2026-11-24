"""Recorded frames: current fit vs FoundationPose alone vs FoundationPose as the start of the fit (ROS Python; needs fp_server)."""
import glob, sys, time
import numpy as np
from scipy.spatial.transform import Rotation as R
from garment_connector import multiview_eval as ev, model_fit as mf, fp_client
from garment_connector.detector import Detection
from garment_connector.vs_control import Intrinsics
W = sys.argv[1]
base_cfg = ev.load_config('connector_mating.yaml'); real = ev.load_config('connector_mating_real.yaml'); rel = real['relative']
cfg_fit = dict(min_inlier_frac=rel['model_min_inliers'], max_rms=rel['model_max_rms'], wide_start=True)
MODEL = {s: ev.make_model(base_cfg, real, s) for s in ('left', 'right')}
def stats(Ts):
    P = np.array([T[:3, 3] for T in Ts]) * 1000; m = R.from_matrix([T[:3, :3] for T in Ts]).mean()
    a = np.array([np.degrees((m.inv() * R.from_matrix(T[:3, :3])).magnitude()) for T in Ts])
    tilt = np.array([np.degrees(np.arccos(np.clip(abs(T[2, 2]), 0, 1))) for T in Ts])
    return f'n {len(Ts):2d} | position std {P.std(0).round(2)} mm | orientation median {np.median(a):4.1f}, 90% {np.percentile(a, 90):5.1f}, worst {a.max():5.1f} deg, over 5 deg {int((a > 5).sum()):2d} | tilt from the tool axis {tilt.min():.1f}..{tilt.max():.1f} deg'
tt = []
for tag in ('c', 'g'):
    for c, name in (('l', 'male (right hand), left camera  '), ('r', 'female (left hand), right camera')):
        CUR, FP, BOTH, miss, nofit = [], [], [], 0, 0
        for f in sorted(glob.glob(f'{W}/inputs/{tag}_{c}_*.npz')):
            d = np.load(f); held = str(d['held']); K = d['K']; k = Intrinsics(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
            g = np.linalg.inv(d['tool_held']) @ d['cam_in_base']
            prior = np.linalg.inv(d['cam_in_base']) @ ev.grasp_face(d['tool_held'], rel)
            t0 = time.time(); face = fp_client.request(d['color'], d['depth'], d['mask'], K, held); tt.append(time.time() - t0)
            if face is None: miss += 1; continue
            face = fp_client.nearest_quarter_turn(face, prior)
            det = Detection(ok=True, mask=d['mask'].astype(np.uint8) * 255)
            fp = mf.fit_face_pose(det, d['depth'], 1.0, k, MODEL[held], face, info={}, rolls_deg=(0.0,), **cfg_fit)
            if not np.isnan(d['fit']).any(): CUR.append(g @ d['fit'])
            FP.append(g @ face)
            if fp is None: nofit += 1; continue
            e = np.eye(4); e[:3, :3] = fp.rotation; e[:3, 3] = fp.position; BOTH.append(g @ e)
        print(f'10-09{tag} {name} (no answer {miss}, fit refused after FoundationPose {nofit})')
        for label, Ts in (('current fit              ', CUR), ('FoundationPose alone     ', FP), ('FoundationPose -> the fit', BOTH)):
            if len(Ts) > 2: print('   ', label, stats(Ts))
print(f'round trip {np.median(tt):.2f} s median, {max(tt):.2f} s worst')
