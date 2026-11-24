"""Which side the latch is on, from the depth where the two half-turn poses differ (2026-11-24).

FoundationPose's scores for the answer and for the same connector turned half a turn about
its mating axis are close (real robot, 2026-11-18: 0.16-0.23 apart, for right and wrong
answers alike): the housing looks nearly the same both ways, and the score looks at all of it.
Here only the pixels where the two poses render differently are compared with the measured
depth: the latch and whatever else is not symmetric. For each turn, FoundationPose's own best
refined hypothesis with that turn (else the answer turned exactly), so each side gets its
best fit before the comparison.

Per pixel of that set, with measured depth: the error of a pose is
    rendered there:      |measured - rendered|, clipped at CAP
    not rendered there:  CAP if the pixel is in the connector's mask, else 0
pixels where the measurement is far IN FRONT of both renders (a finger, the cable) are left
out. The answer: the mean error [mm] at turns 0, 90, 180, -90 from the answer (nan: not
computed; only 0 and 180) and the number of pixels compared.

Runs inside the FoundationPose environment (fp_server.py).
"""
import numpy as np
import torch
from Utils import nvdiffrast_render

CAP = 0.003           # [m] error of one pixel at most
DIFFER = 0.0005       # [m] the two renders differ by more: the pixel is compared
OCCLUDED = 0.003      # [m] measured this much in front of both renders: something else


def _rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0, 0], [np.sin(a), np.cos(a), 0, 0],
                     [0, 0, 1, 0], [0, 0, 0, 1.0]])


def candidates(e, face, mesh_from_face, turns=(0.0, 180.0)):
    """Centered-mesh poses (FoundationPose's frame) for each turn from `face`: its best
    hypothesis with that turn, else the answer turned exactly about the face axis."""
    tf = e.get_tf_to_centered_mesh().data.cpu().numpy().astype(float)
    poses = e.poses.data.cpu().numpy().astype(float).reshape(-1, 4, 4)
    scores = e.scores.data.cpu().numpy().astype(float).ravel()
    face_from_mesh = np.linalg.inv(mesh_from_face)
    out = []
    for turn in turns:
        best, best_s = None, -np.inf
        for pose, s in zip(poses, scores):
            f = pose @ tf @ mesh_from_face
            rel = face[:3, :3].T @ f[:3, :3]
            if rel[2, 2] < np.cos(np.radians(20.0)):
                continue
            ang = np.degrees(np.arctan2(rel[1, 0], rel[0, 0]))
            if abs((ang - turn + 180.0) % 360.0 - 180.0) <= 45.0 and s > best_s:
                best, best_s = pose, s
        if best is None:      # the answer turned: face' = face Rz -> mesh pose
            best = face @ _rz(turn) @ face_from_mesh @ np.linalg.inv(tf)
        out.append(best)
    return np.array(out)


def check(e, face, mesh_from_face, K, depth, mask, glctx):
    """(depth errors [mm], outline disagreement [0..1]; each at 0/90/180/-90 deg from the
    answer, nan where not computed; pixels compared by depth)"""
    poses = candidates(e, face, mesh_from_face)
    H, W = depth.shape[:2]
    _, rd, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=torch.as_tensor(
        poses, device='cuda', dtype=torch.float), glctx=glctx, mesh_tensors=e.mesh_tensors)
    rd = rd.data.cpu().numpy().astype(np.float64)          # (2, H, W), 0 where nothing
    meas = np.asarray(depth, np.float64)
    mask = np.asarray(mask, bool)
    on = rd > 0.001
    differ = (on[0] != on[1]) | (on[0] & on[1] & (np.abs(rd[0] - rd[1]) > DIFFER))
    valid = meas > 0.001
    front = np.where(on, rd, np.inf).min(0)
    sel = differ & valid & ~(meas < front - OCCLUDED)
    n = int(sel.sum())
    err = np.full(4, np.nan)
    sil = np.full(4, np.nan)
    for i, j in ((0, 0), (1, 2)):
        if n:
            e_px = np.where(on[i], np.minimum(np.abs(meas - rd[i]), CAP),
                            np.where(mask, CAP, 0.0))
            err[j] = float(e_px[sel].mean()) * 1000.0
        # the outline: where the two renders' silhouettes differ, the share of pixels where
        # this one disagrees with the mask
        sd = on[0] != on[1]
        if sd.any():
            sil[j] = float((on[i][sd] != mask[sd]).mean())
    return err, sil, n
