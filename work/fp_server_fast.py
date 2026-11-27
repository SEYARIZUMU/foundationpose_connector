"""fp_server.py with the speed-ups of fp_fast.py (2026-11-26). Same protocol and port, so the
node works with either; run one of them at a time.

  cd ~/foundationpose/work
  FP_ITER=3 ~/foundationpose/miniforge3/bin/conda run --no-capture-output -n foundationpose python fp_server_fast.py

Settings (environment): FP_ITER refinement passes (5 = as fp_server.py); FP_CONE hypotheses
only within this angle [deg] of the request's 'axis' (nominal mating axis in the camera; 0 or
no 'axis' in the request = all, as fp_server.py); FP_TRACK_ITER refinement passes when tracking.

FP_MESH=sim: Gazebo's box connectors (left plug, right receptacle) instead of the Molex CAD.

A request with 'init' (4x4, the FACE pose in the camera to start from: the last answer moved
with the arms) is TRACKED (fp_fast.track, ~30 ms) instead of searched: the answer has
tracked=True, turn scores nan and no latch check (the latch side is decided before).
"""
import io, os, socket, struct, sys, time
import numpy as np
import fp_fast, latch_depth

PORT = int(os.environ.get('FP_PORT', 47993))


def recv(conn, n):
    buf = b''
    while len(buf) < n:
        part = conn.recv(n - len(buf))
        if not part:
            raise ConnectionError('closed')
        buf += part
    return buf


def turn_scores(e, face, mesh_from_face):
    """FoundationPose's best score among its hypotheses turned 0, 90, 180 and -90 deg about
    the mating axis from its answer `face` (nan: none of them is turned so). The connector
    looks nearly the same a half turn apart: how far the second is behind the first says how
    much the latch side of this answer can be trusted."""
    best = np.full(4, np.nan)
    try:
        poses = (e.poses @ e.get_tf_to_centered_mesh()).data.cpu().numpy().astype(float)
        scores = e.scores.data.cpu().numpy().astype(float).ravel()
        for pose, score in zip(poses.reshape(-1, 4, 4), scores):
            rel = face[:3, :3].T @ (pose @ mesh_from_face)[:3, :3]
            if rel[2, 2] < np.cos(np.radians(20.0)):       # another axis: not a turn
                continue
            i = int(round(np.degrees(np.arctan2(rel[1, 0], rel[0, 0])) / 90.0)) % 4
            if not score <= best[i]:
                best[i] = score
    except Exception as err:        # the pose is still answered
        print('turn scores:', repr(err)[:200], flush=True)
    return best


def main():
    opts = fp_fast.Opts.from_env()
    est, mesh_from_face, glctx = fp_fast.load()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', PORT))
    srv.listen(4)
    print(f'FoundationPose (fast) ready on 127.0.0.1:{PORT}, {opts}, meshes '
          f'{os.environ.get("FP_MESH", "real")}', flush=True)
    while True:
        conn, _ = srv.accept()
        try:
            n = struct.unpack('<Q', recv(conn, 8))[0]
            d = np.load(io.BytesIO(recv(conn, n)), allow_pickle=False)
            side = str(d['side'])
            t0 = time.time()
            if 'init' in d.files:
                pose = fp_fast.track(est[side], d['color'], d['depth'], d['K'],
                                     np.asarray(d['init'], float)
                                     @ np.linalg.inv(mesh_from_face[side]), opts.track_iter)
                face = np.asarray(pose, float) @ mesh_from_face[side]
                out = io.BytesIO()
                np.savez(out, ok=True, face=face, seconds=time.time() - t0, tracked=True,
                         turn_scores=np.full(4, np.nan))
                print(f'{side}: {time.time() - t0:.3f} s tracked', flush=True)
            else:
                pose = fp_fast.estimate(est[side], d['color'], d['depth'], d['mask'], d['K'], opts,
                                       axis=d['axis'] if 'axis' in d.files else None)
                face = np.asarray(pose, float) @ mesh_from_face[side]
                out = io.BytesIO()
                ts = turn_scores(est[side], face, mesh_from_face[side])
                # which side the latch is on, from the outline / depth where the answer and the
                # half-turned one differ (latch_depth.py); face_half: the best hypothesis
                # half a turn from the answer
                ld_depth, ld_sil, face_half = np.full(4, np.nan), np.full(4, np.nan), face
                try:
                    ld_depth, ld_sil, _ = latch_depth.check(
                        est[side], face, mesh_from_face[side], d['K'].astype(float),
                        d['depth'].astype(np.float32), d['mask'].astype(bool), glctx)
                    tf = est[side].get_tf_to_centered_mesh().data.cpu().numpy().astype(float)
                    face_half = latch_depth.candidates(
                        est[side], face, mesh_from_face[side], turns=(180.0,))[0] @ tf \
                        @ mesh_from_face[side]
                except Exception as err:
                    print('latch check:', repr(err)[:200], flush=True)
                np.savez(out, ok=True, face=face, seconds=time.time() - t0, turn_scores=ts,
                         latch_depth=ld_depth, latch_outline=ld_sil, face_half=face_half)
                print(f'{side}: {time.time() - t0:.2f} s, score at 0/90/180/-90 deg about the '
                      f'axis: ' + ' '.join(f'{x:.3f}' for x in ts)
                      + f', outline 0/180 {ld_sil[0]:.2f}/{ld_sil[2]:.2f}'
                      + f', depth 0/180 {ld_depth[0]:.2f}/{ld_depth[2]:.2f} mm', flush=True)
        except Exception as e:      # answer, and keep serving
            out = io.BytesIO()
            np.savez(out, ok=False, face=np.eye(4), seconds=0.0, error=repr(e)[:300])
            print('error:', repr(e)[:200], flush=True)
        try:
            b = out.getvalue()
            conn.sendall(struct.pack('<Q', len(b)) + b)
        finally:
            conn.close()


if __name__ == '__main__':
    main()
