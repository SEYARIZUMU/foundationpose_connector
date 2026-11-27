"""FoundationPose tracking (fp_fast.track) on the recorded frames (inputs/*.npz, 2026-10-09).

The frames of one recording (c_l, c_r, g_l, g_r) are 6-10 s apart and the arms move between
them (up to 25 mm), so each start is the previous answer moved with the arms: the connector
stays fixed in the holding arm's tool frame and the camera moves with the other arm,
    start = cam_now^-1 tool_now tool_before^-1 cam_before  answer_before
(plain: the previous answer as it is, for comparison). Compared with the full search's answer
of the frame (out/<name>_pose.npy). Pairs whose two full-search answers differ by more than
45 deg (the latch side flipped there) are left out.
  from ref: start from the previous frame's full-search answer
  chain:    start from the previous TRACKED answer (the first one from the full search): drift

  ~/foundationpose/miniforge3/bin/conda run --no-capture-output -n foundationpose \\
      python fp_track_bench.py [iterations ...]
"""
import glob, os, sys, time
import numpy as np
import fp_fast
from fp_bench import angle

W = os.path.dirname(os.path.abspath(__file__))


def moved(a, b):
    """Transform taking a pose in the camera of frame a to the camera of frame b."""
    return np.linalg.inv(b['cam_in_base']) @ b['tool_held'] @ np.linalg.inv(a['tool_held']) \
        @ a['cam_in_base']


def summary(label, rows):
    if not rows:
        return
    t, ang, pos = map(np.array, zip(*rows))
    lost = ang >= 45.0
    print(f'  {label:26s} n {len(rows):3d} | time median {np.median(t) * 1000:.0f} ms | angle '
          f'median {np.median(ang):.2f}, 90% {np.percentile(ang, 90):.2f}, worst {ang.max():.1f} '
          f'deg | position median {np.median(pos):.2f}, 90% {np.percentile(pos, 90):.2f}, '
          f'worst {pos.max():.1f} mm | lost (>45 deg) {int(lost.sum())}', flush=True)


def main():
    iters = [int(a) for a in sys.argv[1:]] or [2]
    est, _, _ = fp_fast.load()
    seqs = {}
    for f in sorted(glob.glob(f'{W}/inputs/*.npz')):
        name = os.path.basename(f)[:-4]
        seqs.setdefault(name[:3], []).append((name, np.load(f),
                                              np.load(f'{W}/out/{name}_pose.npy')))
    move = [np.linalg.norm((moved(a[1], b[1]) @ a[2])[:3, 3] - a[2][:3, 3]) * 1000
            for s in seqs.values() for a, b in zip(s, s[1:])]
    print(f'how far the connector moved in the camera between frames: median '
          f'{np.median(move):.1f}, worst {max(move):.1f} mm', flush=True)
    d = seqs['c_l'][0]      # warm-up
    fp_fast.track(est[str(d[1]['held'])], d[1]['color'], d[1]['depth'], d[1]['K'], d[2])
    for it in iters:
        print(f'iteration {it}:')
        rows = {'from ref, moved with arms': [], 'from ref, as it is': [], 'chain, moved with arms': []}
        for seq in seqs.values():
            last = seq[0][2]
            for (na, a, ra), (nb, b, rb) in zip(seq, seq[1:]):
                e = est[str(b['held'])]
                if angle(ra[:3, :3], rb[:3, :3]) >= 45.0:
                    last = rb      # the full search flipped here: restart the chain from it
                    continue
                for label, start in (('from ref, moved with arms', moved(a, b) @ ra),
                                     ('from ref, as it is', ra),
                                     ('chain, moved with arms', moved(a, b) @ last)):
                    t0 = time.time()
                    p = fp_fast.track(e, b['color'], b['depth'], b['K'], start, iteration=it)
                    dt = time.time() - t0
                    rows[label].append((dt, angle(rb[:3, :3], p[:3, :3]),
                                        np.linalg.norm(rb[:3, 3] - p[:3, 3]) * 1000))
                    if label.startswith('chain'):
                        last = p
        for label, r in rows.items():
            summary(label, r)


if __name__ == '__main__':
    main()
