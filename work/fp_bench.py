"""Speed vs answer of fp_fast.py settings on the recorded frames (inputs/*.npz, 2026-10-09).

The reference is out/<name>_pose.npy (register with 5 passes, run_fp.py). Per setting: time
per frame, and how far each answer is from the reference (angle, position, and frames whose
latch side turned by 90 deg or more about the mating axis).

  cd ~/foundationpose/work
  ~/foundationpose/miniforge3/bin/conda run --no-capture-output -n foundationpose \
      python fp_bench.py iteration=5 cone_deg=45 cone_deg=35
Each argument is one setting: comma-separated Opts fields, e.g. iteration=3,cone_deg=45.
The nominal axis per frame: the tool axis of the holding arm in the camera.
"""
import glob, os, sys, time
import numpy as np
import fp_fast

W = os.path.dirname(os.path.abspath(__file__))


def parse(arg):
    kw = {}
    for item in filter(None, arg.split(',')):
        k, v = item.split('=')
        kw[k] = type(getattr(fp_fast.Opts(), k))(v)
    return fp_fast.Opts(**kw)


def angle(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def nominal_axis(d):
    """Tool z of the holding arm in the camera (= the nominal face z, multiview_eval.grasp_face)."""
    return np.linalg.inv(d['cam_in_base'])[:3, :3] @ d['tool_held'][:3, 2]


# The latch side as the 2026-10-09 recordings were really held (checked in the images): the
# right arm's connector (male) half a turn from the nominal grasp, the left one's as nominal.
TRUE_TURN = {'right': 180.0, 'left': 0.0}


def latch_right(d, face):
    """Is the answer's latch side (turn about the axis from the nominal grasp, nearest 90 deg)
    the true one?"""
    nom = np.linalg.inv(d['cam_in_base'])[:3, :3] @ d['tool_held'][:3, :3] @ fp_fast.RZ90
    rel = nom.T @ face[:3, :3]
    turn = round(np.degrees(np.arctan2(rel[1, 0], rel[0, 0])) / 90.0) * 90.0 % 360.0
    return turn == TRUE_TURN[str(d['held'])] and rel[2, 2] > np.cos(np.radians(45.0))


def main():
    settings = [parse(a) for a in sys.argv[1:]] or [fp_fast.Opts()]
    est, mesh_from_face, _ = fp_fast.load()
    files = sorted(glob.glob(f'{W}/inputs/*.npz'))
    data = [(os.path.basename(f)[:-4], np.load(f)) for f in files]
    d = data[0][1]          # warm-up (CUDA kernels, first render)
    fp_fast.estimate(est[str(d['held'])], d['color'], d['depth'], d['mask'], d['K'], fp_fast.Opts())
    off = np.array([np.degrees(np.arccos(np.clip(
        np.load(f'{W}/out/{n}_pose.npy')[:3, 2] @ nominal_axis(d), -1, 1))) for n, d in data])
    print(f'reference axis vs nominal: median {np.median(off):.1f}, 90% {np.percentile(off, 90):.1f}, '
          f'worst {off.max():.1f} deg', flush=True)
    for opts in settings:
        tt, ang, pos, turned, nh = [], [], [], [], []
        right = {'right': [], 'left': []}
        for name, d in data:
            side = str(d['held'])
            t0 = time.time()
            pose = fp_fast.estimate(est[side], d['color'], d['depth'], d['mask'], d['K'], opts,
                                     axis=nominal_axis(d))
            tt.append(time.time() - t0)
            ref = np.load(f'{W}/out/{name}_pose.npy')
            ang.append(angle(ref[:3, :3], pose[:3, :3]))
            pos.append(np.linalg.norm(ref[:3, 3] - pose[:3, 3]) * 1000)
            fr, fn = ref @ mesh_from_face[side], pose @ mesh_from_face[side]
            rel = fr[:3, :3].T @ fn[:3, :3]        # in the reference face frame
            roll = np.degrees(np.arctan2(rel[1, 0], rel[0, 0]))
            if ang[-1] >= 45.0:     # another answer: a turn about the axis, else another axis
                turned.append(f'{name} {roll:+.0f}' if rel[2, 2] > np.cos(np.radians(45.0)) else f'{name} axis')
            nh.append(len(est[side].poses))
            right[side].append(latch_right(d, fn))
        tt, ang, pos = map(np.array, (tt, ang, pos))
        ok = ang < 45.0         # same answer up to refinement (not another turn)
        print(f'{opts}: latch right: male (right arm) {sum(right["right"])}/{len(right["right"])}, '
              f'female {sum(right["left"])}/{len(right["left"])} | {int(np.median(nh))} hypotheses, time median {np.median(tt):.3f} s, worst {tt.max():.3f} s | '
              f'vs reference, {ok.sum()} same answer: angle median {np.median(ang[ok]):.2f}, '
              f'90% {np.percentile(ang[ok], 90):.2f}, worst {ang[ok].max():.2f} deg; position '
              f'median {np.median(pos[ok]):.2f}, 90% {np.percentile(pos[ok], 90):.2f}, worst '
              f'{pos[ok].max():.2f} mm | turned {len(turned)}: {", ".join(turned)}', flush=True)


if __name__ == '__main__':
    main()
