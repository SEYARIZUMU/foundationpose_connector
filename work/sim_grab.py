"""Save one frame of both simulated wrist cameras with the true partner pose (ROS Python, Gazebo only)."""
import sys, time
import cv2, numpy as np, rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener
from garment_connector.detector import hsv_mask
out, idx = sys.argv[1], sys.argv[2]
rclpy.init(); n = rclpy.create_node('sim_grab')
q = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
buf = Buffer(); TransformListener(buf, n); st = {}
for s in ('left', 'right'):
    n.create_subscription(Image, f'/{s}_wrist_camera/image', lambda m, s=s: st.__setitem__(s + 'c', np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3).copy()), q)
    n.create_subscription(Image, f'/{s}_wrist_camera/depth_image', lambda m, s=s: st.__setitem__(s + 'd', np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()), q)
    n.create_subscription(CameraInfo, f'/{s}_wrist_camera/camera_info', lambda m, s=s: st.__setitem__(s + 'k', np.array(m.k).reshape(3, 3)), 1)
def tf(frame):
    t = buf.lookup_transform('robot_base_link', frame, Time()).transform
    T = np.eye(4); T[:3, :3] = R.from_quat([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]).as_matrix(); T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]; return T
R_LO = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
t0 = time.time(); T = None
while time.time() - t0 < 25:
    rclpy.spin_once(n, timeout_sec=0.05)
    if len(st) == 6 and time.time() - t0 > 2.5:
        try: T = {f: tf(f) for f in ('L_sim_connector_link', 'R_sim_connector_link', 'wrist_l_color_frame', 'wrist_r_color_frame')}; break
        except Exception as e: err = e
if T is None: print('failed', sorted(st), err); sys.exit(1)
for cam, frame, partner, extra in (('left', 'wrist_l_color_frame', 'R_sim_connector_link', R.from_euler('x', 180, degrees=True).as_matrix()), ('right', 'wrist_r_color_frame', 'L_sim_connector_link', np.eye(3))):
    C = T[frame].copy(); C[:3, :3] = C[:3, :3] @ extra @ R_LO
    truth = np.linalg.inv(C) @ T[partner]; K = st[cam + 'k']
    depth = st[cam + 'd'].copy(); depth[~np.isfinite(depth)] = 0
    red = hsv_mask(st[cam + 'c'], [165, 80, 40], [6, 255, 255])
    num, lab, stats, cen = cv2.connectedComponentsWithStats(cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    u, v = K[0, 0] * truth[0, 3] / truth[2, 3] + K[0, 2], K[1, 1] * truth[1, 3] / truth[2, 3] + K[1, 2]
    best = min(range(1, num), key=lambda i: np.hypot(cen[i][0] - u, cen[i][1] - v) if stats[i, cv2.CC_STAT_AREA] > 60 else 1e9)
    mask = lab == best
    np.savez_compressed(f'{out}/sim_{cam[0]}_{idx}.npz', color=st[cam + 'c'], depth=depth.astype(np.float32), mask=mask, K=K, truth=truth, held='right' if cam == 'left' else 'left', cam=cam)
    print(cam, 'camera: partner at', (truth[:3, 3] * 1000).round(0), 'mm, mask', int(mask.sum()), 'px, blob', np.round(cen[best]), 'expected', round(u), round(v))
rclpy.shutdown()
