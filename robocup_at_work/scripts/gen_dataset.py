#!/usr/bin/env python3
"""Generate a YOLO dataset from the sim using gazebo ground truth.

Projects each object's 3D box corners into the camera, and drops boxes
that are mostly occluded. Camera TF must be from base_footprint (the
model root), not base_link.

Needs gazebo up (robot_gazebo.launch.py gui:=false):
    python3 scripts/gen_dataset.py --samples 1500
    python3 scripts/gen_dataset.py --samples 20 --debug
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from gazebo_msgs.srv import SetEntityState, GetEntityState
import tf2_ros

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(PKG, 'config', 'objects_manifest.json')
ENTITY = 'robocup_robot'

# approach poses, validated with nav2
APPROACH = {'WS01': (-2.00,  1.75,  1.5708),
            'WS02': (-2.50,  0.00,  3.1416),
            'WS03': ( 2.00,  1.75,  1.5708),
            'WS04': ( 2.50, -1.50,  0.0000)}

# object area per table (xmin, xmax, ymin, ymax), front half for arm reach
BANDS = {'WS01': (-2.25, -1.85,  2.32,  2.45),
         'WS02': (-3.25, -3.10, -0.25,  0.20),
         'WS03': ( 1.75,  2.15,  2.32,  2.45),
         'WS04': ( 3.10,  3.25, -1.65, -1.25)}

PARK = (20.0, 20.0)        # outside the arena
MIN_SEP = 0.11             # min centre distance
BASE_JITTER_XY = 0.06
BASE_JITTER_YAW = 0.12
MIN_VISIBLE = 0.35         # min visible fraction
MIN_BOX_PX = 8
SETTLE = 0.6


# -------------------------------------------------------------------- sim ---
# uses the gazebo_ros_state services, the gz CLI is ~0.5 s per call
# and hangs when called in parallel
def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def yaw_from_quat(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rpy_from_quat(q):
    x, y, z, w = q
    sp = 2 * (w * y - z * x)
    sp = max(-1.0, min(1.0, sp))
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(sp),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


# --------------------------------------------------------------- geometry ---
def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr]])


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def box_corners(pose, bbox):
    """8 corners of the object's 3D box in world frame."""
    hx, hy, hz = (d / 2.0 for d in bbox)
    local = np.array([[sx * hx, sy * hy, sz * hz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    R = rpy_to_mat(pose[3], pose[4], pose[5])
    return (R @ local.T).T + np.array(pose[:3])


class Sim(Node):
    def __init__(self):
        super().__init__('dataset_generator')
        self.frame = None
        self.info = None
        self.create_subscription(Image, '/rgb_camera/image_raw', self._img, 1)
        self.create_subscription(CameraInfo, '/rgb_camera/camera_info',
                                 self._inf, 1)
        self.buf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.buf, self)
        self.set_cli = self.create_client(SetEntityState,
                                          '/gazebo/set_entity_state')
        self.get_cli = self.create_client(GetEntityState,
                                          '/gazebo/get_entity_state')

    def wait_services(self, timeout=20.0):
        return (self.set_cli.wait_for_service(timeout_sec=timeout)
                and self.get_cli.wait_for_service(timeout_sec=timeout))

    def _call(self, cli, req, timeout=5.0):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result() if fut.done() else None

    def set_pose(self, name, x, y, z, yaw):
        req = SetEntityState.Request()
        st = req.state
        st.name = name
        st.pose.position.x, st.pose.position.y, st.pose.position.z = x, y, z
        qx, qy, qz, qw = quat_from_yaw(yaw)
        (st.pose.orientation.x, st.pose.orientation.y,
         st.pose.orientation.z, st.pose.orientation.w) = qx, qy, qz, qw
        # zero the twist so the model doesn't drift
        st.twist.linear.x = st.twist.linear.y = st.twist.linear.z = 0.0
        st.twist.angular.x = st.twist.angular.y = st.twist.angular.z = 0.0
        st.reference_frame = 'world'
        r = self._call(self.set_cli, req)
        return bool(r and r.success)

    def get_pose(self, name):
        """Return (x, y, z, roll, pitch, yaw) in world, or None."""
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = 'world'
        r = self._call(self.get_cli, req)
        if not r or not r.success:
            return None
        p, o = r.state.pose.position, r.state.pose.orientation
        roll, pitch, yaw = rpy_from_quat((o.x, o.y, o.z, o.w))
        return (p.x, p.y, p.z, roll, pitch, yaw)

    def _img(self, m):
        self.frame = m

    def _inf(self, m):
        self.info = m

    def grab(self, timeout=10.0):
        self.frame = None
        end = time.time() + timeout
        while time.time() < end and self.frame is None:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.frame

    def wait_info(self, timeout=30.0):
        end = time.time() + timeout
        while time.time() < end and self.info is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.info

    def footprint_to_optical(self, timeout=15.0):
        """base_footprint -> camera_rgb_optical_frame."""
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.2)
            try:
                tf = self.buf.lookup_transform('base_footprint',
                                               'camera_rgb_optical_frame',
                                               rclpy.time.Time())
            except Exception:
                continue
            tr, ro = tf.transform.translation, tf.transform.rotation
            return (quat_to_mat((ro.x, ro.y, ro.z, ro.w)),
                    np.array([tr.x, tr.y, tr.z]))
        return None


def img_to_bgr(msg):
    a = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    return a[:, :, ::-1].copy() if msg.encoding == 'rgb8' else a.copy()


# ----------------------------------------------------------------- labels ---
def boxes_for_frame(objects, R_wc, t_wc, K, size, cv2):
    """objects: list of (cls_idx, pose, bbox). Returns list of YOLO rows."""
    fx, fy, cx, cy = K
    W, H = size

    projected = []
    for idx, pose, bbox in objects:
        pts_w = box_corners(pose, bbox)
        pts_c = (pts_w - t_wc) @ R_wc            # world -> optical
        if np.any(pts_c[:, 2] <= 0.01):          # behind the camera
            continue
        u = fx * pts_c[:, 0] / pts_c[:, 2] + cx
        v = fy * pts_c[:, 1] / pts_c[:, 2] + cy
        projected.append(dict(idx=idx, u=u, v=v,
                              depth=float(np.mean(pts_c[:, 2]))))

    # paint far to near, visible pixels / full area = visibility
    mask = np.full((H, W), -1, np.int32)
    order = sorted(range(len(projected)), key=lambda i: -projected[i]['depth'])
    hulls, areas = {}, {}
    for i in order:
        p = projected[i]
        pts = np.stack([p['u'], p['v']], 1).astype(np.int32)
        hull = cv2.convexHull(pts)
        hulls[i] = hull
        solo = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(solo, hull, 1)
        areas[i] = int(solo.sum())
        cv2.fillConvexPoly(mask, hull, i)

    rows, kept = [], []
    for i, p in enumerate(projected):
        if areas.get(i, 0) == 0:
            continue
        visible = int((mask == i).sum()) / float(areas[i])
        if visible < MIN_VISIBLE:
            continue
        x0, x1 = float(np.min(p['u'])), float(np.max(p['u']))
        y0, y1 = float(np.min(p['v'])), float(np.max(p['v']))
        x0, x1 = max(0.0, x0), min(W - 1.0, x1)      # clip to the image
        y0, y1 = max(0.0, y0), min(H - 1.0, y1)
        w, h = x1 - x0, y1 - y0
        if w < MIN_BOX_PX or h < MIN_BOX_PX:
            continue
        rows.append('%d %.6f %.6f %.6f %.6f'
                    % (p['idx'], ((x0 + x1) / 2) / W, ((y0 + y1) / 2) / H,
                       w / W, h / H))
        kept.append((p['idx'], x0, y0, x1, y1, visible))
    return rows, kept


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=800)
    ap.add_argument('--out', default=os.path.join(PKG, 'dataset'))
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--debug', action='store_true',
                    help='also write overlay images to <out>/debug')
    args = ap.parse_args()
    random.seed(args.seed)

    import cv2

    with open(MANIFEST) as f:
        man = json.load(f)
    classes = sorted(c['name'] for c in man['classes'])
    cls_idx = {n: i for i, n in enumerate(classes)}
    by_name = {c['name']: c for c in man['classes']}
    # all object instances
    pool = [(i['name'], i['cls']) for i in man['instances']]

    rclpy.init()
    sim = Sim()
    if sim.wait_info() is None:
        sys.exit('no /rgb_camera/camera_info - is gazebo running?')
    if sim.grab(timeout=30.0) is None:
        sys.exit('no /rgb_camera/image_raw - is the robot spawned?')
    if not sim.wait_services():
        sys.exit('no /gazebo/set_entity_state - the world is missing the\n'
                 'libgazebo_ros_state.so plugin, or gazebo predates the edit '
                 'that added it. Restart gazebo after rebuilding.')
    ext = sim.footprint_to_optical()
    if ext is None:
        sys.exit('no TF base_footprint -> camera_rgb_optical_frame')
    R_bc, t_bc = ext
    K = (sim.info.k[0], sim.info.k[4], sim.info.k[2], sim.info.k[5])
    size = (sim.info.width, sim.info.height)
    print('camera %dx%d fx=%.1f cx=%.1f cy=%.1f' % (size[0], size[1], K[0], K[2], K[3]))
    print('classes: %s' % ', '.join('%d=%s' % (i, n) for i, n in enumerate(classes)))

    for split in ('train', 'val'):
        for kind in ('images', 'labels'):
            d = os.path.join(args.out, kind, split)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
    if args.debug:
        dbg = os.path.join(args.out, 'debug')
        shutil.rmtree(dbg, ignore_errors=True)
        os.makedirs(dbg)

    # park everything, each sample brings back a subset
    for name, _ in pool:
        sim.set_pose(name, PARK[0], PARK[1], 0.2, 0.0)
    time.sleep(1.0)

    n_boxes = 0
    per_class = {n: 0 for n in classes}
    t_start = time.time()
    staged = []

    for s in range(args.samples):
        ws = random.choice(list(APPROACH))
        xmin, xmax, ymin, ymax = BANDS[ws]

        # -------- random objects on this table
        batch = [(name, PARK[0], PARK[1], 0.2, 0.0) for name, _ in staged]
        k = random.randint(1, 4)
        staged = random.sample(pool, k)
        placed = []
        for name, cname in staged:
            spec = by_name[cname]
            for _ in range(30):
                x = random.uniform(xmin, xmax)
                y = random.uniform(ymin, ymax)
                if all(math.hypot(x - px, y - py) > MIN_SEP for px, py in placed):
                    break
            placed.append((x, y))
            batch.append((name, x, y, man['table_top_z'] + spec['rest_z'],
                          random.uniform(-math.pi, math.pi)))

        # -------- jitter the base around the approach pose
        ax, ay, ayaw = APPROACH[ws]
        bx = ax + random.uniform(-BASE_JITTER_XY, BASE_JITTER_XY)
        by = ay + random.uniform(-BASE_JITTER_XY, BASE_JITTER_XY)
        byaw = ayaw + random.uniform(-BASE_JITTER_YAW, BASE_JITTER_YAW)
        batch.append((ENTITY, bx, by, 0.05, byaw))

        for name, x, y, z, yaw in batch:
            sim.set_pose(name, x, y, z, yaw)
        time.sleep(SETTLE)

        # -------- labels use the settled poses
        poses = {n: sim.get_pose(n) for n in [ENTITY] + [n for n, _ in staged]}
        bp = poses[ENTITY]
        if bp is None:
            continue
        R_wb = rpy_to_mat(0.0, 0.0, bp[5])
        R_wc = R_wb @ R_bc
        t_wc = np.array(bp[:3]) + R_wb @ t_bc

        objs = []
        for name, cname in staged:
            p = poses[name]
            if p is None or p[2] < 0.0:
                continue
            objs.append((cls_idx[cname], p, by_name[cname]['bbox']))

        frame = sim.grab()
        if frame is None:
            continue
        img = img_to_bgr(frame)
        rows, kept = boxes_for_frame(objs, R_wc, t_wc, K, size, cv2)
        if not rows:
            continue

        split = 'val' if random.random() < args.val_frac else 'train'
        stem = '%s_%05d' % (ws, s)
        cv2.imwrite(os.path.join(args.out, 'images', split, stem + '.png'), img)
        with open(os.path.join(args.out, 'labels', split, stem + '.txt'), 'w') as f:
            f.write('\n'.join(rows) + '\n')
        n_boxes += len(rows)
        for idx, *_ in kept:
            per_class[classes[idx]] += 1

        if args.debug:
            ov = img.copy()
            for idx, x0, y0, x1, y1, vis in kept:
                cv2.rectangle(ov, (int(x0), int(y0)), (int(x1), int(y1)),
                              (0, 255, 0), 1)
                cv2.putText(ov, '%s %.2f' % (classes[idx], vis),
                            (int(x0), int(y0) - 3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (0, 255, 255), 1)
            cv2.imwrite(os.path.join(args.out, 'debug', stem + '.png'), ov)

        if (s + 1) % 25 == 0:
            el = time.time() - t_start
            print('  %4d/%d  %d boxes  %.2f s/sample  eta %.1f min'
                  % (s + 1, args.samples, n_boxes, el / (s + 1),
                     (args.samples - s - 1) * el / (s + 1) / 60))

    with open(os.path.join(args.out, 'data.yaml'), 'w') as f:
        f.write('# written by gen_dataset.py\n')
        f.write('path: %s\n' % os.path.abspath(args.out))
        f.write('train: images/train\nval: images/val\n')
        f.write('nc: %d\n' % len(classes))
        f.write('names: [%s]\n' % ', '.join(classes))

    ntr = len(os.listdir(os.path.join(args.out, 'images', 'train')))
    nva = len(os.listdir(os.path.join(args.out, 'images', 'val')))
    print('\ndone: %d train / %d val images, %d boxes in %.1f min'
          % (ntr, nva, n_boxes, (time.time() - t_start) / 60))
    print('per class: %s' % ', '.join('%s=%d' % (k, v)
                                      for k, v in sorted(per_class.items())))
    print('wrote %s' % os.path.join(args.out, 'data.yaml'))

    sim.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
