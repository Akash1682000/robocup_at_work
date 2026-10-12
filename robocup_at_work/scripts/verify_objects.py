#!/usr/bin/env python3
"""Check the objects stay still on the tables and are visible to the camera.

Stability compares poses before and after a wait. Visibility projects each
object into the camera from every approach pose, reports pixel size and
saves annotated frames.

Needs gazebo up (robot_gazebo.launch.py gui:=false):

    python3 scripts/verify_objects.py
    python3 scripts/verify_objects.py --settle 30
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(PKG, 'config', 'objects_manifest.json')
ENTITY = 'robocup_robot'
OUT_DIR = os.environ.get('VERIFY_OUT', '/tmp/robocup_object_frames')

# approach poses, validated with nav2
POSES = [('WS01', -2.00,  1.75,  1.5708),
         ('WS02', -2.50,  0.00,  3.1416),
         ('WS03',  2.00,  1.75,  1.5708),
         ('WS04',  2.50, -1.50,  0.0000)]


# ----------------------------------------------------------------- gazebo ---
def read_pose(name):
    """Return (x, y, z, roll, pitch, yaw) for a Gazebo model, or None."""
    r = subprocess.run(['gz', 'model', '-m', name, '-p'],
                       check=False, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        v = [float(t) for t in r.stdout.split()]
    except ValueError:
        return None
    return tuple(v[:6]) if len(v) >= 6 else None


def set_base(x, y, yaw):
    subprocess.run(['gz', 'model', '-m', ENTITY, '-x', str(x), '-y', str(y),
                    '-z', '0.05', '-R', '0', '-P', '0', '-Y', str(yaw)],
                   check=False, capture_output=True, text=True)


# --------------------------------------------------------------- geometry ---
def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Grabber(Node):
    def __init__(self):
        super().__init__('object_verifier')
        self.frame = None
        self.info = None
        self.create_subscription(Image, '/rgb_camera/image_raw', self._img, 1)
        self.create_subscription(CameraInfo, '/rgb_camera/camera_info',
                                 self._inf, 1)
        self.buf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.buf, self)

    def _img(self, m):
        self.frame = m

    def _inf(self, m):
        self.info = m

    def grab(self, timeout=10.0):
        self.frame = None
        end = time.time() + timeout
        while time.time() < end and self.frame is None:
            rclpy.spin_once(self, timeout_sec=0.2)
        return self.frame

    def base_to_optical(self, timeout=10.0):
        """base_footprint -> camera_rgb_optical_frame as (R, t).

        Must be base_footprint, gz reports the model root pose.
        """
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.2)
            try:
                tf = self.buf.lookup_transform(
                    'base_footprint', 'camera_rgb_optical_frame',
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


# -------------------------------------------------------------- stability ---
def check_stability(manifest, settle):
    print('=' * 72)
    print('A. STABILITY  (expected pose vs actual, then drift over %.0f s)' % settle)
    print('=' * 72)

    inst = manifest['instances']
    missing = [i['name'] for i in inst if read_pose(i['name']) is None]
    if missing:
        print('MISSING FROM GAZEBO: %s' % ', '.join(missing))
        print('  -> world not reloaded, or GAZEBO_MODEL_PATH is missing models/')
        return False

    first = {i['name']: read_pose(i['name']) for i in inst}
    print('settling %.0f s ...' % settle)
    time.sleep(settle)
    second = {i['name']: read_pose(i['name']) for i in inst}

    print('%-14s %-9s %9s %9s %9s  %s'
          % ('object', 'class', 'dz(mm)', 'spawn(mm)', 'drift(mm)', 'verdict'))
    ok = True
    for i in inst:
        n = i['name']
        want, a, b = i['pose'], first[n], second[n]
        dz = (b[2] - want['z']) * 1000.0
        spawn_err = math.hypot(a[0] - want['x'], a[1] - want['y']) * 1000.0
        drift = math.hypot(b[0] - a[0], b[1] - a[1]) * 1000.0
        bad = []
        if b[2] < manifest['table_top_z'] - 0.02:
            bad.append('FELL')
        if drift > 5.0:
            bad.append('DRIFTED')
        if abs(dz) > 15.0:
            bad.append('SANK/FLOATS')
        verdict = 'OK' if not bad else ' '.join(bad)
        ok &= not bad
        print('%-14s %-9s %9.1f %9.1f %9.1f  %s'
              % (n, i['cls'], dz, spawn_err, drift, verdict))
    print('-> %s\n' % ('all objects stable' if ok else 'INSTABILITY FOUND'))
    return ok


# ------------------------------------------------------------- visibility ---
def check_visibility(manifest, node, settle):
    print('=' * 72)
    print('B. VISIBILITY  (projection into the live RGB frame)')
    print('=' * 72)

    if node.info is None:
        for _ in range(50):
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.info is not None:
                break
    if node.info is None:
        print('no /rgb_camera/camera_info - is the robot spawned?')
        return False
    fx, fy = node.info.k[0], node.info.k[4]
    cx, cy = node.info.k[2], node.info.k[5]
    W, H = node.info.width, node.info.height
    print('camera %dx%d fx=%.1f fy=%.1f cx=%.1f cy=%.1f' % (W, H, fx, fy, cx, cy))

    ext = node.base_to_optical()
    if ext is None:
        print('no TF base_link -> camera_rgb_optical_frame')
        return False
    R_bc, t_bc = ext

    try:
        import cv2
        os.makedirs(OUT_DIR, exist_ok=True)
    except ImportError:
        cv2 = None

    bbox = {c['name']: max(c['bbox']) for c in manifest['classes']}
    all_ok = True

    for ws, x, y, yaw in POSES:
        set_base(x, y, yaw)
        time.sleep(3.0)
        bp = read_pose(ENTITY)
        if bp is None:
            print('%s: could not read base pose' % ws)
            all_ok = False
            continue
        # world <- base, then world <- optical
        R_wb = rot_z(bp[5])
        t_wb = np.array([bp[0], bp[1], bp[2]])
        R_wc = R_wb @ R_bc
        t_wc = t_wb + R_wb @ t_bc

        frame = node.grab()
        img = img_to_bgr(frame) if (frame is not None and cv2) else None

        here = [i for i in manifest['instances'] if i['workstation'] == ws]
        print('\n%s  base=(%.2f, %.2f, yaw %.3f)  %d objects'
              % (ws, bp[0], bp[1], bp[5], len(here)))
        print('  %-14s %-9s %7s %7s %7s %8s  %s'
              % ('object', 'class', 'u', 'v', 'range', 'size_px', 'verdict'))
        for i in here:
            p = read_pose(i['name'])
            if p is None:
                continue
            P = np.array([p[0], p[1], p[2]])
            pc = R_wc.T @ (P - t_wc)          # world -> optical
            if pc[2] <= 0.01:
                print('  %-14s %-9s %7s %7s %7s %8s  BEHIND CAMERA'
                      % (i['name'], i['cls'], '-', '-', '-', '-'))
                all_ok = False
                continue
            u = fx * pc[0] / pc[2] + cx
            v = fy * pc[1] / pc[2] + cy
            size_px = bbox[i['cls']] * fx / pc[2]
            inframe = 0 <= u < W and 0 <= v < H
            bad = []
            if not inframe:
                bad.append('OUT OF FRAME')
            if size_px < 12:
                bad.append('TINY')
            verdict = 'OK' if not bad else ' '.join(bad)
            all_ok &= not bad
            print('  %-14s %-9s %7.1f %7.1f %7.3f %8.1f  %s'
                  % (i['name'], i['cls'], u, v, pc[2], size_px, verdict))

            if img is not None and inframe:
                h = max(int(size_px / 2), 3)
                cv2.rectangle(img, (int(u - h), int(v - h)),
                              (int(u + h), int(v + h)), (0, 255, 0), 1)
                cv2.putText(img, i['cls'], (int(u - h), int(v - h) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        if img is not None:
            path = os.path.join(OUT_DIR, '%s.png' % ws)
            cv2.imwrite(path, img)
            print('  saved %s' % path)

    print('\n-> %s' % ('all objects visible at usable size'
                       if all_ok else 'VISIBILITY PROBLEM (see rows above)'))
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settle', type=float, default=20.0)
    ap.add_argument('--skip-stability', action='store_true')
    ap.add_argument('--skip-visibility', action='store_true')
    args = ap.parse_args()

    with open(MANIFEST) as f:
        manifest = json.load(f)
    print('manifest: scale %.4g, %d classes, %d instances\n'
          % (manifest['scale'], len(manifest['classes']),
             len(manifest['instances'])))

    ok = True
    if not args.skip_stability:
        ok &= check_stability(manifest, args.settle)

    if not args.skip_visibility:
        rclpy.init()
        node = Grabber()
        print('waiting for /rgb_camera/image_raw ...')
        if node.grab(timeout=60.0) is None:
            print('no camera frames - is gazebo running with the robot spawned?')
            ok = False
        else:
            ok &= check_visibility(manifest, node, args.settle)
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
