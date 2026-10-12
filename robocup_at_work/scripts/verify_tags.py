#!/usr/bin/env python3
"""Check the workstation tags render and decode in gazebo.

Uses pupil_apriltags (same libapriltag as apriltag_ros). Don't use
cv2.aruco, it is stricter and rejects valid tags.

Needs gazebo running with the robot spawned.
"""
import math
import subprocess
import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from pupil_apriltags import Detector
from sensor_msgs.msg import CameraInfo, Image

# name, x, y, yaw, expected tag id
POSES = [
    ('WS01', -2.00,  1.75,  1.5708, 1),
    ('WS02', -2.50,  0.00,  3.1416, 2),
    ('WS03',  2.00,  1.75,  1.5708, 3),
    ('WS04',  2.50, -1.50,  0.0000, 4),
]
ENTITY = 'robocup_robot'
SETTLE = 3.0          # s, let the base settle after a move


class Grabber(Node):
    def __init__(self):
        super().__init__('tag_verifier')
        self.bridge = CvBridge()
        self.frame = None
        self.info = None
        self.create_subscription(Image, '/rgb_camera/image_raw', self._img, 10)
        self.create_subscription(CameraInfo, '/rgb_camera/camera_info', self._inf, 10)

    def _img(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _inf(self, msg):
        self.info = msg

    def grab(self, timeout=10.0):
        self.frame = None
        end = time.time() + timeout
        while time.time() < end and self.frame is None:
            rclpy.spin_once(self, timeout_sec=0.2)
        return self.frame


def _set_pose(name, x, y, yaw):
    # z=0.05 is the resting height, dropping from higher makes it slide
    r = subprocess.run(
        ['gz', 'model', '-m', name, '-x', str(x), '-y', str(y), '-z', '0.05',
         '-R', '0', '-P', '0', '-Y', str(yaw)],
        check=False, capture_output=True, text=True)
    if r.returncode != 0 or 'Invalid' in (r.stdout + r.stderr):
        raise RuntimeError(f'gz model failed: {r.stdout}{r.stderr}')


def read_pose(name):
    r = subprocess.run(['gz', 'model', '-m', name, '-p'],
                       check=True, capture_output=True, text=True)
    v = [float(t) for t in r.stdout.split()]
    return v[0], v[1], v[5]          # x, y, yaw


def teleport(name, x, y, yaw):
    """Place the base, retry if it slid."""
    for attempt in range(4):
        _set_pose(name, x, y, yaw)
        time.sleep(SETTLE)
        px, py, pyaw = read_pose(name)
        err = math.hypot(px - x, py - y)
        dyaw = abs((pyaw - yaw + math.pi) % (2 * math.pi) - math.pi)
        if err < 0.02 and dyaw < 0.02:
            return px, py, pyaw, err, dyaw
    return px, py, pyaw, err, dyaw


def main():
    rclpy.init()
    node = Grabber()

    # wait for the camera to come up
    print('waiting for /rgb_camera/image_raw ...')
    if node.grab(timeout=60.0) is None:
        sys.exit('no camera frames - is gazebo running with the robot spawned?')

    for _ in range(40):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.info is not None:
            break
    if node.info is not None:
        d = np.asarray(node.info.d)
        print(f'camera_info: {node.info.width}x{node.info.height} '
              f'fx={node.info.k[0]:.1f} fy={node.info.k[4]:.1f}')
        print(f'  distortion D = {list(d)} -> '
              f'{"ZERO, image_raw IS rectified" if not np.any(d) else "NONZERO, needs image_proc!"}')

    # same settings as config/apriltag.yaml
    detector = Detector(families='tag36h11', nthreads=2, quad_decimate=1.0,
                        quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)

    results = []
    for name, x, y, yaw, want in POSES:
        px, py, pyaw, err, dyaw = teleport(ENTITY, x, y, yaw)
        print(f'{name}: commanded ({x:+.2f},{y:+.2f},{yaw:.3f}) '
              f'-> actual ({px:+.3f},{py:+.3f},{pyaw:.3f}) '
              f'err {err*100:.1f} cm / {math.degrees(dyaw):.1f} deg')
        for _ in range(10):                       # flush stale frames
            rclpy.spin_once(node, timeout_sec=0.1)
        img = node.grab()
        if img is None:
            results.append((name, want, None, 0.0, 0.0, 0, 'no frame'))
            continue

        cv2.imwrite(f'/tmp/tag_view_{name}.png', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector.detect(gray)

        if not dets:
            results.append((name, want, None, 0.0, 0.0, 0, 'NO DETECTION'))
            continue

        d = max(dets, key=lambda x: x.decision_margin)
        c = np.asarray(d.corners)
        side = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))
        note = 'OK' if d.tag_id == want else f'WRONG ID (wanted {want})'
        results.append((name, want, d.tag_id, side, d.decision_margin, d.hamming, note))

    print('\n=== RESULTS ===')
    ok = True
    for name, want, got, side, margin, ham, note in results:
        print(f'  {name}: expected id {want}, got {got}, '
              f'~{side:.0f} px ({side/8:.1f} px/cell), '
              f'hamming={ham}, margin={margin:.0f}  {note}')
        ok &= (note == 'OK')
    print(f'\nframes saved to /tmp/tag_view_WS0*.png')
    print('ALL FOUR OK' if ok else 'FAILURES PRESENT - if none decoded at all, '
          'suspect (a) the camera seeing its own chassis, or (b) a mirrored '
          'texture: flip every tag yaw by pi in the world')

    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
