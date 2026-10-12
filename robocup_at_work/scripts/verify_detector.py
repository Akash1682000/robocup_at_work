#!/usr/bin/env python3
"""Compare the detector's 3D poses with gazebo ground truth.

Reports position error, yaw error (mod 180) and misses per workstation.
Use --spin to randomise object yaw, the default layout is all 0/90 deg.
Claimed sigma: 5 deg = depth, 15 = box only, 90 = no yaw.

Needs gazebo and perception.launch.py running:

    python3 scripts/verify_detector.py
    python3 scripts/verify_detector.py --settle 3 --gate 0.15
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState, GetEntityState
from vision_msgs.msg import Detection3DArray

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(PKG, 'config', 'objects_manifest.json')
ENTITY = 'robocup_robot'

# approach poses, validated with nav2
APPROACH = {'WS01': (-2.00,  1.75,  1.5708),
            'WS02': (-2.50,  0.00,  3.1416),
            'WS03': ( 2.00,  1.75,  1.5708),
            'WS04': ( 2.50, -1.50,  0.0000)}

SQUARE_TOL = 0.005      # square footprint, no yaw
POSE_TOPIC = '/objects/poses'


# --------------------------------------------------------------- geometry ---
def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def yaw_from_quat(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def into_base(p, base):
    """World point to base_footprint for a base at (x, y, yaw)."""
    bx, by, byaw = base
    dx, dy = p[0] - bx, p[1] - by
    c, s = math.cos(-byaw), math.sin(-byaw)
    return np.array([dx * c - dy * s, dx * s + dy * c, p[2]])


def yaw_gap(a, b):
    """Yaw difference in degrees, modulo 180."""
    d = math.degrees(a - b) % 180.0
    return min(d, 180.0 - d)


# ------------------------------------------------------------------- ros ---
class Checker(Node):
    def __init__(self):
        super().__init__('verify_detector')
        self.poses = None
        self.create_subscription(Detection3DArray, POSE_TOPIC,
                                 self._poses, 1)
        self.set_cli = self.create_client(SetEntityState,
                                          '/gazebo/set_entity_state')
        self.get_cli = self.create_client(GetEntityState,
                                          '/gazebo/get_entity_state')

    def _poses(self, m):
        self.poses = m

    def wait_services(self, timeout=20.0):
        return (self.set_cli.wait_for_service(timeout_sec=timeout)
                and self.get_cli.wait_for_service(timeout_sec=timeout))

    def _call(self, cli, req, timeout=5.0):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result() if fut.done() else None

    def place_base(self, x, y, yaw):
        req = SetEntityState.Request()
        st = req.state
        st.name = ENTITY
        st.pose.position.x, st.pose.position.y = x, y
        (st.pose.orientation.x, st.pose.orientation.y,
         st.pose.orientation.z, st.pose.orientation.w) = quat_from_yaw(yaw)
        st.twist.linear.x = st.twist.linear.y = st.twist.linear.z = 0.0
        st.twist.angular.x = st.twist.angular.y = st.twist.angular.z = 0.0
        st.reference_frame = 'world'
        r = self._call(self.set_cli, req)
        return bool(r and r.success)

    def place_object(self, name, x, y, z, yaw):
        req = SetEntityState.Request()
        st = req.state
        st.name = name
        st.pose.position.x, st.pose.position.y, st.pose.position.z = x, y, z
        (st.pose.orientation.x, st.pose.orientation.y,
         st.pose.orientation.z, st.pose.orientation.w) = quat_from_yaw(yaw)
        st.twist.linear.x = st.twist.linear.y = st.twist.linear.z = 0.0
        st.twist.angular.x = st.twist.angular.y = st.twist.angular.z = 0.0
        st.reference_frame = 'world'
        r = self._call(self.set_cli, req)
        return bool(r and r.success)

    def read_pose(self, name):
        """(x, y, z, yaw) in world, or None."""
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = 'world'
        r = self._call(self.get_cli, req)
        if not r or not r.success:
            return None
        p, o = r.state.pose.position, r.state.pose.orientation
        return (p.x, p.y, p.z, yaw_from_quat((o.x, o.y, o.z, o.w)))

    def fresh_poses(self, settle, timeout=15.0):
        """Wait for a fresh detection message."""
        self.poses = None
        end = time.time() + settle
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        self.poses = None
        end = time.time() + timeout
        while time.time() < end and self.poses is None:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.poses


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settle', type=float, default=2.0,
                    help='seconds to let the base and detector settle')
    ap.add_argument('--gate', type=float, default=0.12,
                    help='match radius in metres before calling it a miss')
    ap.add_argument('--spin', action='store_true',
                    help='turn every object to a random yaw first')
    ap.add_argument('--rounds', type=int, default=1,
                    help='measurements per workstation')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    with open(MANIFEST) as fh:
        manifest = json.load(fh)
    specs = {c['name']: c for c in manifest['classes']}
    by_ws = {}
    for inst in manifest['instances']:
        by_ws.setdefault(inst['workstation'], []).append(inst)

    rclpy.init()
    node = Checker()
    if not node.wait_services():
        print('no /gazebo/*_entity_state - is gazebo running with the '
              'state plugin in the world?')
        return 1

    print('=' * 72)
    print('DETECTOR 3D ACCURACY  (published pose vs gazebo ground truth)')
    print('=' * 72)

    errors = {}       # class -> list of (position error m, yaw error deg)
    misses = []
    extras = []       # detections with no matching object

    rng = random.Random(args.seed)

    for ws in sorted(APPROACH):
        x, y, yaw = APPROACH[ws]
        if not node.place_base(x, y, yaw):
            print('%s: could not teleport the base' % ws)
            continue

        for rnd in range(args.rounds):
            if args.spin:
                for inst in by_ws.get(ws, []):
                    pose = inst['pose']
                    node.place_object(inst['name'], pose['x'], pose['y'],
                                      pose['z'], rng.uniform(0.0, math.pi))
            if not measure(node, ws, (x, y, yaw), by_ws, specs, args,
                           errors, misses, extras, rnd == 0):
                break

    if args.spin:
        # restore the manifest layout
        for inst in manifest['instances']:
            pose = inst['pose']
            node.place_object(inst['name'], pose['x'], pose['y'],
                              pose['z'], pose['yaw'])

    return report(errors, misses, extras, args, node)


def measure(node, ws, base, by_ws, specs, args, errors, misses,
            extras, verbose):
    """One reading at a workstation. False if nothing was detected."""
    x, y, yaw = base
    msg = node.fresh_poses(args.settle)
    if msg is None:
        print('%s: nothing on %s - is the detector node up?'
              % (ws, POSE_TOPIC))
        return False
    truth = []
    for inst in by_ws.get(ws, []):
        gt = node.read_pose(inst['name'])
        if gt is None:
            continue
        truth.append((inst, into_base(gt[:3], base), gt[3] - yaw))

    if verbose:
        print('\n%s  base=(%.2f, %.2f, yaw %.3f)  %d on the table, '
              '%d published'
              % (ws, x, y, yaw, len(truth), len(msg.detections)))
        print('  %-14s %-9s %8s %8s %8s %8s  %s'
              % ('object', 'class', 'err mm', 'dz mm', 'yaw deg',
                 'claimed', 'result'))

    taken = set()
    for inst, gt_xyz, gt_yaw in truth:
        cls = inst['cls']
        best, best_d = None, 1e9
        for i, det in enumerate(msg.detections):
            if i in taken or not det.results:
                continue
            if det.results[0].hypothesis.class_id != cls:
                continue
            p = det.bbox.center.position
            d = math.hypot(p.x - gt_xyz[0], p.y - gt_xyz[1])
            if d < best_d:
                best, best_d = i, d

        if best is None or best_d > args.gate:
            if verbose:
                print('  %-14s %-9s %8s %8s %8s %8s  MISSED'
                      % (inst['name'], cls, '-', '-', '-', '-'))
            misses.append(inst['name'])
            continue

        taken.add(best)
        det = msg.detections[best]
        p = det.bbox.center.position
        o = det.bbox.center.orientation
        dz = (p.z - gt_xyz[2]) * 1000.0
        bb = specs[cls]['bbox']
        square = abs(bb[0] - bb[1]) < SQUARE_TOL
        fit = yaw_from_quat((o.x, o.y, o.z, o.w))
        dyaw = None if square else yaw_gap(fit, gt_yaw)
        # yaw sigma reported by the node
        claimed = math.degrees(math.sqrt(
            det.results[0].pose.covariance[35]))
        if verbose:
            print('  %-14s %-9s %8.1f %8.1f %8s %8.0f  ok'
                  % (inst['name'], cls, best_d * 1000.0, dz,
                     '-' if dyaw is None else '%.1f' % dyaw, claimed))
        errors.setdefault(cls, []).append((best_d, dyaw, claimed))

    extras.append(len(msg.detections) - len(taken))
    return True


def report(errors, misses, extras, args, node):

    def pct(vals, q):
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(q * len(vals)))]

    print('\n' + '=' * 72)
    print('%-10s %6s %8s %8s %8s %8s %8s'
          % ('class', 'n', 'mean mm', 'p90 mm', 'max mm', 'med yaw',
             '>10 deg'))
    worst = 0.0
    bad_total, yaw_total, fell_back = 0, 0, 0
    for cls in sorted(errors):
        ds = [e[0] * 1000.0 for e in errors[cls]]
        ys = [e[1] for e in errors[cls] if e[1] is not None]
        fell_back += sum(1 for e in errors[cls]
                         if e[1] is not None and e[2] > 10.0)
        worst = max(worst, max(ds))
        bad = sum(1 for v in ys if v > 10.0)
        bad_total += bad
        yaw_total += len(ys)
        print('%-10s %6d %8.1f %8.1f %8.1f %8s %8s'
              % (cls, len(ds), sum(ds) / len(ds), pct(ds, 0.9), max(ds),
                 '-' if not ys else '%.1f' % pct(ys, 0.5),
                 '-' if not ys else '%.0f%%' % (100.0 * bad / len(ys))))

    total = sum(len(v) for v in errors.values())
    print('\nmatched %d, missed %d, unmatched detections %d'
          % (total, len(misses), sum(extras)))
    if misses:
        print('misses: %s' % ', '.join(sorted(set(misses))))
    print('-> worst position error %.1f mm' % worst)
    print('   an 80 mm gripper wants this comfortably under about 15 mm')
    if fell_back:
        print('   %d of %d yaws fell back to the box - check the depth camera'
              % (fell_back, yaw_total))
    if yaw_total and not args.spin:
        print('   the yaw column here is meaningless - every object is at 0 or'
              ' 90 deg.\n   rerun with --spin --rounds 10 to actually test it')
    elif yaw_total:
        print('   yaw off by over 10 deg on %.0f%% of %d elongated objects'
              % (100.0 * bad_total / yaw_total, yaw_total))

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
