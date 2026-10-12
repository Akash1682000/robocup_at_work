#!/usr/bin/env python3
"""Stage 4b check: reach and collision matrix.

Runs /compute_ik on a top-down grasp for every object from its approach
pose, then checks the named poses are valid and a pose through the base
is rejected.

Needs gazebo and moveit.launch.py running:

    python3 scripts/verify_moveit.py
    python3 scripts/verify_moveit.py --clearance 0.12
"""
import argparse
import json
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.msg import Constraints, PositionIKRequest, RobotState
from moveit_msgs.srv import GetPositionIK, GetStateValidity
from sensor_msgs.msg import JointState

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(PKG, 'config', 'objects_manifest.json')

# approach poses, validated with nav2
APPROACH = {'WS01': (-2.00,  1.75,  1.5708),
            'WS02': (-2.50,  0.00,  3.1416),
            'WS03': ( 2.00,  1.75,  1.5708),
            'WS04': ( 2.50, -1.50,  0.0000)}

# panda_link8 -> panda_tcp, see urdf/panda_arm.urdf.xacro
HAND_YAW = -math.pi / 4
TCP_OFFSET = 0.1034

# shoulder in base_footprint, reach is measured to panda_link8 (flange)
SHOULDER = (-0.05, 0.0, 0.05 + 0.15 + 0.333)
REACH = 0.855

ARM = ['panda_joint%d' % i for i in range(1, 8)]

READY = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
TRANSPORT = [0.0, -0.5599, 0.0, -2.97, 0.0, 1.5, 0.785]

# arm folded through the chassis, found by sampling
THROUGH_BASE = [0.9220, -1.3813, 2.5626, -2.6851, -0.9997, 2.7295, 0.5782]


# --------------------------------------------------------------- geometry ---
def quat_from_matrix(m):
    """Column-major rotation matrix as three axis tuples -> x, y, z, w."""
    (xx, yx, zx), (xy, yy, zy), (xz, yz, zz) = m
    trace = xx + yy + zz
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return ((zy - yz) / s, (xz - zx) / s, (yx - xy) / s, 0.25 * s)
    if xx > yy and xx > zz:
        s = math.sqrt(1.0 + xx - yy - zz) * 2
        return (0.25 * s, (xy + yx) / s, (xz + zx) / s, (zy - yz) / s)
    if yy > zz:
        s = math.sqrt(1.0 + yy - xx - zz) * 2
        return ((xy + yx) / s, 0.25 * s, (yz + zy) / s, (xz - zx) / s)
    s = math.sqrt(1.0 + zz - xx - yy) * 2
    return ((xz + zx) / s, (yz + zy) / s, 0.25 * s, (yx - xy) / s)


def world_to_base(x, y, base):
    """World point to base_footprint for a base at (bx, by, byaw)."""
    bx, by, byaw = base
    dx, dy = x - bx, y - by
    c, s = math.cos(byaw), math.sin(byaw)
    return (dx * c + dy * s, -dx * s + dy * c)


def top_down_grasp(x, y, z, yaw):
    """tcp pose pointing straight down, fingers across the short axis."""
    c, s = math.cos(yaw), math.sin(yaw)
    x_axis = (-c, -s, 0.0)
    y_axis = (-s, c, 0.0)
    z_axis = (0.0, 0.0, -1.0)
    return (x, y, z), quat_from_matrix((x_axis, y_axis, z_axis))


def tcp_to_link8(position, quat):
    """Convert a tcp pose to the matching panda_link8 pose."""
    qx, qy, qz, qw = quat
    half = -HAND_YAW / 2.0
    cz, sz = math.cos(half), math.sin(half)
    # quat * Rz(-HAND_YAW)
    out = (qx * cz + qy * sz,
           qy * cz - qx * sz,
           qz * cz + qw * sz,
           qw * cz - qz * sz)
    # tcp z points down here
    back = (position[0], position[1], position[2] + TCP_OFFSET)
    return back, out


# ------------------------------------------------------------------ probe ---
class Probe(Node):
    def __init__(self):
        super().__init__('verify_moveit')
        self.set_parameters(
            [rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.joints = None
        self.create_subscription(JointState, '/joint_states',
                                 self._on_joints, 10)
        self.ik = self.create_client(GetPositionIK, '/compute_ik')
        self.valid = self.create_client(GetStateValidity,
                                        '/check_state_validity')

    def _on_joints(self, msg):
        self.joints = msg

    def settle(self, secs):
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def solve(self, position, quat, attempts, timeout):
        req = GetPositionIK.Request()
        r = PositionIKRequest()
        r.group_name = 'panda_arm'
        r.ik_link_name = 'panda_link8'
        r.avoid_collisions = True
        r.pose_stamped.header.frame_id = 'base_footprint'
        r.pose_stamped.pose = Pose()
        r.pose_stamped.pose.position.x = position[0]
        r.pose_stamped.pose.position.y = position[1]
        r.pose_stamped.pose.position.z = position[2]
        (r.pose_stamped.pose.orientation.x,
         r.pose_stamped.pose.orientation.y,
         r.pose_stamped.pose.orientation.z,
         r.pose_stamped.pose.orientation.w) = quat
        r.timeout.sec = int(timeout)
        r.timeout.nanosec = int((timeout % 1) * 1e9)
        state = RobotState()
        state.joint_state = self.joints
        r.robot_state = state
        req.ik_request = r

        for _ in range(attempts):
            fut = self.ik.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
            res = fut.result()
            if res is not None and res.error_code.val == 1:
                return True
        return False


    def check(self, positions):
        """Check a configuration for collisions. Returns (ok, contacts)."""
        req = GetStateValidity.Request()
        req.group_name = 'panda_arm'
        state = RobotState()
        state.joint_state = JointState()
        state.joint_state.name = list(self.joints.name)
        state.joint_state.position = list(self.joints.position)
        for name, value in zip(ARM, positions):
            state.joint_state.position[
                state.joint_state.name.index(name)] = value
        req.robot_state = state
        req.constraints = Constraints()

        fut = self.valid.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        res = fut.result()
        if res is None:
            return None, []
        pairs = sorted({(c.contact_body_1, c.contact_body_2)
                        for c in res.contacts})
        return res.valid, pairs


# ------------------------------------------------------------------ report ---
def reach_report(probe, clearance, attempts, timeout):
    with open(MANIFEST) as f:
        manifest = json.load(f)
    bbox = {c['name']: c['bbox'] for c in manifest['classes']}

    print('reachability of a top-down grasp, from the navigation approach poses')
    print('  %-12s %-10s %7s %7s %7s  %-8s %-8s'
          % ('workstation', 'object', 'fwd m', 'side m', 'reach m',
             'grasp', 'pre'))

    rows = []
    for inst in manifest['instances']:
        ws = inst['workstation']
        base = APPROACH[ws]
        pose = inst['pose']
        bx, by = world_to_base(pose['x'], pose['y'], base)
        yaw = pose['yaw'] - base[2]

        height = bbox[inst['cls']][2]
        grasp_z = manifest['table_top_z'] + height / 2.0

        reach = math.dist(SHOULDER, (bx, by, grasp_z + TCP_OFFSET))

        at_grasp = probe.solve(
            *tcp_to_link8(*top_down_grasp(bx, by, grasp_z, yaw)),
            attempts=attempts, timeout=timeout)
        at_pre = probe.solve(
            *tcp_to_link8(*top_down_grasp(bx, by, grasp_z + clearance, yaw)),
            attempts=attempts, timeout=timeout)

        rows.append((ws, inst['cls'], bx, by, reach, at_grasp, at_pre))
        print('  %-12s %-10s %7.3f %7.3f %7.3f  %-8s %-8s'
              % (ws, inst['cls'], bx, by, reach,
                 'ok' if at_grasp else 'NO IK', 'ok' if at_pre else 'NO IK'))

    ok = sum(1 for r in rows if r[5] and r[6])
    print('\n  %d of %d objects reachable at both the grasp and the pre-grasp'
          % (ok, len(rows)))
    print('  furthest solved grasp is %.3f m from the shoulder, against a '
          '%.3f m reach' % (max(r[4] for r in rows), REACH))

    bad = [r for r in rows if not (r[5] and r[6])]
    if bad:
        near = min(r[4] for r in bad)
        far = max(r[4] for r in rows if r[5] and r[6]) if ok else 0.0
        print('  the closest failure is %.3f m from the shoulder; the furthest'
              % near)
        print('  success is %.3f m. Anything past about that is out of reach'
              % far)
        print('  and the base has to move closer before picking.')
    return rows


def collision_report(probe):
    """Named poses must be valid, THROUGH_BASE must be rejected."""
    print('\nthe collision matrix')
    outcome = True
    for name, positions, want in (('ready', READY, True),
                                  ('transport', TRANSPORT, True),
                                  ('folded into the base', THROUGH_BASE, False)):
        valid, pairs = probe.check(positions)
        if valid is None:
            print('  %-22s no answer from /check_state_validity' % name)
            outcome = False
            continue
        agreed = (valid == want)
        outcome = outcome and agreed
        print('  %-22s %-14s %s'
              % (name,
                 'collision free' if valid else 'in collision',
                 'as expected' if agreed else
                 ('EXPECTED A COLLISION' if want is False
                  else 'EXPECTED IT TO BE FREE')))
        if pairs:
            for a, b in pairs[:4]:
                print('      %s / %s' % (a, b))
    return outcome


# -------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clearance', type=float, default=0.10,
                    help='pre-grasp height above the grasp, in metres')
    ap.add_argument('--attempts', type=int, default=4,
                    help='IK restarts before calling a pose unreachable. KDL is '
                         'a numerical solver and one failure means little')
    ap.add_argument('--timeout', type=float, default=0.2)
    args = ap.parse_args()

    rclpy.init()
    probe = Probe()
    if not probe.ik.wait_for_service(timeout_sec=20.0):
        print('/compute_ik never appeared - is move_group running?')
        return 1
    probe.settle(2.0)
    if probe.joints is None:
        print('nothing on /joint_states')
        return 1

    if not probe.valid.wait_for_service(timeout_sec=10.0):
        print('/check_state_validity never appeared')
        return 1

    collisions_ok = collision_report(probe)
    rows = reach_report(probe, args.clearance, args.attempts, args.timeout)

    rclpy.shutdown()
    reach_ok = all(r[5] and r[6] for r in rows)
    return 0 if (collisions_ok and reach_ok) else 1


if __name__ == '__main__':
    sys.exit(main())
