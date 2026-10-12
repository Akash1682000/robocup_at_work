#!/usr/bin/env python3
"""Pick every object and check in gazebo that it left the table.

For each object: teleport to the approach pose, call /pick/execute,
measure the object height, release, and restore the table.

Needs gazebo, moveit.launch.py and perception.launch.py running:

    python3 scripts/verify_pick.py
    python3 scripts/verify_pick.py --only WS03 --settle 3
"""
import argparse
import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState, SetEntityState
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(PKG, 'config', 'objects_manifest.json')
ENTITY = 'robocup_robot'

# approach poses, validated with nav2
APPROACH = {'WS01': (-2.00,  1.75,  1.5708),
            'WS02': (-2.50,  0.00,  3.1416),
            'WS03': ( 2.00,  1.75,  1.5708),
            'WS04': ( 2.50, -1.50,  0.0000)}

FINGERS = ['panda_finger_joint1', 'panda_finger_joint2']
POSE_TOPIC = '/objects/poses'


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


# ----------------------------------------------------------------- driver ---
class Driver(Node):
    def __init__(self):
        super().__init__('verify_pick')
        self.set_parameters(
            [rclpy.parameter.Parameter('use_sim_time', value=True)])

        self.poses = None
        self.joints = {}
        self.create_subscription(Detection3DArray, POSE_TOPIC,
                                 self._on_poses, 1)
        self.create_subscription(JointState, '/joint_states',
                                 self._on_joints, 10)

        self.set_cli = self.create_client(SetEntityState,
                                          '/gazebo/set_entity_state')
        self.get_cli = self.create_client(GetEntityState,
                                          '/gazebo/get_entity_state')
        self.pick_cli = self.create_client(Trigger, '/pick/execute')
        self.release_cli = self.create_client(Trigger, '/pick/release')
        self.param_cli = self.create_client(SetParameters,
                                            '/object_picker/set_parameters')

    def _on_poses(self, msg):
        self.poses = msg

    def _on_joints(self, msg):
        self.joints.update(zip(msg.name, msg.position))

    def settle(self, secs):
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _call(self, cli, req, timeout=10.0):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result() if fut.done() else None

    def wait_for(self, timeout=30.0):
        missing = []
        for cli, name in ((self.set_cli, '/gazebo/set_entity_state'),
                          (self.get_cli, '/gazebo/get_entity_state'),
                          (self.pick_cli, '/pick/execute'),
                          (self.release_cli, '/pick/release'),
                          (self.param_cli, '/object_picker/set_parameters')):
            if not cli.wait_for_service(timeout_sec=timeout):
                missing.append(name)
            timeout = 5.0
        return missing

    # ------------------------------------------------------------ gazebo ---
    def place(self, name, x, y, z, yaw):
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
        """(x, y, z) in world, or None."""
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = 'world'
        r = self._call(self.get_cli, req)
        if not r or not r.success:
            return None
        p = r.state.pose.position
        return (p.x, p.y, p.z)

    # ------------------------------------------------------------ picker ---
    def target(self, cls):
        req = SetParameters.Request()
        value = ParameterValue(type=ParameterType.PARAMETER_STRING,
                               string_value=cls)
        req.parameters = [Parameter(name='target_class', value=value)]
        r = self._call(self.param_cli, req)
        return bool(r and r.results and r.results[0].successful)

    def pick(self, timeout):
        r = self._call(self.pick_cli, Trigger.Request(), timeout)
        if r is None:
            return False, 'no answer from /pick/execute inside %.0f s' % timeout
        return r.success, r.message

    def release(self, timeout=30.0):
        r = self._call(self.release_cli, Trigger.Request(), timeout)
        return bool(r and r.success)

    def fresh_poses(self, settle, timeout=20.0):
        """Wait for a fresh detection message."""
        self.settle(settle)
        self.poses = None
        end = time.time() + timeout
        while time.time() < end and self.poses is None:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.poses

    def opening(self):
        return sum(self.joints.get(f, 0.0) for f in FINGERS)


# ------------------------------------------------------------------ main ---
def reset_table(node, instances, settle):
    """Reset all objects at this workstation to the manifest poses.

    A dropped object can knock the next one off its mark.
    """
    for inst in instances:
        pose = inst['pose']
        node.place(inst['name'], pose['x'], pose['y'], pose['z'], pose['yaw'])
    node.settle(settle)


def attempt(node, inst, spec, args, rows):
    """Pick one object, measure, put it back."""
    name, cls = inst['name'], inst['cls']
    width = min(spec['bbox'][0], spec['bbox'][1])

    before = node.read_pose(name)
    if before is None:
        print('  %-12s gazebo has no model called %s' % (cls, name))
        return

    if not node.target(cls):
        print('  %-12s could not set target_class on the picker' % cls)
        return

    ok, message = node.pick(args.timeout)
    node.settle(0.5)
    gap = node.opening()
    after = node.read_pose(name)
    risen = (after[2] - before[2]) if after else 0.0
    slid = math.dist(after[:2], before[:2]) if after else 0.0

    # Pass/fail on gazebo height only. The finger gap is printed but not
    # used, it gives false negatives on M20_100 (head wider than shaft).
    lifted = risen > args.lift_gate
    rows.append((inst['workstation'], cls, ok, gap, width, risen,
                 lifted and ok, message, slid))

    print('  %-12s %-9s gap %5.1f mm (%.1f)  rose %6.1f mm  slid %5.1f mm  %s'
          % (cls, 'picked' if ok else 'FAILED', gap * 1000.0, width * 1000.0,
             risen * 1000.0, slid * 1000.0,
             'held' if (ok and lifted) else 'not held'))
    if not ok or not lifted:
        print('      %s' % message)

    node.release()
    node.settle(args.restore)


def report(rows, args):
    print('\nsummary')
    if not rows:
        print('  nothing ran')
        return 1

    held = [r for r in rows if r[6]]
    planned = [r for r in rows if r[2]]
    print('  %d of %d objects came off the table' % (len(held), len(rows)))
    print('  %d of %d picks the node itself called a success'
          % (len(planned), len(rows)))

    if held:
        errs = [abs(r[3] - r[4]) for r in held]
        print('  finger gap against the object width, on the ones held: '
              'worst %.1f mm, median %.1f mm'
              % (max(errs) * 1000.0, sorted(errs)[len(errs) // 2] * 1000.0))
        print('  smallest lift measured in gazebo: %.0f mm'
              % (min(r[5] for r in held) * 1000.0))

    lied = [r for r in rows if r[2] and not r[6]]
    if lied:
        print('\n  %d pick(s) reported success without the object moving:'
              % len(lied))
        for r in lied:
            print('    %s %s: gap %.1f mm against %.1f mm, rose %.1f mm'
                  % (r[0], r[1], r[3] * 1000.0, r[4] * 1000.0, r[5] * 1000.0))

    failed = [r for r in rows if not r[2]]
    if failed:
        print('\n  %d pick(s) the node refused or gave up on:' % len(failed))
        for r in failed:
            print('    %s %s: %s' % (r[0], r[1], r[7]))

    return 0 if len(held) == len(rows) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', action='append', default=[],
                    help='workstation to test, repeatable. Default is all four')
    ap.add_argument('--classes', action='append', default=[],
                    help='class to test, repeatable. Default is every object '
                         'at the workstation')
    ap.add_argument('--settle', type=float, default=2.5,
                    help='seconds to let the base and the detector settle '
                         'after a teleport')
    ap.add_argument('--timeout', type=float, default=180.0,
                    help='seconds to give one whole pick')
    ap.add_argument('--lift-gate', type=float, default=0.05,
                    help='how far the object has to rise in gazebo, in metres, '
                         'before the pick counts')
    ap.add_argument('--restore', type=float, default=1.0,
                    help='seconds to wait after releasing, before the object '
                         'is put back')
    args = ap.parse_args()

    with open(MANIFEST) as fh:
        manifest = json.load(fh)
    specs = {c['name']: c for c in manifest['classes']}
    by_ws = {}
    for inst in manifest['instances']:
        by_ws.setdefault(inst['workstation'], []).append(inst)

    stations = sorted(args.only) if args.only else sorted(APPROACH)

    rclpy.init()
    node = Driver()
    missing = node.wait_for()
    if missing:
        print('nothing on %s' % ', '.join(missing))
        print('gazebo, moveit.launch.py and perception.launch.py all have to '
              'be up')
        return 1

    print('picking every object in the manifest, from the navigation approach '
          'poses')

    rows = []
    for ws in stations:
        if ws not in APPROACH:
            print('%s is not a workstation' % ws)
            continue
        x, y, yaw = APPROACH[ws]
        print('\n%s' % ws)
        if not node.place(ENTITY, x, y, 0.0, yaw):
            print('  could not teleport the base')
            continue

        if node.fresh_poses(args.settle) is None:
            print('  nothing on %s - is the detector running?' % POSE_TOPIC)
            continue

        here = by_ws.get(ws, [])
        for inst in here:
            spec = specs.get(inst['cls'])
            if spec is None:
                continue
            if args.classes and inst['cls'] not in args.classes:
                continue
            reset_table(node, here, 0.5)
            node.fresh_poses(args.settle)
            attempt(node, inst, spec, args, rows)

    code = report(rows, args)
    node.destroy_node()
    rclpy.shutdown()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
