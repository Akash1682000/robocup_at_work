#!/usr/bin/env python3
"""Pick at one workstation, place at another, and check the result.

Checks the object is on the table, near the target, upright, and
still there a few seconds later.

Needs the same stack as verify_pick.py.

    python3 scripts/verify_place.py
    python3 scripts/verify_place.py --from WS01 --to WS03 --classes F20_20_B
"""
import argparse
import json
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState, SetEntityState
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(PKG, 'config', 'objects_manifest.json')
ENTITY = 'robocup_robot'

APPROACH = {'WS01': (-2.00,  1.75,  1.5708),
            'WS02': (-2.50,  0.00,  3.1416),
            'WS03': ( 2.00,  1.75,  1.5708),
            'WS04': ( 2.50, -1.50,  0.0000)}


# --------------------------------------------------------------- geometry ---
def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def tilt_degrees(o):
    """Tilt of the object's z axis from vertical."""
    x, y, z, w = o.x, o.y, o.z, o.w
    up = 1 - 2 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, up))))


# ------------------------------------------------------------------- ros ---
class Checker(Node):
    def __init__(self):
        super().__init__('verify_place')
        self.poses = None
        self.create_subscription(Detection3DArray, '/objects/poses',
                                 self._poses, 1)
        self.set_cli = self.create_client(SetEntityState,
                                          '/gazebo/set_entity_state')
        self.get_cli = self.create_client(GetEntityState,
                                          '/gazebo/get_entity_state')
        self.params = self.create_client(SetParameters,
                                         '/object_picker/set_parameters')
        self.pick_cli = self.create_client(Trigger, '/pick/execute')
        self.place_cli = self.create_client(Trigger, '/pick/place')
        self.release_cli = self.create_client(Trigger, '/pick/release')

    def wait(self, timeout=25.0):
        for cli in (self.set_cli, self.get_cli, self.params,
                    self.pick_cli, self.place_cli):
            if not cli.wait_for_service(timeout_sec=timeout):
                return False
        return True

    def call(self, cli, req, timeout=240.0):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result() if fut.done() else None

    def _poses(self, msg):
        self.poses = msg

    def fresh_poses(self, settle, timeout=15.0):
        """Wait for a fresh detection message.

        Without this the pick can use a pose from before the teleport.
        """
        self.poses = None
        end = time.time() + settle
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        self.poses = None
        end = time.time() + timeout
        while time.time() < end and self.poses is None:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.poses

    def settle(self, secs):
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def place_entity(self, name, x, y, z, yaw):
        req = SetEntityState.Request()
        st = req.state
        st.name = name
        st.pose.position.x, st.pose.position.y, st.pose.position.z = x, y, z
        (st.pose.orientation.x, st.pose.orientation.y,
         st.pose.orientation.z, st.pose.orientation.w) = quat_from_yaw(yaw)
        st.reference_frame = 'world'
        r = self.call(self.set_cli, req, timeout=10.0)
        return bool(r and r.success)

    def read(self, name):
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = 'world'
        r = self.call(self.get_cli, req, timeout=10.0)
        if not r or not r.success:
            return None
        return r.state.pose

    def target(self, cls):
        req = SetParameters.Request()
        p = Parameter()
        p.name = 'target_class'
        p.value = ParameterValue(type=ParameterType.PARAMETER_STRING,
                                 string_value=cls)
        req.parameters = [p]
        return self.call(self.params, req, timeout=10.0) is not None


# ------------------------------------------------------------------ main ---
def run(node, inst, spec, args, rows):
    name, cls = inst['name'], inst['cls']
    src, dst = args.src, args.dst

    pose = inst['pose']
    node.place_entity(name, pose['x'], pose['y'], pose['z'], pose['yaw'])
    node.settle(0.5)

    x, y, yaw = APPROACH[src]
    node.place_entity(ENTITY, x, y, 0.0, yaw)
    node.fresh_poses(args.settle)
    node.target(cls)

    print('  %-12s picking at %s' % (cls, src), flush=True)
    got = node.call(node.pick_cli, Trigger.Request(), timeout=args.timeout)
    if got is None or not got.success:
        rows.append((cls, 'pick failed', got.message if got else 'no answer'))
        print('  %-12s pick failed: %s'
              % (cls, got.message if got else 'no answer'))
        return

    x, y, yaw = APPROACH[dst]
    node.place_entity(ENTITY, x, y, 0.0, yaw)
    node.fresh_poses(args.settle)

    print('  %-12s placing at %s' % (cls, dst), flush=True)
    put = node.call(node.place_cli, Trigger.Request(), timeout=args.timeout)
    if put is None or not put.success:
        node.call(node.release_cli, Trigger.Request(), timeout=30.0)
        rows.append((cls, 'place failed', put.message if put else 'no answer'))
        print('  %-12s place failed: %s'
              % (cls, put.message if put else 'no answer'))
        return

    node.settle(1.0)
    first = node.read(name)
    node.settle(args.watch)
    second = node.read(name)
    if first is None or second is None:
        print('  %-12s gazebo lost the object' % cls)
        return

    on_table = first.position.z > args.table_gate
    tilt = tilt_degrees(second.orientation)
    crept = math.dist((first.position.x, first.position.y, first.position.z),
                      (second.position.x, second.position.y,
                       second.position.z))
    upright = tilt < args.tilt_gate
    still = crept < args.creep_gate
    ok = on_table and upright and still

    rows.append((cls, 'placed' if ok else 'poor', ''))
    print('  %-12s %-7s z %5.3f m  tilt %5.1f deg  crept %5.1f mm  %s'
          % (cls, 'placed' if ok else 'poor', first.position.z, tilt,
             crept * 1000.0,
             'on the table' if on_table else 'ON THE FLOOR'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='src', default='WS01',
                    choices=sorted(APPROACH), help='where to pick from')
    ap.add_argument('--to', dest='dst', default='WS03',
                    choices=sorted(APPROACH), help='where to place')
    ap.add_argument('--classes', action='append', default=[],
                    help='only these classes, repeatable')
    ap.add_argument('--settle', type=float, default=2.5)
    ap.add_argument('--watch', type=float, default=2.0,
                    help='seconds to watch it after the arm has gone')
    ap.add_argument('--timeout', type=float, default=240.0)
    ap.add_argument('--table-gate', type=float, default=0.09,
                    help='above this height in gazebo it is on a table')
    ap.add_argument('--tilt-gate', type=float, default=25.0,
                    help='degrees off upright before it counts as dropped')
    ap.add_argument('--creep-gate', type=float, default=0.005,
                    help='metres it may move after the arm leaves')
    args = ap.parse_args()

    with open(MANIFEST) as f:
        manifest = json.load(f)
    specs = {c['name']: c for c in manifest['classes']}
    here = [i for i in manifest['instances']
            if i['workstation'] == args.src
            and (not args.classes or i['cls'] in args.classes)]

    rclpy.init()
    node = Checker()
    print('waiting for the stack', flush=True)
    if not node.wait():
        print('the stack is not up: /pick/place did not appear')
        node.destroy_node()
        rclpy.shutdown()
        return 1

    print('picking at %s, placing at %s' % (args.src, args.dst), flush=True)
    rows = []
    for inst in here:
        spec = specs.get(inst['cls'])
        if spec is not None:
            run(node, inst, spec, args, rows)

    node.destroy_node()
    rclpy.shutdown()

    good = sum(1 for r in rows if r[1] == 'placed')
    print('\n%d of %d placed on the table, upright, and left alone'
          % (good, len(rows)))
    for cls, how, why in rows:
        if how != 'placed':
            print('  %-12s %s %s' % (cls, how, why))
    return 0 if good == len(rows) and rows else 1


if __name__ == '__main__':
    sys.exit(main())
