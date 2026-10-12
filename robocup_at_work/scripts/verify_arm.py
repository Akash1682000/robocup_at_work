#!/usr/bin/env python3
"""Stage 4a check: the arm moves and TF agrees with gazebo.

Sends random joint targets and compares the panda_leftfinger pose from
TF against /gazebo/link_states (fixed-joint links are merged away there).

Needs gazebo up (robot_gazebo.launch.py gui:=false):

    python3 scripts/verify_arm.py
    python3 scripts/verify_arm.py --targets 12 --seed 7
"""
import argparse
import math
import random
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

ARM = ['panda_joint%d' % i for i in range(1, 8)]
FINGERS = ['panda_finger_joint1', 'panda_finger_joint2']

# Franka limits, copied (not parsed) so a urdf typo shows up here
LIMITS = [(-2.9671, 2.9671), (-1.8326, 1.8326), (-2.9671, 2.9671),
          (-3.1416, 0.0873), (-2.9671, 2.9671), (-0.0873, 3.8223),
          (-2.9671, 2.9671)]

READY = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

# deepest link gazebo still keeps
WITNESS = 'panda_leftfinger'

ARM_ACTION = '/panda_arm_controller/follow_joint_trajectory'
HAND_ACTION = '/panda_hand_controller/follow_joint_trajectory'


# ------------------------------------------------------------------ probe ---
class Probe(Node):
    def __init__(self):
        super().__init__('verify_arm')
        self.set_parameters(
            [rclpy.parameter.Parameter('use_sim_time', value=True)])

        self.joints = {}
        self.track = 0.0
        self.links = {}

        self.create_subscription(JointState, '/joint_states',
                                 self._on_joints, 10)
        self.create_subscription(JointTrajectoryControllerState,
                                 '/panda_arm_controller/controller_state',
                                 self._on_state, 10)
        self.create_subscription(LinkStates, '/gazebo/link_states',
                                 self._on_links, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.arm = ActionClient(self, FollowJointTrajectory, ARM_ACTION)
        self.hand = ActionClient(self, FollowJointTrajectory, HAND_ACTION)

    def _on_joints(self, msg):
        self.joints.update(zip(msg.name, msg.position))

    def _on_state(self, msg):
        if msg.error.positions:
            self.track = max(self.track, max(abs(e) for e in msg.error.positions))

    def _on_links(self, msg):
        self.links = dict(zip(msg.name, msg.pose))

    def settle(self, secs):
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def send(self, client, names, positions, secs):
        """Run one trajectory. Returns the error code."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = names
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.velocities = [0.0] * len(positions)
        point.time_from_start.sec = int(secs)
        point.time_from_start.nanosec = int((secs % 1) * 1e9)
        goal.trajectory.points = [point]

        sent = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent)
        handle = sent.result()
        if not handle.accepted:
            return None
        done = handle.get_result_async()
        rclpy.spin_until_future_complete(self, done)
        return done.result().result.error_code


# --------------------------------------------------------------- geometry ---
def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def relative_position(parent, child):
    """Child pose in the parent frame, both given in world."""
    r = quat_to_matrix(parent.orientation)
    d = (child.position.x - parent.position.x,
         child.position.y - parent.position.y,
         child.position.z - parent.position.z)
    return tuple(sum(r[j][i] * d[j] for j in range(3)) for i in range(3))


def random_target(rng):
    """Random joint configuration inside the limits, with a margin."""
    out = []
    for lo, hi in LIMITS:
        margin = 0.1 * (hi - lo)
        out.append(rng.uniform(lo + margin, hi - margin))
    return out


# ------------------------------------------------------------------ checks ---
def check_startup(probe):
    print('start pose')
    worst = 0.0
    for name, want in zip(ARM, READY):
        off = abs(probe.joints.get(name, float('nan')) - want)
        worst = max(worst, off)
    print('  worst joint off the configured ready pose: %.2f deg'
          % math.degrees(worst))
    print('  (sag between the model appearing in gazebo and a controller '
          'taking\n   hold of it. Only means anything on a freshly launched '
          'sim - after a\n   run of this script the arm is already sitting at '
          'ready)')
    return worst


def check_targets(probe, count, seed, secs):
    rng = random.Random(seed)
    print('\ntrajectories (%d random targets inside the joint limits)' % count)
    print('  %-4s %9s %9s %10s' % ('n', 'code', 'end deg', 'hand mm'))

    endpoint_worst = 0.0
    hand_worst = 0.0
    failures = 0

    for n in range(count):
        target = random_target(rng)
        probe.track = 0.0
        code = probe.send(probe.arm, ARM, target, secs)
        probe.settle(1.0)

        if code != 0:
            failures += 1

        endpoint = max(abs(probe.joints.get(j, float('nan')) - t)
                       for j, t in zip(ARM, target))
        endpoint_worst = max(endpoint_worst, endpoint)

        hand = hand_disagreement(probe)
        if hand is not None:
            hand_worst = max(hand_worst, hand)

        print('  %-4d %9s %9.3f %10s'
              % (n + 1, 'reject' if code is None else code,
                 math.degrees(endpoint),
                 '-' if hand is None else '%.2f' % (hand * 1000.0)))

    print('  worst endpoint error   %.4f deg' % math.degrees(endpoint_worst))
    print('  worst forward kinematics vs gazebo   %.2f mm  (at %s)'
          % (hand_worst * 1000.0, WITNESS))
    return failures, endpoint_worst, hand_worst


def hand_disagreement(probe):
    """Error between TF and gazebo for the witness link."""
    probe.settle(0.4)
    base = probe.links.get('robocup_robot::base_footprint')
    hand = probe.links.get('robocup_robot::' + WITNESS)
    if base is None or hand is None:
        return None
    truth = relative_position(base, hand)

    try:
        tf = probe.tf_buffer.lookup_transform(
            'base_footprint', WITNESS, rclpy.time.Time())
    except Exception:
        return None
    told = (tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z)
    return math.dist(truth, told)


def check_limits(probe):
    """Command joint 1 past its limit and check gazebo stops it."""
    print('\njoint limits')
    limit = LIMITS[0][1]
    beyond = list(READY)
    beyond[0] = limit + 0.5
    probe.send(probe.arm, ARM, beyond, 3)
    probe.settle(1.0)
    reached = probe.joints.get(ARM[0], float('nan'))
    overshoot = reached - limit
    ok = overshoot < 0.02
    print('  asked for %.4f rad, limit is %.4f, stopped at %.4f'
          % (beyond[0], limit, reached))
    print('  %s' % ('held at the stop, %.1f mrad over' % (overshoot * 1000)
                    if ok else
                    'DROVE %.3f RAD PAST THE STOP' % overshoot))
    probe.send(probe.arm, ARM, READY, 3)
    return ok


def check_gripper(probe):
    print('\ngripper')
    worst_gap = 0.0
    ok = True
    for width in (0.04, 0.0, 0.02, 0.04):
        code = probe.send(probe.hand, FINGERS, [width, width], 2)
        probe.settle(0.6)
        left = probe.joints.get(FINGERS[0], float('nan'))
        right = probe.joints.get(FINGERS[1], float('nan'))
        gap = abs(left - right)
        worst_gap = max(worst_gap, gap)
        if code != 0 or abs(left - width) > 0.002:
            ok = False
        print('  commanded %.3f m  reached %.4f / %.4f m  code %s'
              % (width, left, right, code))
    print('  worst left-right disagreement: %.2f mm' % (worst_gap * 1000.0))
    return ok, worst_gap


# -------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--secs', type=float, default=3.0,
                    help='how long each trajectory is given')
    args = ap.parse_args()

    rclpy.init()
    probe = Probe()

    for client, name in ((probe.arm, ARM_ACTION), (probe.hand, HAND_ACTION)):
        if not client.wait_for_server(timeout_sec=20.0):
            print('no action server on %s - are the controllers spawned?' % name)
            return 1

    probe.settle(2.0)
    if not probe.joints:
        print('nothing on /joint_states - is joint_state_broadcaster active?')
        return 1

    sag = check_startup(probe)
    failures, endpoint, hand = check_targets(
        probe, args.targets, args.seed, args.secs)
    limits_ok = check_limits(probe)
    gripper_ok, gap = check_gripper(probe)

    probe.send(probe.arm, ARM, READY, 3)

    print('\nsummary')
    print('  %d of %d trajectories failed' % (failures, args.targets))
    print('  endpoint      %.4f deg' % math.degrees(endpoint))
    print('  fk vs gazebo  %.2f mm' % (hand * 1000.0))
    print('  start sag     %.2f deg' % math.degrees(sag))
    print('  fingers       %s, worst disagreement %.2f mm'
          % ('symmetric' if gripper_ok else 'PROBLEM', gap * 1000.0))

    rclpy.shutdown()
    bad = failures or not limits_ok or not gripper_ok or hand > 0.005
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
