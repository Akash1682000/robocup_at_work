#!/usr/bin/env python3
"""Check the base holds still under the arm load, and still drives.

Holds an arm pose and tracks base drift in the base frame, then drives
the base to make sure the hold settings didn't break motion.
--sweep tries all arm poses, --no-drive skips the drive test.

Needs gazebo up (robot_gazebo.launch.py gui:=false):

    python3 scripts/verify_base_hold.py
    python3 scripts/verify_base_hold.py --sweep
    python3 scripts/verify_base_hold.py --ws WS04 --seconds 60
"""
import argparse
import math
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetEntityState, GetEntityState
from trajectory_msgs.msg import JointTrajectoryPoint

ENTITY = 'robocup_robot'
ARM = ['panda_joint%d' % i for i in range(1, 8)]
ARM_ACTION = '/panda_arm_controller/follow_joint_trajectory'

# approach poses, validated with nav2
APPROACH = {'WS01': (-2.00,  1.75,  1.5708),
            'WS02': (-2.50,  0.00,  3.1416),
            'WS03': ( 2.00,  1.75,  1.5708),
            'WS04': ( 2.50, -1.50,  0.0000)}

# ready/transport from the srdf, up/out put the mass over and in front
# of the base
POSES = {
    'ready':     [0.0, -0.785, 0.0, -2.356, 0.0,  1.571, 0.785],
    'transport': [0.0, -0.5599, 0.0, -2.97, 0.0,  1.500, 0.785],
    'up':        [0.0, -0.300, 0.0, -0.300, 0.0,  1.571, 0.785],
    'out':       [0.0,  0.600, 0.0, -1.200, 0.0,  1.800, 0.785],
}

# max drift over the window
HOLD_MM = 25.0

# drive tests (label, vx, vy, wz), each held for DRIVE_SECS
DRIVES = [('forward', 0.20, 0.00, 0.0),
          ('left',    0.00, 0.15, 0.0),
          ('turn',    0.00, 0.00, 0.5)]
DRIVE_SECS = 3.0
DRIVE_TOL = 0.10        # allowed error, fraction of commanded distance

# Loose on purpose: planar_move undershoots turns (1.5 rad -> ~1.0).
# Nav2 is closed loop so it only slows turns. This catches regressions.
TURN_TOL = 0.40

# open floor, not an approach pose (those face a table 0.2 m away)
DRIVE_POSE = (-0.5, -1.0, 0.0)


# --------------------------------------------------------------- geometry ---
def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def yaw_from_quat(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def into_base(dx, dy, yaw):
    """World displacement to (forward, left) in the base frame."""
    c, s = math.cos(-yaw), math.sin(-yaw)
    return dx * c - dy * s, dx * s + dy * c


def slope(ts, vs):
    """Least squares slope of vs over ts."""
    n = len(ts)
    if n < 2:
        return 0.0
    tbar = sum(ts) / n
    vbar = sum(vs) / n
    num = sum((t - tbar) * (v - vbar) for t, v in zip(ts, vs))
    den = sum((t - tbar) ** 2 for t in ts)
    return num / den if den else 0.0


# ------------------------------------------------------------------ probe ---
class Probe(Node):
    def __init__(self):
        super().__init__('verify_base_hold')
        self.set_parameters(
            [rclpy.parameter.Parameter('use_sim_time', value=True)])

        self.set_cli = self.create_client(SetEntityState,
                                          '/gazebo/set_entity_state')
        self.get_cli = self.create_client(GetEntityState,
                                          '/gazebo/get_entity_state')
        self.arm = ActionClient(self, FollowJointTrajectory, ARM_ACTION)
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)

    def wait_services(self, timeout=20.0):
        return (self.set_cli.wait_for_service(timeout_sec=timeout)
                and self.get_cli.wait_for_service(timeout_sec=timeout)
                and self.arm.wait_for_server(timeout_sec=timeout))

    def _call(self, cli, req, timeout=5.0):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result() if fut.done() else None

    def settle(self, secs):
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

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

    def read_base(self):
        """(x, y, z, yaw) in world, or None."""
        req = GetEntityState.Request()
        req.name = ENTITY
        req.reference_frame = 'world'
        r = self._call(self.get_cli, req)
        if not r or not r.success:
            return None
        p, o = r.state.pose.position, r.state.pose.orientation
        return (p.x, p.y, p.z, yaw_from_quat((o.x, o.y, o.z, o.w)))

    def drive(self, vx, vy, wz, secs):
        """Send cmd_vel for a while. Returns (forward, left, turned)."""
        start = self.read_base()
        if start is None:
            return None

        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = vx, vy, wz
        end = time.time() + secs
        while time.time() < end:
            self.cmd.publish(msg)
            self.settle(0.05)
        self.cmd.publish(Twist())
        self.settle(1.0)

        here = self.read_base()
        if here is None:
            return None
        f, l = into_base(here[0] - start[0], here[1] - start[1], start[3])
        return f, l, here[3] - start[3]

    def hold_arm(self, positions, secs=4.0):
        """Move the arm to a pose. Returns the error code."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.velocities = [0.0] * len(positions)
        point.time_from_start.sec = int(secs)
        point.time_from_start.nanosec = int((secs % 1) * 1e9)
        goal.trajectory.points = [point]

        sent = self.arm.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent)
        handle = sent.result()
        if not handle.accepted:
            return None
        done = handle.get_result_async()
        rclpy.spin_until_future_complete(self, done)
        return done.result().result.error_code


# ------------------------------------------------------------------ track ---
def track(probe, ws, pose_name, seconds, period):
    """Hold an arm pose and track the base. Returns a summary."""
    x, y, yaw = APPROACH[ws]

    if not probe.place_base(x, y, yaw):
        print('  could not teleport the base')
        return None

    probe.settle(1.0)
    if probe.hold_arm(POSES[pose_name]) != 0:
        print('  arm refused the %s pose' % pose_name)
        return None

    # reset the base, moving the arm pushes it
    probe.place_base(x, y, yaw)
    probe.settle(1.5)

    start = probe.read_base()
    if start is None:
        print('  gazebo has no entity called %s' % ENTITY)
        return None

    ts, fwd, lat, yaws, zs = [], [], [], [], []
    t0 = time.time()
    while True:
        now = time.time() - t0
        here = probe.read_base()
        if here is not None:
            f, l = into_base(here[0] - start[0], here[1] - start[1], yaw)
            ts.append(now)
            fwd.append(f * 1000.0)
            lat.append(l * 1000.0)
            yaws.append(math.degrees(here[3] - start[3]))
            zs.append((here[2] - start[2]) * 1000.0)
        if now >= seconds:
            break
        probe.settle(period)

    return {'ws': ws, 'pose': pose_name, 'span': ts[-1],
            'fwd': fwd[-1], 'lat': lat[-1], 'yaw': yaws[-1], 'z': zs[-1],
            'rate': slope(ts, fwd), 'lat_rate': slope(ts, lat)}


def check_drive(probe):
    """Run the drive tests. Returns the number that failed."""
    x, y, yaw = DRIVE_POSE
    print('\n  %-8s %10s %10s %10s' % ('cmd_vel', 'want', 'got', 'off'))

    bad = 0
    for label, vx, vy, wz in DRIVES:
        probe.place_base(x, y, yaw)
        probe.settle(1.0)

        got = probe.drive(vx, vy, wz, DRIVE_SECS)
        if got is None:
            print('  %-8s %10s' % (label, 'no reading'))
            bad += 1
            continue

        f, l, t = got
        if wz:
            want, have, unit = wz * DRIVE_SECS, t, 'rad'
        elif vy:
            want, have, unit = vy * DRIVE_SECS, l, 'm'
        else:
            want, have, unit = vx * DRIVE_SECS, f, 'm'

        off = abs(have - want) / abs(want) if want else abs(have)
        if off > (TURN_TOL if wz else DRIVE_TOL):
            bad += 1
        print('  %-8s %10.3f %10.3f %9.0f%% %s'
              % (label, want, have, off * 100.0, unit))

    probe.place_base(x, y, yaw)
    probe.settle(1.0)
    return bad


# ----------------------------------------------------------------- report ---
def row(s):
    print('  %-6s %-10s %7.1f %8.1f %8.2f %8.2f %9.2f'
          % (s['ws'], s['pose'], s['fwd'], s['lat'], s['yaw'], s['z'],
             s['rate']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ws', default='WS01', choices=sorted(APPROACH),
                    help='which approach pose to stand on')
    ap.add_argument('--pose', default='ready', choices=sorted(POSES),
                    help='arm pose to hold while the base is watched')
    ap.add_argument('--sweep', action='store_true',
                    help='run every arm pose instead of just one')
    ap.add_argument('--seconds', type=float, default=30.0,
                    help='how long to watch, per pose')
    ap.add_argument('--period', type=float, default=1.0,
                    help='seconds between samples')
    ap.add_argument('--no-drive', action='store_true',
                    help='skip the check that the base still moves when told')
    args = ap.parse_args()

    rclpy.init()
    probe = Probe()
    if not probe.wait_services():
        print('gazebo and the arm controller are not both up')
        probe.destroy_node()
        rclpy.shutdown()
        return 1

    poses = sorted(POSES) if args.sweep else [args.pose]
    print('base hold, %s, %.0f s per pose' % (args.ws, args.seconds))
    print('  %-6s %-10s %7s %8s %8s %8s %9s'
          % ('ws', 'arm', 'fwd mm', 'left mm', 'yaw deg', 'z mm', 'mm/s'))

    results = []
    for name in poses:
        s = track(probe, args.ws, name, args.seconds, args.period)
        if s is not None:
            row(s)
            results.append(s)

    if not results:
        probe.destroy_node()
        rclpy.shutdown()
        print('\nnothing measured')
        return 1

    bad_drive = 0 if args.no_drive else check_drive(probe)

    worst = max(results, key=lambda s: abs(s['fwd']))
    probe.destroy_node()
    rclpy.shutdown()

    print('\nworst travel: %.1f mm over %.0f s, arm at %s'
          % (worst['fwd'], worst['span'], worst['pose']))
    print('a pick holding that pose for 15 s would close the fingers '
          '%.0f mm\npast where it aimed'
          % abs(worst['rate'] * 15.0))

    if bad_drive:
        print('%d of %d cmd_vel moves did not arrive - something that holds '
              'the base\nstill is also holding it back'
              % (bad_drive, len(DRIVES)))
        return 1
    if abs(worst['fwd']) > HOLD_MM:
        print('the base does not hold')
        return 1
    print('the base holds, and still drives')
    return 0


if __name__ == '__main__':
    sys.exit(main())
