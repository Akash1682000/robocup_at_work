#!/usr/bin/env python3
"""Wait until a service or action server is available.

Uses rclpy, `ros2 service list` can show stale entries from the daemon.

    python3 scripts/wait_for.py --service /pick/execute --timeout 90
    python3 scripts/wait_for.py \
        --action /panda_arm_controller/follow_joint_trajectory
"""
import argparse
import sys

import rclpy
from rclpy.node import Node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--service', action='append', default=[])
    ap.add_argument('--action', action='append', default=[])
    ap.add_argument('--topic', action='append', default=[])
    ap.add_argument('--graph-service', action='append', default=[],
                    help='a service name to wait for by graph lookup, for '
                         'the ones whose type is not worth importing here')
    ap.add_argument('--timeout', type=float, default=90.0)
    args = ap.parse_args()

    rclpy.init()
    node = Node('wait_for')
    deadline = args.timeout
    ok = True

    for name in args.service:
        from std_srvs.srv import Trigger
        cli = node.create_client(Trigger, name)
        if not cli.wait_for_service(timeout_sec=deadline):
            print('no service %s' % name)
            ok = False

    for name in args.action:
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory
        cli = ActionClient(node, FollowJointTrajectory, name)
        if not cli.wait_for_server(timeout_sec=deadline):
            print('no action server %s' % name)
            ok = False

    for name in args.topic:
        import time
        end = time.time() + deadline
        while time.time() < end and not node.count_publishers(name):
            rclpy.spin_once(node, timeout_sec=0.1)
        if not node.count_publishers(name):
            print('nothing publishing %s' % name)
            ok = False

    for name in args.graph_service:
        import time
        end = time.time() + deadline
        seen = False
        while time.time() < end and not seen:
            seen = any(n == name for n, _ in
                       node.get_service_names_and_types())
            if not seen:
                rclpy.spin_once(node, timeout_sec=0.2)
        if not seen:
            print('no service %s' % name)
            ok = False

    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
