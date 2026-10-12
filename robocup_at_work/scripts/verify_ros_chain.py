#!/usr/bin/env python3
"""End to end check of the AprilTag chain.

Checks /detections, the map->ws0X_tag TF against the world file, and
/workstation/current. Needs:
  ros2 launch robocup_at_work robot_gazebo.launch.py
  ros2 launch robocup_at_work navigation.launch.py use_amcl:=False
  ros2 launch robocup_at_work perception.launch.py

use_amcl:=False makes map equal to the world frame.
"""
import math, subprocess, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
import tf2_ros
from apriltag_msgs.msg import AprilTagDetectionArray
from std_msgs.msg import String

# name, robot x, y, yaw, tag id, tag (x, y, z) from the world file
POSES = [('WS01', -2.00, 1.75, 1.5708, 1, (-2.000,  2.248, 0.05)),
         ('WS02', -2.50, 0.00, 3.1416, 2, (-3.048,  0.000, 0.05)),
         ('WS03',  2.00, 1.75, 1.5708, 3, ( 2.000,  2.248, 0.05)),
         ('WS04',  2.50, -1.50, 0.0000, 4, ( 3.048, -1.500, 0.05))]
ENTITY = 'robocup_robot'


class Watch(Node):
    def __init__(self):
        super().__init__('ros_tag_verifier')
        self.dets, self.current = [], None
        self.buf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.buf, self)
        self.create_subscription(AprilTagDetectionArray, '/detections', self._d, 10)
        # TRANSIENT_LOCAL to get the latched value
        self.create_subscription(String, '/workstation/current', self._c,
                                 QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def _d(self, m): self.dets = list(m.detections)
    def _c(self, m): self.current = m.data


def teleport(x, y, yaw):
    subprocess.run(['gz', 'model', '-m', ENTITY, '-x', str(x), '-y', str(y),
                    '-z', '0.05', '-R', '0', '-P', '0', '-Y', str(yaw)],
                   check=False, capture_output=True, text=True)


def main():
    rclpy.init()
    n = Watch()
    print(f'{"WS":6}{"id":>3}{"marg":>6}{"ham":>4}  {"current":9}'
          f'{"map->tag (x, y, z)":30}{"ground truth":30}{"err":>8}')
    print('-' * 98)
    ok = True
    for name, x, y, yaw, want, gt in POSES:
        n.current, n.dets = None, []
        teleport(x, y, yaw)
        end = time.time() + 12.0
        while time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.1)
        got = n.dets[0].id if n.dets else None
        marg = n.dets[0].decision_margin if n.dets else float('nan')
        ham = n.dets[0].hamming if n.dets else -1
        try:
            t = n.buf.lookup_transform('map', f'ws0{want}_tag',
                                       rclpy.time.Time()).transform.translation
            tf_s = f'({t.x:+.3f}, {t.y:+.3f}, {t.z:+.3f})'
            err = math.dist((t.x, t.y, t.z), gt)
            err_s, tf_ok = f'{err*100:.1f} cm', err < 0.10
        except Exception as e:
            tf_s, err_s, tf_ok = f'NO TF ({type(e).__name__})', '-', False
        good = got == want and n.current == name and tf_ok
        ok &= good
        print(f'{name:6}{str(got):>3}{marg:>6.0f}{ham:>4}  {str(n.current):9}'
              f'{tf_s:30}{str(gt):30}{err_s:>8}{"" if good else "  <-- FAIL"}')
    print('-' * 98)
    print('ALL FOUR PASS' if ok else 'FAILURES PRESENT')
    n.destroy_node(); rclpy.shutdown()


main()
