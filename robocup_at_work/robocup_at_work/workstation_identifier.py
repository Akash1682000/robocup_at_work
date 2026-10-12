#!/usr/bin/env python3
"""Identify the current workstation from AprilTag detections.

Needs N consecutive frames with the same id and drops low decision
margins, so a glancing or settling view doesn't give a wrong answer.

Publishes:
  /workstation/current    std_msgs/String            stable id, latched
  /workstation/tag_pose   geometry_msgs/PoseStamped  tag pose in map frame
  ~/identify              std_srvs/Trigger           one-shot query for a sequencer
"""
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy

import tf2_ros
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger

# tag id -> (workstation name, tf frame published by apriltag_ros)
TAGS = {
    1: ('WS01', 'ws01_tag'),
    2: ('WS02', 'ws02_tag'),
    3: ('WS03', 'ws03_tag'),
    4: ('WS04', 'ws04_tag'),
}


class WorkstationIdentifier(Node):

    def __init__(self):
        super().__init__('workstation_identifier')

        self.declare_parameter('min_decision_margin', 30.0)
        self.declare_parameter('consecutive_frames', 5)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('detection_timeout', 2.0)

        self.min_margin = self.get_parameter('min_decision_margin').value
        self.need = int(self.get_parameter('consecutive_frames').value)
        self.map_frame = self.get_parameter('map_frame').value
        self.timeout = float(self.get_parameter('detection_timeout').value)

        self.history = deque(maxlen=self.need)
        self.stable = None
        self.last_seen = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # latched so late subscribers still get it
        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.pub_current = self.create_publisher(String, '/workstation/current', latched)
        self.pub_pose = self.create_publisher(PoseStamped, '/workstation/tag_pose', 10)

        self.create_subscription(AprilTagDetectionArray, '/detections', self.on_detections, 10)
        self.create_service(Trigger, '~/identify', self.on_identify)

        self.create_timer(1.0, self.check_stale)

        self.get_logger().info(
            f'workstation_identifier up: need {self.need} consecutive frames, '
            f'min decision margin {self.min_margin}')

    def on_detections(self, msg):
        best = None
        for det in msg.detections:
            if det.id not in TAGS:
                self.get_logger().warn(f'ignoring unknown tag id {det.id}', once=True)
                continue
            if det.decision_margin < self.min_margin:
                continue
            # keep the best decode
            if best is None or det.decision_margin > best.decision_margin:
                best = det

        if best is None:
            self.history.clear()
            return

        self.last_seen = self.get_clock().now()
        self.history.append(best.id)

        if len(self.history) < self.need or len(set(self.history)) != 1:
            return

        tag_id = self.history[0]
        name, frame = TAGS[tag_id]

        if self.stable != name:
            self.stable = name
            self.get_logger().info(
                f'identified {name} (tag {tag_id}, margin {best.decision_margin:.0f})')
            self.pub_current.publish(String(data=name))

        self.publish_pose(frame, msg.header.stamp)

    def publish_pose(self, frame, stamp):
        try:
            # latest transform, using stamp throws extrapolation errors
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, frame, rclpy.time.Time())
        except tf2_ros.TransformException as exc:
            self.get_logger().debug(f'no {self.map_frame}->{frame} yet: {exc}')
            return

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.map_frame
        pose.pose.position.x = tf.transform.translation.x
        pose.pose.position.y = tf.transform.translation.y
        pose.pose.position.z = tf.transform.translation.z
        pose.pose.orientation = tf.transform.rotation
        self.pub_pose.publish(pose)

    def check_stale(self):
        if self.stable is None or self.last_seen is None:
            return
        age = (self.get_clock().now() - self.last_seen).nanoseconds / 1e9
        if age > self.timeout:
            self.get_logger().info(f'lost sight of {self.stable}')
            self.stable = None
            self.history.clear()
            self.pub_current.publish(String(data=''))

    def on_identify(self, request, response):
        if self.stable:
            response.success = True
            response.message = self.stable
        else:
            response.success = False
            response.message = 'no workstation tag in view'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = WorkstationIdentifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
