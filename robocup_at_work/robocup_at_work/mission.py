#!/usr/bin/env python3
"""Mission sequencer: start area, then navigate, pick, navigate, place
for each job, then the end area.

Docking does not need to be exact, the picker works in base_footprint
from the tag and camera. A failed job releases the object.

  ros2 run robocup_at_work mission
  ros2 service call /mission/run std_srvs/srv/Trigger
"""
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import String
from std_srvs.srv import Trigger

import math

# approach poses in the map frame (same as the verify scripts)
APPROACH = {'WS01': (-2.00,  1.75,  1.5708),
            'WS02': (-2.50,  0.00,  3.1416),
            'WS03': ( 2.00,  1.75,  1.5708),
            'WS04': ( 2.50, -1.50,  0.0000)}

# start and end areas marked on the arena floor (robocup_arena.world)
AREA = {'START': ( 2.50, -2.20,  2.3500),
        'END':   (-2.80,  2.20,  3.1416)}


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


# ---------------------------------------------------------------- mission ---
class Mission(Node):
    def __init__(self):
        super().__init__('mission')
        self.set_parameters(
            [rclpy.parameter.Parameter('use_sim_time', value=True)])

        # source:class:destination, cylinders left out (less reliable)
        self.declare_parameter('jobs', ['WS01:F20_20_B:WS03',
                                        'WS02:S40_40_G:WS04'])
        self.declare_parameter('navigate', True)
        self.declare_parameter('settle', 3.0)
        self.declare_parameter('nav_timeout', 180.0)
        self.declare_parameter('arm_timeout', 240.0)

        group = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_state = self.create_publisher(String, '/mission/state',
                                               latched)
        self.nav = ActionClient(self, NavigateToPose, '/navigate_to_pose',
                                callback_group=group)
        self.params = self.create_client(
            SetParameters, '/object_picker/set_parameters',
            callback_group=group)
        self.pick = self.create_client(Trigger, '/pick/execute',
                                       callback_group=group)
        self.place = self.create_client(Trigger, '/pick/place',
                                        callback_group=group)
        self.release = self.create_client(Trigger, '/pick/release',
                                          callback_group=group)
        self.create_service(Trigger, '/mission/run', self.on_run,
                            callback_group=group)
        self.stage('idle')

    def param(self, name):
        return self.get_parameter(name).value

    def stage(self, text):
        self.pub_state.publish(String(data=text))
        self.get_logger().info(text)

    def wait(self, future, timeout):
        end = time.time() + timeout
        while time.time() < end and not future.done():
            time.sleep(0.02)
        return future.result() if future.done() else None

    def settle(self, seconds):
        time.sleep(seconds)

    # ------------------------------------------------------------- moves ---
    def goto(self, ws):
        """Drive to a workstation. True if it arrived."""
        if not self.param('navigate'):
            self.stage('skipping navigation to %s, as asked' % ws)
            return True

        x, y, yaw = AREA[ws] if ws in AREA else APPROACH[ws]
        if not self.nav.wait_for_server(timeout_sec=10.0):
            self.stage('no /navigate_to_pose - is navigation.launch.py up')
            return False

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        (goal.pose.pose.orientation.x, goal.pose.pose.orientation.y,
         goal.pose.pose.orientation.z,
         goal.pose.pose.orientation.w) = quat_from_yaw(yaw)

        # one retry, early nav2 aborts usually clear on their own
        for attempt in (1, 2):
            self.stage('driving to %s (%.2f, %.2f)%s'
                       % (ws, x, y, '' if attempt == 1 else ', retry'))
            sent = self.wait(self.nav.send_goal_async(goal), 20.0)
            if sent is None or not sent.accepted:
                self.stage('navigation would not take the goal for %s' % ws)
                continue
            done = self.wait(sent.get_result_async(),
                             float(self.param('nav_timeout')))
            if done is None:
                self.stage('navigation to %s timed out' % ws)
                continue
            # nav2 always returns a result, check the status
            if done.status == GoalStatus.STATUS_SUCCEEDED:
                return True
            self.stage('navigation to %s came back %d' % (ws, done.status))
            self.settle(2.0)
        return False

    def target_class(self, cls):
        req = SetParameters.Request()
        p = Parameter()
        p.name = 'target_class'
        p.value = ParameterValue(type=ParameterType.PARAMETER_STRING,
                                 string_value=cls)
        req.parameters = [p]
        if not self.params.wait_for_service(timeout_sec=10.0):
            return False
        return self.wait(self.params.call_async(req), 10.0) is not None

    def arm(self, client, what):
        """Call a picker service. Returns (ok, message)."""
        if not client.wait_for_service(timeout_sec=10.0):
            return False, 'no %s - is moveit.launch.py up' % what
        answer = self.wait(client.call_async(Trigger.Request()),
                           float(self.param('arm_timeout')))
        if answer is None:
            return False, '%s never answered' % what
        return answer.success, answer.message

    # --------------------------------------------------------------- run ---
    def job(self, spec):
        """Run one source:class:destination job. Returns (ok, message)."""
        try:
            src, cls, dst = spec.split(':')
        except ValueError:
            return False, '%r is not source:class:destination' % spec
        if src not in APPROACH or dst not in APPROACH:
            return False, '%s names a workstation that does not exist' % spec

        if not self.goto(src):
            return False, 'could not get to %s' % src

        # give the detector time to see the table
        self.settle(float(self.param('settle')))
        if not self.target_class(cls):
            return False, 'could not tell the picker to look for %s' % cls

        self.stage('picking %s at %s' % (cls, src))
        ok, said = self.arm(self.pick, '/pick/execute')
        if not ok:
            return False, 'pick failed at %s: %s' % (src, said)

        if not self.goto(dst):
            # can't reach the destination, drop it
            self.arm(self.release, '/pick/release')
            return False, 'picked %s but could not get to %s' % (cls, dst)

        self.settle(float(self.param('settle')))
        self.stage('placing %s at %s' % (cls, dst))
        ok, said = self.arm(self.place, '/pick/place')
        if not ok:
            self.arm(self.release, '/pick/release')
            return False, 'place failed at %s: %s' % (dst, said)
        return True, '%s from %s to %s' % (cls, src, dst)

    def on_run(self, request, response):
        jobs = list(self.param('jobs'))
        done, failed = [], []

        # already there after a fresh spawn, this just confirms it
        if not self.goto('START'):
            failed.append('could not get to the start area')

        for spec in jobs:
            ok, said = self.job(spec)
            (done if ok else failed).append(said)
            self.stage('%s: %s' % ('done' if ok else 'failed', said))

        # finish in the end area even if a job failed
        if not self.goto('END'):
            failed.append('could not get to the end area')
        else:
            self.stage('in the end area')

        self.stage('idle')
        response.success = not failed
        response.message = ('%d of %d jobs done%s'
                            % (len(done), len(jobs),
                               '' if not failed
                               else '; ' + '; '.join(failed)))
        return response


def main(args=None):
    rclpy.init(args=args)
    node = Mission()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
