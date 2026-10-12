#!/usr/bin/env python3
"""Pick and place node for the panda arm.

Top-down grasp: open, pre-grasp, straight descent, close, lift.
Collision objects are in base_footprint, so the scene is rebuilt on
every pick and must not be trusted across a drive.

  ros2 param set /object_picker target_class F20_20_B
  ros2 service call /pick/execute std_srvs/srv/Trigger
  ros2 service call /pick/release std_srvs/srv/Trigger

Needs gazebo, moveit.launch.py and perception.launch.py running.
"""
import json
import math
import os
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

import tf2_ros
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (AttachedCollisionObject, BoundingVolume,
                             CollisionObject, Constraints, JointConstraint,
                             MotionPlanRequest, OrientationConstraint,
                             PlanningScene, PositionConstraint,
                             WorkspaceParameters)
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.srv import GetCartesianPath
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint
from vision_msgs.msg import Detection3DArray

# realpath so --symlink-install resolves back to the source tree
PKG = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MANIFEST = os.path.join(PKG, 'config', 'objects_manifest.json')

ARM_GROUP = 'panda_arm'
ARM = ['panda_joint%d' % i for i in range(1, 8)]
IK_LINK = 'panda_link8'
READY = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

FINGERS = ['panda_finger_joint1', 'panda_finger_joint2']
HAND_ACTION = '/panda_hand_controller/follow_joint_trajectory'
# links the held object may touch
HAND_LINKS = ['panda_hand', 'panda_leftfinger', 'panda_rightfinger']
FINGER_LIMIT = 0.04         # per finger

# panda_link8 -> panda_tcp, see urdf/panda_arm.urdf.xacro
HAND_YAW = -math.pi / 4
TCP_OFFSET = 0.1034

# yaw sigma the detector uses for "no yaw" (cylinders)
YAW_UNKNOWN = math.radians(90.0)

# all tables are the same, tag centred on the front face
TABLE_WIDTH = 0.8
TABLE_DEPTH = 0.5
TABLE_THICKNESS = 0.1
TAG_TO_TOP = 0.05

TAG_FRAME = {'WS01': 'ws01_tag', 'WS02': 'ws02_tag',
             'WS03': 'ws03_tag', 'WS04': 'ws04_tag'}

TABLE_ID = 'table'


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


def quat_to_matrix(q):
    """x, y, z, w -> rows of the rotation matrix."""
    x, y, z, w = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def yaw_from_quat(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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
    out = (qx * cz + qy * sz,
           qy * cz - qx * sz,
           qz * cz + qw * sz,
           qw * cz - qz * sz)
    # tcp z points down here
    back = (position[0], position[1], position[2] + TCP_OFFSET)
    return back, out


def make_pose(position, quat):
    p = Pose()
    p.position.x, p.position.y, p.position.z = [float(v) for v in position]
    (p.orientation.x, p.orientation.y,
     p.orientation.z, p.orientation.w) = [float(v) for v in quat]
    return p


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------- picker ---
class ObjectPicker(Node):

    def __init__(self):
        super().__init__('object_picker')

        self.declare_parameter('target_class', '')
        self.declare_parameter('clearance', 0.10)
        self.declare_parameter('lift', 0.10)
        self.declare_parameter('min_score', 0.5)
        self.declare_parameter('detection_age', 3.0)
        self.declare_parameter('open_margin', 0.020)
        self.declare_parameter('squeeze', 0.005)
        self.declare_parameter('grasp_depth', 0.020)
        self.declare_parameter('tip_clearance', 0.003)
        self.declare_parameter('grip_tolerance', 0.008)
        self.declare_parameter('hold_margin', 0.0005)
        self.declare_parameter('close_step', 0.0005)
        self.declare_parameter('close_slow_at', 0.006)
        self.declare_parameter('stall_tolerance', 0.0006)
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('planning_attempts', 10)
        self.declare_parameter('velocity_scale', 0.3)
        self.declare_parameter('cartesian_step', 0.005)
        self.declare_parameter('cartesian_speed', 0.05)
        self.declare_parameter('table_margin', 0.02)
        self.declare_parameter('table_front', 0.50)
        self.declare_parameter('place_reach', 0.10)
        self.declare_parameter('place_side', 0.0)
        self.declare_parameter('retreat', True)

        self.table_top = 0.1
        try:
            with open(MANIFEST) as f:
                self.table_top = float(json.load(f)['table_top_z'])
        except (OSError, KeyError, ValueError) as exc:
            self.get_logger().warn(
                'no table height in the manifest (%s), assuming %.3f m'
                % (exc, self.table_top))

        self.lock = threading.Lock()
        self.detections = None
        self.joints = {}
        self.workstation = ''
        self.scene_ids = []
        self.attached = None
        self.held = None

        group = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.create_subscription(Detection3DArray, '/objects/poses',
                                 self.on_detections, 10, callback_group=group)
        self.create_subscription(JointState, '/joint_states',
                                 self.on_joints, 10, callback_group=group)
        self.create_subscription(String, '/workstation/current',
                                 self.on_workstation, latched,
                                 callback_group=group)

        self.pub_state = self.create_publisher(String, '/pick/state', latched)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.move = ActionClient(self, MoveGroup, '/move_action',
                                 callback_group=group)
        self.execute = ActionClient(self, ExecuteTrajectory,
                                    '/execute_trajectory',
                                    callback_group=group)
        self.hand = ActionClient(self, FollowJointTrajectory, HAND_ACTION,
                                 callback_group=group)
        self.weld = self.create_client(Trigger, '/grasp/attach',
                                       callback_group=group)
        self.unweld = self.create_client(Trigger, '/grasp/detach',
                                         callback_group=group)
        self.cartesian = self.create_client(GetCartesianPath,
                                            '/compute_cartesian_path',
                                            callback_group=group)
        self.scene = self.create_client(ApplyPlanningScene,
                                        '/apply_planning_scene',
                                        callback_group=group)

        self.create_service(Trigger, '/pick/execute', self.on_execute,
                            callback_group=group)
        self.create_service(Trigger, '/pick/place', self.on_place,
                            callback_group=group)
        self.create_service(Trigger, '/pick/release', self.on_release,
                            callback_group=group)

        self.stage('idle')
        self.get_logger().info(
            'object_picker up: /pick/execute picks, /pick/release lets go')

    # ------------------------------------------------------------ inputs ---
    def on_detections(self, msg):
        self.detections = msg

    def on_joints(self, msg):
        self.joints.update(zip(msg.name, msg.position))

    def on_workstation(self, msg):
        self.workstation = msg.data

    def param(self, name):
        return self.get_parameter(name).value

    def stage(self, text):
        self.pub_state.publish(String(data=text))
        self.get_logger().info(text)

    def wait(self, future, timeout):
        """Block on a future. Needs the multi-threaded executor."""
        end = time.time() + timeout
        while rclpy.ok() and not future.done() and time.time() < end:
            time.sleep(0.005)
        return future.result() if future.done() else None

    # ---------------------------------------------------------- services ---
    def on_execute(self, request, response):
        if not self.lock.acquire(blocking=False):
            response.success = False
            response.message = 'a pick is already running'
            return response
        try:
            response.success, response.message = self.pick()
        except Exception as exc:          # a thrown pick is a failed pick
            self.get_logger().error('pick threw: %s' % exc)
            response.success, response.message = False, 'pick threw: %s' % exc
        finally:
            self.stage('idle')
            self.lock.release()
        return response

    def on_place(self, request, response):
        if not self.lock.acquire(blocking=False):
            response.success = False
            response.message = 'a pick is already running'
            return response
        try:
            response.success, response.message = self.place()
        except Exception as exc:
            self.get_logger().error('place threw: %s' % exc)
            response.success, response.message = (False,
                                                  'place threw: %s' % exc)
        finally:
            self.stage('idle')
            self.lock.release()
        return response

    def on_release(self, request, response):
        if not self.lock.acquire(blocking=False):
            response.success = False
            response.message = 'a pick is already running'
            return response
        try:
            self.stage('release: opening')
            self.ungrip()
            code = self.gripper(FINGER_LIMIT, 1.0)
            self.detach()
            self.held = None
            response.success = code == 0
            response.message = ('open' if code == 0
                                else 'the hand controller returned %s' % code)
        finally:
            self.stage('idle')
            self.lock.release()
        return response

    # ------------------------------------------------------------- pick ----
    def pick(self):
        """One whole attempt. Returns (ok, what happened)."""
        if self.attached is not None:
            return False, ('already holding %s - call /pick/release first'
                           % self.attached)
        for client, what in ((self.move, '/move_action'),
                             (self.execute, '/execute_trajectory'),
                             (self.hand, HAND_ACTION)):
            if not client.wait_for_server(timeout_sec=5.0):
                return False, 'no action server on %s' % what
        for client, what in ((self.cartesian, '/compute_cartesian_path'),
                             (self.scene, '/apply_planning_scene')):
            if not client.wait_for_service(timeout_sec=5.0):
                return False, 'no service on %s' % what

        target, others = self.choose()
        if target is None:
            return False, others                  # others carries the reason

        name, position, yaw, size, known_yaw = target
        width = min(size[0], size[1])
        clearance = float(self.param('clearance'))

        self.stage('scene: table and %d other objects' % len(others))
        self.publish_scene(others, target)

        self.stage('open: %.1f mm across for a %.1f mm object'
                   % ((width + 2 * float(self.param('open_margin'))) * 1000.0,
                      width * 1000.0))
        opening = min(FINGER_LIMIT,
                      width / 2.0 + float(self.param('open_margin')))
        if self.gripper(opening, 1.0) != 0:
            return False, 'the hand controller would not open'

        # Grip below the centre of mass or the object tips over.
        # Only safe because the finger pad box ends at the tcp plane,
        # when it overran the tip this drove the hand into the table.
        floor = self.table_top + float(self.param('tip_clearance'))
        grasp_z = max(floor, position[2] - float(self.param('grasp_depth')))
        chosen = None
        for angle in self.candidates(position, yaw, known_yaw):
            pre = top_down_grasp(position[0], position[1],
                                 grasp_z + clearance, angle)
            self.stage('pre-grasp: %s at (%.3f, %.3f, %.3f) in '
                       'base_footprint, wrist %.0f deg, %.0f mm up'
                       % (name, position[0], position[1], grasp_z,
                          math.degrees(angle), clearance * 1000.0))
            if self.plan_to_pose(*tcp_to_link8(*pre)):
                chosen = angle
                break
        if chosen is None:
            return False, ('no plan to the pre-grasp for %s at any candidate '
                           'wrist angle' % name)

        self.drop_target_box()
        self.stage('approach: %.0f mm down' % (clearance * 1000.0))
        moved = self.straight(-clearance)
        if moved < 0.9:
            self.recover()
            return False, ('the descent only got %.0f%% of the way down'
                           % (moved * 100.0))

        self.stage('close')
        gap, holding = self.close_until_stall(opening, width)

        # Hold where the fingers stalled, otherwise they keep creeping
        # and squeeze the object out during the lift.
        if holding:
            hold = max(0.0, gap / 2.0 - float(self.param('hold_margin')))
            self.stage('hold: %.1f mm' % (gap * 1000.0))
            self.gripper(hold, 0.3)
            self.settle(0.2)
            gap = self.opening()

        if not holding:
            self.stage('close: nothing there, gap %.1f mm against %.1f mm'
                       % (gap * 1000.0, width * 1000.0))
            self.straight(clearance)
            self.recover()
            return False, ('closed on nothing: the fingers stopped %.1f mm '
                           'apart and %s is %.1f mm across'
                           % (gap * 1000.0, name, width * 1000.0))

        self.held = {'name': name, 'size': size, 'yaw': chosen}
        self.attach(name, position, chosen, size)
        if not self.grip():
            self.straight(clearance)
            self.recover()
            return False, ('the fingers closed on %s but there was nothing '
                           'between them to hold' % name)

        lift = float(self.param('lift'))
        self.stage('lift: %.0f mm' % (lift * 1000.0))
        risen = self.straight(lift) * lift

        if self.param('retreat'):
            self.stage('retreat: back to ready')
            if not self.plan_to_joints(READY):
                return True, ('holding %s at %.1f mm, but there was no plan '
                              'back to ready' % (name, gap * 1000.0))

        return True, ('picked %s: fingers %.1f mm apart against %.1f mm of '
                      'object, lifted %.0f mm'
                      % (name, gap * 1000.0, width * 1000.0, risen * 1000.0))

    # ------------------------------------------------------------ place ----
    def place(self):
        """Put the held object down on the table in front of the robot."""
        if self.held is None:
            return False, 'not holding anything'

        for client, what in ((self.move, '/move_action'),
                             (self.execute, '/execute_trajectory')):
            if not client.wait_for_server(timeout_sec=5.0):
                return False, 'no action server on %s' % what

        name = self.held['name']
        wrist = self.held['yaw']
        clearance = float(self.param('clearance'))
        rest = self.table_top + float(self.param('tip_clearance'))

        spot = self.place_spot()

        self.stage('scene: table, for the place')
        self.publish_scene([])

        above = top_down_grasp(spot[0], spot[1], rest + clearance, wrist)
        self.stage('pre-place: %s at (%.3f, %.3f) in base_footprint'
                   % (name, spot[0], spot[1]))
        if not self.plan_to_pose(*tcp_to_link8(*above)):
            return False, 'no plan to the pre-place pose for %s' % name

        self.stage('lower: %.0f mm down' % (clearance * 1000.0))
        moved = self.straight(-clearance)
        if moved < 0.9:
            return False, ('the descent to the table only got %.0f%% of the '
                           'way down' % (moved * 100.0))

        # release the weld before opening, or the object gets dragged
        self.stage('let go')
        self.ungrip()
        self.detach()
        self.gripper(FINGER_LIMIT, 1.0)
        self.settle(0.5)

        self.stage('clear: %.0f mm up' % (clearance * 1000.0))
        self.straight(clearance)

        placed = name
        self.held = None

        if self.param('retreat'):
            self.stage('retreat: back to ready')
            if not self.plan_to_joints(READY):
                return True, ('placed %s, but there was no plan back to ready'
                              % placed)
        return True, 'placed %s' % placed

    def place_spot(self):
        """Place point in base_footprint, place_side shifts it sideways."""
        inward, front = self.table_face()
        reach = front + float(self.param('place_reach'))
        side = float(self.param('place_side'))
        return (reach * inward[0] - side * inward[1],
                reach * inward[1] + side * inward[0])

    def recover(self):
        """Get the arm back out of the way after a failed attempt."""
        self.ungrip()
        self.gripper(FINGER_LIMIT, 1.0)
        if self.param('retreat'):
            self.plan_to_joints(READY)

    # ------------------------------------------------------ what to pick ---
    def choose(self):
        """Returns (target, others) or (None, reason).

        Detections are read once, the arm blocks the camera mid-pick.
        """
        msg = self.detections
        if msg is None or not msg.detections:
            return None, 'nothing on /objects/poses'

        age = (self.get_clock().now()
               - rclpy.time.Time.from_msg(msg.header.stamp)).nanoseconds / 1e9
        if age > float(self.param('detection_age')):
            return None, 'the last detection is %.1f s old' % age
        if msg.header.frame_id != 'base_footprint':
            return None, ('detections are in %s, not base_footprint'
                          % msg.header.frame_id)

        wanted = str(self.param('target_class'))
        floor = float(self.param('min_score'))

        found = []
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            if hyp.hypothesis.score < floor:
                continue
            centre = det.bbox.center
            quat = (centre.orientation.x, centre.orientation.y,
                    centre.orientation.z, centre.orientation.w)
            found.append({
                'cls': hyp.hypothesis.class_id,
                'score': hyp.hypothesis.score,
                'xyz': (centre.position.x, centre.position.y,
                        centre.position.z),
                'yaw': yaw_from_quat(quat),
                'size': (det.bbox.size.x, det.bbox.size.y, det.bbox.size.z),
                # [35] is the yaw variance, large for cylinders
                'known_yaw': hyp.pose.covariance[35] < YAW_UNKNOWN ** 2,
            })

        if not found:
            return None, ('no detection scored above %.2f' % floor)

        candidates = [f for f in found if not wanted or f['cls'] == wanted]
        if not candidates:
            return None, ('no %s in view - there is %s'
                          % (wanted, ', '.join(sorted({f['cls']
                                                       for f in found}))))

        # nearest one first
        best = min(candidates, key=lambda f: math.hypot(f['xyz'][0],
                                                        f['xyz'][1]))
        others = [f for f in found if f is not best]
        return ((best['cls'], best['xyz'], best['yaw'], best['size'],
                 best['known_yaw']), others)

    def candidates(self, position, yaw, known_yaw):
        """Wrist angles to try, the one closest to neutral first."""
        bearing = math.atan2(position[1], position[0])
        if known_yaw:
            near = abs(wrap(yaw - bearing)) < math.pi / 2
            first = yaw if near else yaw + math.pi
            return [wrap(first), wrap(first + math.pi)]
        return [wrap(bearing + k * math.pi / 2) for k in range(4)]

    # ------------------------------------------------------------ scene ----
    def publish_scene(self, others, target=None):
        """Rebuild the scene: table plus objects.

        The target stays in until the pre-grasp is reached, otherwise
        the planner swings the hand through it.
        """
        objects = [self.clear(i) for i in self.scene_ids]
        self.scene_ids = []

        table = self.table()
        if table is not None:
            objects.append(table)
            self.scene_ids.append(TABLE_ID)

        for n, other in enumerate(others):
            name = 'obj%d_%s' % (n, other['cls'])
            objects.append(self.box(name, other['xyz'], other['yaw'],
                                    other['size']))
            self.scene_ids.append(name)

        self.target_id = None
        if target is not None:
            name, position, yaw, size, _ = target
            self.target_id = 'target_%s' % name
            objects.append(self.box(self.target_id, position, yaw, size))
            self.scene_ids.append(self.target_id)

        self.apply(objects)

    def drop_target_box(self):
        """Remove the target box before the descent."""
        if getattr(self, 'target_id', None) is None:
            return
        self.apply([self.clear(self.target_id)])
        if self.target_id in self.scene_ids:
            self.scene_ids.remove(self.target_id)
        self.target_id = None

    def table_face(self):
        """Returns (inward unit vector, distance to table edge) from the tag."""
        inward = (1.0, 0.0)
        front = float(self.param('table_front'))
        frame = TAG_FRAME.get(self.workstation)

        if frame:
            try:
                tf = self.tf_buffer.lookup_transform(
                    'base_footprint', frame, rclpy.time.Time())
            except tf2_ros.TransformException as exc:
                self.get_logger().warn('no base_footprint->%s (%s), placing '
                                       'the table straight ahead'
                                       % (frame, exc))
            else:
                t = tf.transform.translation
                r = tf.transform.rotation
                rows = quat_to_matrix((r.x, r.y, r.z, r.w))
                n = (rows[0][2], rows[1][2])
                if math.hypot(*n) > 1e-6:
                    if n[0] * t.x + n[1] * t.y < 0.0:
                        n = (-n[0], -n[1])
                    scale = math.hypot(*n)
                    inward = (n[0] / scale, n[1] / scale)
                front = t.x * inward[0] + t.y * inward[1]
        else:
            self.get_logger().warn('no workstation on /workstation/current, '
                                   'placing the table straight ahead')
        return inward, front

    def table(self):
        """The table slab as a collision object, in base_footprint."""
        inward, front = self.table_face()
        margin = float(self.param('table_margin'))
        centre = front + TABLE_DEPTH / 2.0
        obj = CollisionObject()
        obj.header.frame_id = 'base_footprint'
        obj.id = TABLE_ID
        obj.operation = CollisionObject.ADD
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [TABLE_DEPTH + 2 * margin, TABLE_WIDTH + 2 * margin,
                           TABLE_THICKNESS]
        obj.primitives = [prim]
        obj.primitive_poses = [make_pose(
            (centre * inward[0], centre * inward[1],
             self.table_top - TABLE_THICKNESS / 2.0),
            quat_from_yaw(math.atan2(inward[1], inward[0])))]
        return obj

    def box(self, name, position, yaw, size):
        obj = CollisionObject()
        obj.header.frame_id = 'base_footprint'
        obj.id = name
        obj.operation = CollisionObject.ADD
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(s) for s in size]
        obj.primitives = [prim]
        obj.primitive_poses = [make_pose(position, quat_from_yaw(yaw))]
        return obj

    def clear(self, name):
        obj = CollisionObject()
        obj.header.frame_id = 'base_footprint'
        obj.id = name
        obj.operation = CollisionObject.REMOVE
        return obj

    def apply(self, objects, attached=None):
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects = objects
        if attached is not None:
            scene.robot_state.attached_collision_objects = [attached]
        req = ApplyPlanningScene.Request()
        req.scene = scene
        res = self.wait(self.scene.call_async(req), 10.0)
        if res is None or not res.success:
            self.get_logger().warn('/apply_planning_scene did not take the '
                                   'update')
        # let the scene monitor catch up
        self.settle(0.3)
        return res is not None and res.success

    def attach(self, name, position, yaw, size):
        """Attach the object in MoveIt, call while the fingers are on it."""
        held = AttachedCollisionObject()
        held.link_name = 'panda_hand'
        held.object = self.box('held_%s' % name, position, yaw, size)
        held.touch_links = list(HAND_LINKS)
        self.apply([], attached=held)
        self.attached = 'held_%s' % name

    def grip(self):
        """Weld the object to the hand in gazebo (robocup_grasp plugin).

        Returns False if nothing is between the fingers.
        """
        if not self.weld.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('no /grasp/attach, is the plugin loaded')
            return False
        answer = self.wait(self.weld.call_async(Trigger.Request()), 5.0)
        if answer is None:
            self.get_logger().warn('/grasp/attach never answered')
            return False
        if not answer.success:
            self.get_logger().warn('/grasp/attach: %s' % answer.message)
        return answer.success

    def ungrip(self):
        """Remove the gazebo weld. Safe when holding nothing."""
        if not self.unweld.wait_for_service(timeout_sec=2.0):
            return
        self.wait(self.unweld.call_async(Trigger.Request()), 5.0)

    def detach(self):
        if self.attached is None:
            return
        gone = AttachedCollisionObject()
        gone.link_name = 'panda_hand'
        gone.object.id = self.attached
        gone.object.operation = CollisionObject.REMOVE
        self.apply([self.clear(self.attached)], attached=gone)
        self.attached = None

    # ------------------------------------------------------------ motion ---
    def plan_to_pose(self, position, quat):
        """Plan and execute IK_LINK to a pose."""
        goal = Constraints()

        pos = PositionConstraint()
        pos.header.frame_id = 'base_footprint'
        pos.link_name = IK_LINK
        pos.weight = 1.0
        region = BoundingVolume()
        ball = SolidPrimitive()
        ball.type = SolidPrimitive.SPHERE
        ball.dimensions = [0.005]
        region.primitives = [ball]
        region.primitive_poses = [make_pose(position, (0.0, 0.0, 0.0, 1.0))]
        pos.constraint_region = region
        goal.position_constraints = [pos]

        ori = OrientationConstraint()
        ori.header.frame_id = 'base_footprint'
        ori.link_name = IK_LINK
        (ori.orientation.x, ori.orientation.y,
         ori.orientation.z, ori.orientation.w) = [float(v) for v in quat]
        ori.absolute_x_axis_tolerance = 0.02
        ori.absolute_y_axis_tolerance = 0.02
        ori.absolute_z_axis_tolerance = 0.02
        ori.weight = 1.0
        goal.orientation_constraints = [ori]

        return self.run_move_group(goal)

    def plan_to_joints(self, positions):
        goal = Constraints()
        for name, value in zip(ARM, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(value)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            goal.joint_constraints.append(jc)
        return self.run_move_group(goal)

    def run_move_group(self, constraints):
        req = MotionPlanRequest()
        req.group_name = ARM_GROUP
        req.num_planning_attempts = int(self.param('planning_attempts'))
        req.allowed_planning_time = float(self.param('planning_time'))
        scale = float(self.param('velocity_scale'))
        req.max_velocity_scaling_factor = scale
        req.max_acceleration_scaling_factor = scale
        req.goal_constraints = [constraints]
        req.start_state.is_diff = True

        space = WorkspaceParameters()
        space.header.frame_id = 'base_footprint'
        space.min_corner.x = space.min_corner.y = -1.5
        space.min_corner.z = -0.1
        space.max_corner.x = space.max_corner.y = space.max_corner.z = 1.5
        req.workspace_parameters = space

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.replan = False
        goal.planning_options.look_around = False

        sent = self.wait(self.move.send_goal_async(goal), 10.0)
        if sent is None or not sent.accepted:
            self.get_logger().warn('move_group would not take the goal')
            return False
        done = self.wait(sent.get_result_async(),
                         float(self.param('planning_time')) * 4 + 60.0)
        if done is None:
            self.get_logger().warn('move_group never came back')
            return False
        code = done.result.error_code.val
        if code != 1:
            self.get_logger().warn('move_group returned error %d' % code)
        return code == 1

    def straight(self, dz):
        """Move IK_LINK straight up or down by dz. Returns the fraction done.

        Collision checking is off since the fingers are meant to be close
        to the object and table.
        """
        start = self.link8_pose()
        if start is None:
            return 0.0
        position, quat = start
        goal = make_pose((position[0], position[1], position[2] + dz), quat)

        req = GetCartesianPath.Request()
        req.header.frame_id = 'base_footprint'
        req.group_name = ARM_GROUP
        req.link_name = IK_LINK
        req.waypoints = [goal]
        req.max_step = float(self.param('cartesian_step'))
        req.jump_threshold = 0.0
        req.avoid_collisions = False
        req.start_state.is_diff = True

        res = self.wait(self.cartesian.call_async(req), 20.0)
        if (res is None or res.error_code.val != 1
                or not res.solution.joint_trajectory.points):
            self.get_logger().warn('no cartesian path for %.0f mm'
                                   % (dz * 1000.0))
            return 0.0

        speed = float(self.param('cartesian_speed'))
        self.retime(res.solution.joint_trajectory,
                    abs(dz) * res.fraction / speed)
        if not self.run_trajectory(res.solution):
            return 0.0
        self.settle(0.3)
        end = self.link8_pose()
        self.get_logger().info(
            'cartesian %.0f mm: %.0f%% of the path, %s ended at z %.4f '
            '(asked for %.4f)'
            % (dz * 1000.0, res.fraction * 100.0, IK_LINK,
               end[0][2] if end else float('nan'), position[2] + dz))
        return res.fraction

    def retime(self, trajectory, seconds):
        """Add timestamps if the cartesian path came back without them."""
        points = trajectory.points
        if len(points) < 2:
            return
        last = points[-1].time_from_start
        if last.sec != 0 or last.nanosec != 0:
            return
        seconds = max(seconds, 0.5)
        for n, point in enumerate(points):
            t = seconds * n / (len(points) - 1)
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t % 1.0) * 1e9)
            point.velocities = []
            point.accelerations = []

    def run_trajectory(self, solution):
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = solution
        sent = self.wait(self.execute.send_goal_async(goal), 10.0)
        if sent is None or not sent.accepted:
            self.get_logger().warn('/execute_trajectory would not take it')
            return False
        done = self.wait(sent.get_result_async(), 60.0)
        if done is None:
            self.get_logger().warn('/execute_trajectory never came back')
            return False
        code = done.result.error_code.val
        if code != 1:
            self.get_logger().warn('/execute_trajectory returned %d' % code)
        return code == 1

    def link8_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform('base_footprint', IK_LINK,
                                                 rclpy.time.Time())
        except tf2_ros.TransformException as exc:
            self.get_logger().warn('no base_footprint->%s: %s'
                                   % (IK_LINK, exc))
            return None
        t, r = tf.transform.translation, tf.transform.rotation
        return (t.x, t.y, t.z), (r.x, r.y, r.z, r.w)

    # ----------------------------------------------------------- gripper ---
    def gripper(self, finger, seconds):
        """Move both fingers to a per-finger position. Returns the error code."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(FINGERS)
        point = JointTrajectoryPoint()
        point.positions = [float(finger)] * 2
        point.velocities = [0.0] * 2
        point.time_from_start.sec = int(seconds)
        point.time_from_start.nanosec = int((seconds % 1) * 1e9)
        goal.trajectory.points = [point]

        sent = self.wait(self.hand.send_goal_async(goal), 10.0)
        if sent is None or not sent.accepted:
            return None
        done = self.wait(sent.get_result_async(), seconds + 10.0)
        return None if done is None else done.result.error_code

    def close_until_stall(self, start, expect):
        """Close step by step until the fingers stall on the object.

        Returns (gap, stalled).
        """
        fine = float(self.param('close_step'))
        coarse = fine * 4.0
        slow_at = expect / 2.0 + float(self.param('close_slow_at'))
        tol = float(self.param('stall_tolerance'))
        target = float(start)

        while target > 0.0:
            # Coarse steps in open air, fine near the object. expect only
            # sets where to slow down, never where to stop, so a wrong bbox
            # width is harmless.
            step = fine if target <= slow_at else coarse
            target = max(0.0, target - step)
            if self.gripper(target, 0.25) is None:
                break
            self.settle(0.15)
            gap = self.opening()
            if gap - 2.0 * target > tol:
                return gap, True
        return self.opening(), False

    def opening(self):
        """Gap between the fingertips from the joint states."""
        return sum(self.joints.get(f, 0.0) for f in FINGERS)

    def settle(self, seconds):
        time.sleep(seconds)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectPicker()
    # picks block inside service callbacks, so we need extra threads
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
