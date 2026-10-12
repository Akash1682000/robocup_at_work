#!/usr/bin/env python3
"""YOLO object detector with 3D poses.

Position comes from intersecting the box with the known table plane,
yaw from the principal axis of the depth cloud.

Run it with the yolo_env python (perception.launch.py does this).
No cv_bridge here, it segfaults next to the venv's opencv.

Publishes /objects/detections (2D), /objects/poses (3D), /objects/overlay.
"""
import json
import math
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import (BoundingBox2D, BoundingBox3D, Detection2D,
                             Detection2DArray, Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose, Pose2D, Point2D)

# realpath so --symlink-install resolves back to the source tree
PKG = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
WEIGHTS = os.path.join(PKG, 'runs', 'robocup', 'weights', 'best.pt')
MANIFEST = os.path.join(PKG, 'config', 'objects_manifest.json')

# square footprint = upright cylinder, no yaw
SQUARE_TOL = 0.005

FIT_ROUNDS = 4              # centre pixel refinement steps
YAW_STEP = 1.0              # deg
IOU_BAND = 0.02             # IoU within this of the best counts as a tie

# measured noise levels
POS_SIGMA = 0.005           # m
Z_SIGMA = 0.001             # m
TILT_SIGMA = 0.01           # rad
YAW_FLOOR = math.radians(15.0)
YAW_UNKNOWN = math.radians(90.0)  # cylinders or failed fit
DEPTH_YAW_SIGMA = math.radians(5.0)

DEPTH_MARGIN = 0.015        # m, crop padding
DEPTH_FLOOR = 0.004         # m above table to count as object
DEPTH_MIN_POINTS = 60
DEPTH_MIN_ELONGATION = 1.4  # long/short axis ratio
DEPTH_MAX_AGE = 0.5         # s


# --------------------------------------------------------------- geometry ---
def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def corner_offsets(bbox):
    """8 corners of a box centred on the origin, shape (8, 3)."""
    hx, hy, hz = bbox[0] / 2.0, bbox[1] / 2.0, bbox[2] / 2.0
    return np.array([[sx, sy, sz]
                     for sx in (-hx, hx)
                     for sy in (-hy, hy)
                     for sz in (-hz, hz)])


def extent_iou(u, v, px):
    """IoU of each row's pixel extent against one observed box."""
    a1, b1, a2, b2 = u.min(1), v.min(1), u.max(1), v.max(1)
    ix = np.clip(np.minimum(a2, px[2]) - np.maximum(a1, px[0]), 0.0, None)
    iy = np.clip(np.minimum(b2, px[3]) - np.maximum(b1, px[1]), 0.0, None)
    inter = ix * iy
    union = ((a2 - a1) * (b2 - b1)
             + (px[2] - px[0]) * (px[3] - px[1]) - inter)
    return inter / np.maximum(union, 1e-9)


# ----------------------------------------------------------------- images ---
def img_to_bgr(msg):
    a = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    return a[:, :, ::-1].copy() if msg.encoding == 'rgb8' else a.copy()


def bgr_to_img(bgr, header):
    msg = Image()
    msg.header = header
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = bgr.tobytes()
    return msg


# ------------------------------------------------------------------- node ---
class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')

        self.declare_parameter('weights', WEIGHTS)
        self.declare_parameter('manifest', MANIFEST)
        self.declare_parameter('image_topic', '/rgb_camera/image_raw')
        self.declare_parameter('camera_info_topic', '/rgb_camera/camera_info')
        self.declare_parameter('target_frame', 'base_footprint')
        self.declare_parameter('camera_frame', 'camera_rgb_optical_frame')
        self.declare_parameter('depth_topic', '/depth_camera/depth/image_raw')
        self.declare_parameter('depth_info_topic',
                               '/depth_camera/depth/camera_info')
        self.declare_parameter('depth_frame', 'camera_depth_optical_frame')
        self.declare_parameter('use_depth', True)
        self.declare_parameter('table_top_z', 0.10)
        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('max_rate', 10.0)
        self.declare_parameter('device', '')
        self.declare_parameter('publish_overlay', True)

        gp = self.get_parameter
        self.target_frame = gp('target_frame').value
        self.camera_frame = gp('camera_frame').value
        self.depth_frame = gp('depth_frame').value
        self.use_depth = bool(gp('use_depth').value)
        self.table_top_z = float(gp('table_top_z').value)
        self.conf = float(gp('confidence').value)
        self.device = gp('device').value
        self.overlay = bool(gp('publish_overlay').value)
        rate = float(gp('max_rate').value)
        self.min_gap = 1.0 / rate if rate > 0.0 else 0.0
        self.last_run = 0.0

        self.specs = self.load_specs(gp('manifest').value)
        self.model = self.load_model(gp('weights').value)
        self.names = self.model.names

        missing = [n for n in self.names.values() if n not in self.specs]
        if missing:
            self.get_logger().warn(
                'no manifest entry for %s, those stay 2D only'
                % ', '.join(missing))

        self.info = None
        self.depth = None
        self.depth_info = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_2d = self.create_publisher(
            Detection2DArray, '/objects/detections', 10)
        self.pub_3d = self.create_publisher(
            Detection3DArray, '/objects/poses', 10)
        self.pub_overlay = self.create_publisher(Image, '/objects/overlay', 1)

        self.create_subscription(CameraInfo, gp('camera_info_topic').value,
                                 self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, gp('image_topic').value,
                                 self.on_image, qos_profile_sensor_data)
        if self.use_depth:
            self.create_subscription(Image, gp('depth_topic').value,
                                     self.on_depth, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, gp('depth_info_topic').value,
                                     self.on_depth_info,
                                     qos_profile_sensor_data)

        self.get_logger().info(
            'object_detector up: %d classes, conf %.2f, poses in %s, '
            'yaw from %s'
            % (len(self.names), self.conf, self.target_frame,
               'the depth cloud' if self.use_depth else 'the box only'))

# ------------------------------------------------------------------ setup ---
    def load_specs(self, path):
        """class name -> (bbox, rest height, is_cylinder)."""
        with open(path) as fh:
            manifest = json.load(fh)
        out = {}
        for c in manifest['classes']:
            bbox = tuple(c['bbox'])
            out[c['name']] = (bbox, c['rest_z'],
                              abs(bbox[0] - bbox[1]) < SQUARE_TOL)
        return out

    def load_model(self, weights):
        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().fatal(
                'no ultralytics here. Start this node with the venv python, '
                'see the module docstring.')
            raise
        if not os.path.exists(weights):
            self.get_logger().fatal(
                '%s is missing - train first with scripts/train_yolo.py'
                % weights)
            raise SystemExit(1)
        self.get_logger().info('loading %s' % weights)
        return YOLO(weights)

# ----------------------------------------------------------------- inputs ---
    def on_info(self, msg):
        self.info = msg

    def on_depth(self, msg):
        self.depth = msg

    def on_depth_info(self, msg):
        self.depth_info = msg

    def on_image(self, msg):
        if self.info is None:
            self.get_logger().warn('waiting on camera_info', once=True)
            return
        now = time.time()
        if now - self.last_run < self.min_gap:
            return
        self.last_run = now

        frame = img_to_bgr(msg)
        kwargs = {'conf': self.conf, 'verbose': False}
        if self.device:
            kwargs['device'] = self.device
        result = self.model.predict(frame, **kwargs)[0]

        found = []
        for box in result.boxes:
            u1, v1, u2, v2 = [float(c) for c in box.xyxy[0].tolist()]
            found.append({'cls': self.names[int(box.cls[0])],
                          'score': float(box.conf[0]),
                          'px': (u1, v1, u2, v2)})

        self.pub_2d.publish(self.make_2d(found, msg.header))

        if self.camera_pose():
            for det in found:
                self.locate(det)
            if self.use_depth and self.depth_pose(msg.header.stamp):
                self.yaw_from_cloud(found)
            self.pub_3d.publish(self.make_3d(found, msg.header))

        if self.overlay:
            self.pub_overlay.publish(
                bgr_to_img(self.draw(frame, found), msg.header))

    def camera_pose(self):
        """Cache the camera pose. False until TF is available."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, self.camera_frame, rclpy.time.Time())
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(
                'no %s -> %s yet: %s'
                % (self.target_frame, self.camera_frame, exc),
                throttle_duration_sec=5.0)
            return False
        tr, ro = tf.transform.translation, tf.transform.rotation
        self.rot = quat_to_mat((ro.x, ro.y, ro.z, ro.w))
        self.org = np.array([tr.x, tr.y, tr.z])
        k = self.info.k
        self.fx, self.fy, self.cx, self.cy = k[0], k[4], k[2], k[5]
        return True

# ---------------------------------------------------------------- the fit ---
    def lift(self, u, v, plane_z):
        """Intersect pixel rays with a horizontal plane. Returns (N, 3)."""
        d = np.stack([(u - self.cx) / self.fx,
                      (v - self.cy) / self.fy,
                      np.ones_like(u)], axis=1) @ self.rot.T
        with np.errstate(divide='ignore', invalid='ignore'):
            s = (plane_z - self.org[2]) / d[:, 2]
        return self.org + s[:, None] * d

    def project(self, pts):
        """Points (N, 8, 3) in target_frame to pixels, as two (N, 8) arrays."""
        local = (pts.reshape(-1, 3) - self.org) @ self.rot
        z = np.clip(local[:, 2], 1e-6, None)
        shape = pts.shape[:2]
        return ((self.fx * local[:, 0] / z + self.cx).reshape(shape),
                (self.fy * local[:, 1] / z + self.cy).reshape(shape))

    def corners(self, xyz, yaws, offsets):
        """Box corners for every candidate pose, as (N, 8, 3)."""
        c, s = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
        lx, ly, lz = offsets[:, 0], offsets[:, 1], offsets[:, 2]
        return np.stack([
            xyz[:, 0][:, None] + lx * c - ly * s,
            xyz[:, 1][:, None] + lx * s + ly * c,
            xyz[:, 2][:, None] + lz], axis=-1)

    def locate(self, det):
        """Add 3D position, yaw and yaw sigma to a detection."""
        spec = self.specs.get(det['cls'])
        if spec is None:
            return
        bbox, rest_z, round_class = spec
        plane_z = self.table_top_z + rest_z
        px = det['px']
        ou, ov = (px[0] + px[2]) / 2.0, (px[1] + px[3]) / 2.0

        if round_class:
            yaws = np.zeros(1)
        else:
            yaws = np.radians(np.arange(0.0, 180.0, YAW_STEP))
        offsets = corner_offsets(bbox)

        # shift the centre pixel until the projected box matches the detection
        u = np.full(yaws.shape, ou)
        v = np.full(yaws.shape, ov)
        xyz = self.lift(u, v, plane_z)
        if not np.all(np.isfinite(xyz)):
            return
        for _ in range(FIT_ROUNDS):
            mu, mv = self.project(self.corners(xyz, yaws, offsets))
            u += ou - (mu.min(1) + mu.max(1)) / 2.0
            v += ov - (mv.min(1) + mv.max(1)) / 2.0
            xyz = self.lift(u, v, plane_z)

        mu, mv = self.project(self.corners(xyz, yaws, offsets))
        ious = extent_iou(mu, mv, px)
        best = int(np.argmax(ious))

        det['xyz'] = xyz[best]
        det['bbox'] = bbox
        det['yaw'] = float(yaws[best])
        det['iou'] = float(ious[best])
        det['yaw_sigma'] = self.yaw_spread(ious, best, round_class)

    def yaw_spread(self, ious, best, round_class):
        """Yaw sigma from the width of the IoU peak."""
        if round_class or len(ious) == 1:
            return YAW_UNKNOWN
        tied = np.nonzero(ious >= ious[best] - IOU_BAND)[0]
        step = math.radians(YAW_STEP)
        gap = np.abs(tied - best) * step
        gap = np.minimum(gap, math.pi - gap)
        return max(float(gap.max()), YAW_FLOOR)

# -------------------------------------------------------------- depth yaw ---
    def depth_pose(self, stamp):
        """Cache the depth camera pose and intrinsics. False if not ready."""
        if self.depth is None or self.depth_info is None:
            self.get_logger().warn('no depth yet, yaw stays on the box',
                                   throttle_duration_sec=10.0)
            return False
        age = abs(stamp.sec - self.depth.header.stamp.sec
                  + (stamp.nanosec - self.depth.header.stamp.nanosec) / 1e9)
        if age > DEPTH_MAX_AGE:
            self.get_logger().warn(
                'depth frame is %.1f s stale, skipping' % age,
                throttle_duration_sec=10.0)
            return False
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, self.depth_frame, rclpy.time.Time())
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(
                'no %s -> %s yet: %s'
                % (self.target_frame, self.depth_frame, exc),
                throttle_duration_sec=5.0)
            return False
        tr, ro = tf.transform.translation, tf.transform.rotation
        self.d_rot = quat_to_mat((ro.x, ro.y, ro.z, ro.w))
        self.d_org = np.array([tr.x, tr.y, tr.z])
        k = self.depth_info.k
        self.d_fx, self.d_fy, self.d_cx, self.d_cy = k[0], k[4], k[2], k[5]
        return True

    def depth_window(self, xyz, radius, shape):
        """Pixel window around xyz, or None if off the image."""
        loc = self.d_rot.T @ (xyz - self.d_org)
        if loc[2] < 1e-3:
            return None
        u = self.d_fx * loc[0] / loc[2] + self.d_cx
        v = self.d_fy * loc[1] / loc[2] + self.d_cy
        ru = self.d_fx * radius / loc[2]
        rv = self.d_fy * radius / loc[2]
        height, width = shape
        u0, u1 = max(0, int(u - ru)), min(width, int(u + ru) + 1)
        v0, v1 = max(0, int(v - rv)), min(height, int(v + rv) + 1)
        if u1 <= u0 or v1 <= v0:
            return None
        return u0, v0, u1, v1

    def window_cloud(self, depth, window):
        """Depth window to points in target_frame."""
        u0, v0, u1, v1 = window
        z = depth[v0:v1, u0:u1].ravel()
        vs, us = np.mgrid[v0:v1, u0:u1]
        us = us.ravel().astype(np.float64)
        vs = vs.ravel().astype(np.float64)
        ok = np.isfinite(z) & (z > 0.05)
        if not ok.any():
            return np.empty((0, 3))
        z, us, vs = z[ok], us[ok], vs[ok]
        local = np.stack([(us - self.d_cx) * z / self.d_fx,
                          (vs - self.d_cy) * z / self.d_fy, z], axis=1)
        return local @ self.d_rot.T + self.d_org

    def yaw_from_cloud(self, found):
        """Take yaw from the principal axis of the depth points."""
        placed = [d for d in found if 'xyz' in d]
        if not placed:
            return
        depth = np.frombuffer(self.depth.data, dtype=np.float32).reshape(
            self.depth.height, self.depth.width)
        centres = np.array([d['xyz'][:2] for d in placed])

        for det in placed:
            bbox = det['bbox']
            if self.specs[det['cls']][2]:
                continue                     # cylinder, no yaw
            reach = 0.5 * math.hypot(bbox[0], bbox[1]) + DEPTH_MARGIN
            window = self.depth_window(det['xyz'], reach,
                                       (self.depth.height, self.depth.width))
            if window is None:
                continue
            pts = self.window_cloud(depth, window)
            if len(pts) < DEPTH_MIN_POINTS:
                continue

            lo = self.table_top_z + DEPTH_FLOOR
            hi = self.table_top_z + bbox[2] + DEPTH_MARGIN
            pts = pts[(pts[:, 2] > lo) & (pts[:, 2] < hi)]
            if len(pts) < DEPTH_MIN_POINTS:
                continue

            # give each point to its nearest detection
            gap = np.linalg.norm(pts[:, None, :2] - centres[None, :, :],
                                 axis=2)
            mine = np.argmin(gap, axis=1) == placed.index(det)
            pts = pts[mine & (gap.min(axis=1) < reach)]
            if len(pts) < DEPTH_MIN_POINTS:
                continue

            xy = pts[:, :2] - pts[:, :2].mean(axis=0)
            spread, axes = np.linalg.eigh(xy.T @ xy / len(xy))
            if spread[0] <= 1e-12:
                continue
            elongation = math.sqrt(spread[1] / spread[0])
            if elongation < DEPTH_MIN_ELONGATION:
                continue
            long_axis = axes[:, 1]

            det['yaw'] = math.atan2(long_axis[1], long_axis[0]) % math.pi
            det['yaw_sigma'] = DEPTH_YAW_SIGMA
            det['yaw_from'] = 'depth'
            det['elongation'] = elongation
            # redo the position fit with the new yaw
            self.relocate(det)

    def relocate(self, det):
        """Position fit with a known yaw."""
        bbox, rest_z, _ = self.specs[det['cls']]
        plane_z = self.table_top_z + rest_z
        px = det['px']
        ou, ov = (px[0] + px[2]) / 2.0, (px[1] + px[3]) / 2.0
        offsets = corner_offsets(bbox)
        yaws = np.array([det['yaw']])
        u = np.array([ou])
        v = np.array([ov])
        xyz = self.lift(u, v, plane_z)
        if not np.all(np.isfinite(xyz)):
            return
        for _ in range(FIT_ROUNDS):
            mu, mv = self.project(self.corners(xyz, yaws, offsets))
            u += ou - (mu.min(1) + mu.max(1)) / 2.0
            v += ov - (mv.min(1) + mv.max(1)) / 2.0
            xyz = self.lift(u, v, plane_z)
        mu, mv = self.project(self.corners(xyz, yaws, offsets))
        det['xyz'] = xyz[0]
        det['iou'] = float(extent_iou(mu, mv, px)[0])

# ---------------------------------------------------------------- outputs ---
    def make_2d(self, found, header):
        msg = Detection2DArray()
        msg.header = header
        for det in found:
            u1, v1, u2, v2 = det['px']
            d = Detection2D()
            d.header = header
            d.bbox = BoundingBox2D()
            d.bbox.center = Pose2D()
            d.bbox.center.position = Point2D(x=(u1 + u2) / 2.0,
                                             y=(v1 + v2) / 2.0)
            d.bbox.size_x = u2 - u1
            d.bbox.size_y = v2 - v1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = det['cls']
            hyp.hypothesis.score = det['score']
            d.results.append(hyp)
            msg.detections.append(d)
        return msg

    def make_3d(self, found, header):
        msg = Detection3DArray()
        msg.header.stamp = header.stamp
        msg.header.frame_id = self.target_frame
        for det in found:
            if 'xyz' not in det:
                continue
            x, y, z = det['xyz']
            qx, qy, qz, qw = quat_from_yaw(det['yaw'])
            d = Detection3D()
            d.header = msg.header
            d.bbox = BoundingBox3D()
            d.bbox.center.position.x = float(x)
            d.bbox.center.position.y = float(y)
            d.bbox.center.position.z = float(z)
            d.bbox.center.orientation.x = qx
            d.bbox.center.orientation.y = qy
            d.bbox.center.orientation.z = qz
            d.bbox.center.orientation.w = qw
            d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = det['bbox']
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = det['cls']
            hyp.hypothesis.score = det['score']
            hyp.pose.pose = d.bbox.center
            hyp.pose.covariance = self.covariance(det['yaw_sigma'])
            d.results.append(hyp)
            msg.detections.append(d)
        return msg

    def covariance(self, yaw_sigma):
        """Row-major 6x6 over x, y, z, roll, pitch, yaw."""
        cov = [0.0] * 36
        cov[0] = cov[7] = POS_SIGMA ** 2
        cov[14] = Z_SIGMA ** 2
        cov[21] = cov[28] = TILT_SIGMA ** 2
        cov[35] = yaw_sigma ** 2
        return cov

    def draw(self, frame, found):
        import cv2
        out = frame.copy()
        for det in found:
            u1, v1, u2, v2 = [int(round(c)) for c in det['px']]
            cv2.rectangle(out, (u1, v1), (u2, v2), (0, 220, 0), 2)
            label = '%s %.2f' % (det['cls'], det['score'])
            if 'xyz' in det:
                label += ' %.2f,%.2f' % (det['xyz'][0], det['xyz'][1])
                if det['yaw_sigma'] < YAW_UNKNOWN:
                    label += ' %.0fdeg%s' % (
                        math.degrees(det['yaw']),
                        '' if det.get('yaw_from') == 'depth' else '?')
            cv2.putText(out, label, (u1, max(12, v1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1,
                        cv2.LINE_AA)
        return out


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
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
