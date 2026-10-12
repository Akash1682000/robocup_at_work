import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_path = get_package_share_directory('robocup_at_work')
    apriltag_cfg = os.path.join(pkg_path, 'config', 'apriltag.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')

    # Real topics are /rgb_camera/*, the urdf remappings don't apply.
    # The sim camera has no distortion, so image_raw is already rectified.
    apriltag = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag',
        output='screen',
        parameters=[apriltag_cfg, {'use_sim_time': use_sim_time}],
        remappings=[
            ('image_rect', '/rgb_camera/image_raw'),
            ('camera_info', '/rgb_camera/camera_info'),
        ],
    )

    identifier = Node(
        package='robocup_at_work',
        executable='workstation_identifier',
        name='workstation_identifier',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # run with the yolo_env python, ultralytics needs a newer opencv
    # than the system one
    detector = Node(
        package='robocup_at_work',
        executable='object_detector',
        name='object_detector',
        output='screen',
        prefix=LaunchConfiguration('detector_python'),
        parameters=[{
            'use_sim_time': use_sim_time,
            'target_frame': LaunchConfiguration('pose_frame'),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'detector_python',
            default_value=os.path.expanduser(
                '~/robocup_ws/yolo_env/bin/python'),
            description='interpreter with ultralytics installed'),
        DeclareLaunchArgument(
            'pose_frame', default_value='base_footprint',
            description='frame the 3D object poses come out in'),
        apriltag,
        identifier,
        detector,
    ])
