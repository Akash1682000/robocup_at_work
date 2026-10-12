"""MoveIt move_group for the panda, plus object_picker and mission.

Controllers come from robot_gazebo.launch.py, this only adds planning.

  ros2 launch robocup_at_work moveit.launch.py
  ros2 launch robocup_at_work moveit.launch.py rviz:=true
  ros2 launch robocup_at_work moveit.launch.py picker:=false
"""
import os
import re

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def load_yaml(pkg_path, *parts):
    with open(os.path.join(pkg_path, 'config', 'moveit', *parts)) as f:
        return yaml.safe_load(f)


def generate_launch_description():
    pkg_path = get_package_share_directory('robocup_at_work')

    xacro_file = os.path.join(pkg_path, 'urdf', 'robot.urdf.xacro')
    description = re.sub(r'<!--.*?-->', '',
                         xacro.process_file(xacro_file).toxml(),
                         flags=re.DOTALL)

    with open(os.path.join(pkg_path, 'config', 'moveit', 'robocup.srdf')) as f:
        semantic = f.read()

    ompl = load_yaml(pkg_path, 'ompl_planning.yaml')
    kinematics = load_yaml(pkg_path, 'kinematics.yaml')
    limits = load_yaml(pkg_path, 'joint_limits.yaml')
    controllers = load_yaml(pkg_path, 'moveit_controllers.yaml')

    # without these the scene never tracks /joint_states and plans start
    # from the pose at startup
    scene_monitor = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
        'publish_robot_description_semantic': True,
    }

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            {'robot_description': description},
            {'robot_description_semantic': semantic},
            {'robot_description_kinematics': kinematics},
            {'robot_description_planning': limits},
            {'planning_pipelines': ['ompl']},
            {'default_planning_pipeline': 'ompl'},
            {'ompl': ompl},
            ompl,
            controllers,
            scene_monitor,
            {'use_sim_time': True},
        ],
    )

    rviz_config = os.path.join(pkg_path, 'config', 'moveit.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[
            {'robot_description': description},
            {'robot_description_semantic': semantic},
            {'robot_description_kinematics': kinematics},
            {'use_sim_time': True},
        ],
    )

    picker = Node(
        package='robocup_at_work',
        executable='object_picker',
        name='object_picker',
        output='screen',
        condition=IfCondition(LaunchConfiguration('picker')),
        parameters=[{'use_sim_time': True}],
    )

    mission = Node(
        package='robocup_at_work',
        executable='mission',
        name='mission',
        output='screen',
        condition=IfCondition(LaunchConfiguration('mission')),
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false',
                              description='also start rviz with the MoveIt '
                                          'motion planning panel'),
        DeclareLaunchArgument('picker', default_value='true',
                              description='also start the pick node'),
        DeclareLaunchArgument('mission', default_value='true',
                              description='also start the mission node, which '
                                          'needs navigation up to be useful'),
        move_group,
        rviz,
        picker,
        mission,
    ])
