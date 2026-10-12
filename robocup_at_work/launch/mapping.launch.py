import os
import re
from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, RegisterEventHandler,
                            TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro


def strip_comments(urdf):
    """Remove xml comments from the expanded urdf.

    gazebo_ros2_control passes the urdf to the controller_manager as a
    yaml param, and newlines or ': ' inside comments break that parse.
    The controller_manager then never starts and every spawner hangs.
    """
    return re.sub(r'<!--.*?-->', '', urdf, flags=re.DOTALL)


def build_full_urdf():
    """Expand the xacro and write a copy for spawn_entity."""
    pkg_path = get_package_share_directory('robocup_at_work')
    xacro_file = os.path.join(pkg_path, 'urdf', 'robot.urdf.xacro')
    full_urdf = strip_comments(xacro.process_file(xacro_file).toxml())

    with open('/tmp/full_robot.urdf', 'w') as f:
        f.write(full_urdf)

    return full_urdf


def generate_launch_description():
    pkg_path = get_package_share_directory('robocup_at_work')
    gazebo_ros_path = get_package_share_directory('gazebo_ros')

    robot_description = build_full_urdf()

    # mapping world, no obstacles
    world_file = os.path.join(pkg_path, 'worlds', 'robocup_arena_mapping.world')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description,
                      'use_sim_time': True}],
        output='screen'
    )

    # joint_state_broadcaster publishes /joint_states, no joint_state_publisher.
    # The controller_manager lives in gzserver, so spawn after the robot.
    controllers = [
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[name, '--controller-manager', '/controller_manager'],
            output='screen',
        )
        for name in ('joint_state_broadcaster',
                     'panda_arm_controller',
                     'panda_hand_controller')
    ]

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_path, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_file}.items()
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', '/tmp/full_robot.urdf',
            '-entity', 'robocup_robot',
            '-x', '2.5',
            '-y', '-2.2',
            '-z', '0.1'
        ],
        output='screen'
    )

    spawn_robot = TimerAction(period=8.0, actions=[spawn_entity])

    # start controllers as soon as the spawn finishes so the arm doesn't sag
    spawn_controllers = RegisterEventHandler(
        OnProcessExit(target_action=spawn_entity, on_exit=controllers))

    # SLAM Toolbox for mapping
    slam_params = os.path.join(pkg_path, 'config', 'slam_params.yaml')
    slam = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                parameters=[slam_params],
                output='screen'
            )
        ]
    )

    # RViz
    rviz_config = os.path.join(pkg_path, 'config', 'mapping.rviz')
    rviz = TimerAction(
        period=15.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                arguments=['-d', rviz_config],
                parameters=[{'use_sim_time': True}],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_robot,
        spawn_controllers,
        slam,
        rviz,
    ])
