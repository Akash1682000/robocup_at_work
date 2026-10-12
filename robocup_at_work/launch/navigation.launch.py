import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_path = get_package_share_directory('robocup_at_work')
    nav2_bringup_path = get_package_share_directory('nav2_bringup')

    nav2_params = os.path.join(pkg_path, 'config', 'nav2_params.yaml')
    map_file = os.path.join(pkg_path, 'maps', 'arena_map.yaml')
    rviz_config = os.path.join(pkg_path, 'config', 'navigation.rviz')

    # use_amcl:=False uses a static map->odom instead (sim only, odom is
    # ground truth). Useful to rule out localization when debugging.
    use_amcl = LaunchConfiguration('use_amcl')

    declare_use_amcl = DeclareLaunchArgument(
        'use_amcl',
        default_value='True',
        description='Run AMCL for map->odom. False falls back to a static identity transform (sim only).'
    )

    # localization
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        parameters=[nav2_params],
        output='screen',
        condition=IfCondition(use_amcl)
    )

    # fallback: static map -> odom
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        parameters=[{'use_sim_time': True}],
        output='screen',
        condition=UnlessCondition(use_amcl)
    )

    # map server
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{
            'use_sim_time': True,
            'yaml_filename': map_file,
        }],
        output='screen'
    )

    # two lifecycle managers, with and without amcl
    lifecycle_with_amcl = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['map_server', 'amcl']
        }],
        output='screen',
        condition=IfCondition(use_amcl)
    )

    lifecycle_map_only = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['map_server']
        }],
        output='screen',
        condition=UnlessCondition(use_amcl)
    )

    # nav2 planner, controller and behaviors
    nav2 = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(nav2_bringup_path, 'launch', 'navigation_launch.py')
                ),
                launch_arguments={
                    'use_sim_time': 'True',
                    'params_file': nav2_params,
                }.items()
            )
        ]
    )

    # rviz
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
        declare_use_amcl,
        amcl,
        static_tf,
        map_server,
        lifecycle_with_amcl,
        lifecycle_map_only,
        nav2,
        rviz,
    ])
