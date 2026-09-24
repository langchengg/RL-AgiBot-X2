"""Two independent processes; launch never constructs or resets a simulator."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arguments = [('controller', 'reference_residual', str), ('controller_run', '', str),
                 ('expected_checkpoint_sha256',
                  '11d8f3e203936d2bd49b3b6cfb62e91909c6a8c8825425f540dacc712bb54c64', str),
                 ('seed', '60', int), ('episode_timeout_s', '20.0', float),
                 ('recovery_timeout_s', '30.0', float)]
    parameters = {name: ParameterValue(LaunchConfiguration(name), value_type=kind)
                  for name, _, kind in arguments}
    return LaunchDescription([
        *(DeclareLaunchArgument(name, default_value=default) for name, default, _ in arguments),
        Node(package='x2_recovery', executable='recovery_node', output='screen',
             parameters=[parameters, {'use_sim_time': False}]),
        Node(package='x2_recovery', executable='telemetry_node', output='screen',
             parameters=[{'use_sim_time': False}]),
    ])
