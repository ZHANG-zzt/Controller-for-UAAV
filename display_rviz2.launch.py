import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.actions import ExecuteProcess
from simple_launch import SimpleLauncher, GazeboBridge


def generate_launch_description():
    package_name = 'fishbot_description'
    urdf_name = "qingzhuan.urdf"
    sdf_name = "fishbot.sdf"

    ld = LaunchDescription()

    pkg_share = FindPackageShare(package=package_name).find(package_name)
    urdf_model_path = os.path.join(pkg_share, f'urdf/{urdf_name}')
    sdf_model_path = os.path.join(os.getenv('HOME'), 'ami_ws', 'src', 'fishbot_description', 'urdf', 'underAUV.sdf')
    world_file_path = os.path.join(os.getenv('HOME'), 'ami_ws', 'src', 'fishbot_description', 'urdf', 'demo_world.sdf')

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        arguments=[urdf_model_path],
        output='screen'
    )

    joint_state_publisher_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
        parameters=[{'use_gui': False}]
    )

    ros_init = Node(
        package='ignition_gazebo',
        executable='ignition gazebo',
        name='ignition_gazebo',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    gazebo_node = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_file_path],
        output='screen'
    )

    reset_node = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so', '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_force_system.so', world_file_path],
        output='log'
    )

    spawn_model_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'qingzhuan',
            '-file', sdf_model_path,
            '-x', '0',
            '-y', '0',
            '-z', '0',
            '-R', '0',
            '-P', '0',
            '-Y', '1.5708'
        ],
        output='screen'
    )

    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
    )

    # Joint bridge topics
    joint_names = ['lb_trust', 'lf_trust', 'rb_trust', 'rf_trust', 'behind', 'front']
    bridge_topics = []
    for joint in joint_names:
        topic = f"/qingzhuan/{joint}_joint/cmd"
        bridge_topics.append((topic, "std_msgs/msg/Float64", "gz.msgs.Double"))

    bridge_nodes = []
    for topic, ros_msg, gz_msg in bridge_topics:
        bridge_nodes.append(ExecuteProcess(
            cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                 f'{topic}@{ros_msg}@{gz_msg}'],
            output='screen'
        ))

    # IMU and odometry bridge topics 
    bridge_2 = [
        ('/qingzhuan/depth', 'ros_gz_interfaces/msg/Altimeter', 'gz.msgs.Altimeter'),
        ('/qingzhuan/imu',   'sensor_msgs/msg/Imu',             'gz.msgs.IMU'),
        ('/model/qingzhuan/odometry', 'nav_msgs/msg/Odometry',  'gz.msgs.Odometry'),
    ]

    bridge_nodes_2 = []
    for topic, ros_msg, gz_msg in bridge_2:
        bridge_nodes_2.append(ExecuteProcess(
            cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                 f'{topic}@{ros_msg}[{gz_msg}'],
            output='screen'
        ))

    pub_reset_pose = ExecuteProcess(
        cmd=['ros2', 'service', 'call', '/world/ocean/set_pose', 'gz.msgs.Pose',
             "{name: 'qingzhuan', position: {x: 0.0, y: 0.0, z: -1.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}"],
        output='screen'
    )

    bridge_trajectory_marker = ExecuteProcess(
        cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
             '/trajectory_marker@visualization_msgs/msg/Marker[gz.msgs.Marker'],
        output='screen'
    )

    pub_velocity_node_lf = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '/qingzhuan/lf_joint/cmd', 'std_msgs/msg/Float64', 'data: 0.0'],
        output='screen'
    )
    pub_velocity_node_lb = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '/qingzhuan/lb_joint/cmd', 'std_msgs/msg/Float64', 'data: -0.0'],
        output='screen'
    )
    pub_velocity_node_rb = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '/qingzhuan/rb_joint/cmd', 'std_msgs/msg/Float64', 'data: 0.0'],
        output='screen'
    )
    pub_velocity_node_rf = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '/qingzhuan/rf_joint/cmd', 'std_msgs/msg/Float64', 'data: -0.0'],
        output='screen'
    )

    pub_lb_trust = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '/qingzhuan/lb_trust_joint/cmd', 'std_msgs/msg/Float64', 'data: 0'],
        output='screen'
    )
    pub_lf_trust = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '/qingzhuan/lf_trust_joint/cmd', 'std_msgs/msg/Float64', 'data: 0'],
        output='screen'
    )
    pub_rf_trust = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '/qingzhuan/rf_trust_joint/cmd', 'std_msgs/msg/Float64', 'data: 0'],
        output='screen'
    )
    pub_rb_trust = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '/qingzhuan/rb_trust_joint/cmd', 'std_msgs/msg/Float64', 'data: 0'],
        output='screen'
    )

    ld.add_action(robot_state_publisher_node)
    # ld.add_action(joint_state_publisher_node)

    ld.add_action(gazebo_node)
    # ld.add_action(load_model_node)
    ld.add_action(spawn_model_node)
    ld.add_action(bridge_trajectory_marker)

    for bridge_node in bridge_nodes:
        ld.add_action(bridge_node)  

    for bridge_2 in bridge_nodes_2:
        ld.add_action(bridge_2)     

    # for bridge_service_node in bridge_service_nodes:
    #     ld.add_action(bridge_service_node)  

    ld.add_action(rviz2_node)

    ld.add_action(pub_lb_trust)
    ld.add_action(pub_lf_trust)
    ld.add_action(pub_rf_trust)
    ld.add_action(pub_rb_trust)
    ld.add_action(pub_reset_pose)

    return ld