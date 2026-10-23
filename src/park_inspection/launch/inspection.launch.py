from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    mavros_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("mavros"), "launch", "px4.launch"]
            )
        ),
        launch_arguments={
            "fcu_url": LaunchConfiguration("fcu_url"),
            "gcs_url": LaunchConfiguration("gcs_url"),
        }.items(),
    )
    park_config = PathJoinSubstitution(
        [FindPackageShare("park_inspection"), "config", "park_mvp.yaml"]
    )
    campus_config = PathJoinSubstitution(
        [FindPackageShare("park_inspection"), "config", "campus_obstacles.yaml"]
    )
    mission = Node(
        package="park_inspection",
        executable="inspection_mission",
        name="inspection_mission",
        output="screen",
        parameters=[
            park_config,
            campus_config,
            {
                "enable_online_replanning": ParameterValue(
                    LaunchConfiguration("enable_online_replanning"),
                    value_type=bool,
                )
            },
        ],
    )
    mapping_enabled = IfCondition(LaunchConfiguration("enable_mapping"))
    lidar_enabled = IfCondition(
        PythonExpression(
            [
                "'",
                LaunchConfiguration("enable_mapping"),
                "' == 'true' and '",
                LaunchConfiguration("enable_lidar"),
                "' == 'true'",
            ]
        )
    )
    depth_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="depth_points_bridge",
        output="screen",
        condition=mapping_enabled,
        parameters=[
            {
                "config_file": PathJoinSubstitution(
                    [
                        FindPackageShare("park_inspection"),
                        "config",
                        "gz_points_bridge.yaml",
                    ]
                )
            }
        ],
    )
    lidar_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="lidar_scan_bridge",
        output="screen",
        condition=lidar_enabled,
        parameters=[
            {
                "config_file": PathJoinSubstitution(
                    [
                        FindPackageShare("park_inspection"),
                        "config",
                        "gz_lidar_bridge.yaml",
                    ]
                )
            }
        ],
    )
    mapper = Node(
        package="park_inspection",
        executable="online_mapper",
        name="online_mapper",
        output="screen",
        condition=mapping_enabled,
        parameters=[
            park_config,
            campus_config,
            {
                "enable_lidar": ParameterValue(
                    LaunchConfiguration("enable_lidar"),
                    value_type=bool,
                ),
                "enable_exploration": ParameterValue(
                    LaunchConfiguration("enable_exploration"),
                    value_type=bool,
                ),
            },
        ],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "fcu_url",
                default_value="udp://:14540@127.0.0.1:14557",
                description="MAVROS FCU connection URL for PX4 SITL",
            ),
            DeclareLaunchArgument(
                "gcs_url",
                default_value="",
                description="Optional MAVROS ground-control connection URL",
            ),
            DeclareLaunchArgument(
                "enable_mapping",
                default_value="true",
                description=(
                    "Bridge the depth camera and compare the online map with the known map"
                ),
            ),
            DeclareLaunchArgument(
                "enable_lidar",
                default_value="false",
                description=(
                    "Also bridge the 16-beam lidar. Requires the x500_depth_lidar model"
                ),
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Open RViz2 with the park map, path, and lidar cloud",
            ),
            DeclareLaunchArgument(
                "enable_exploration",
                default_value="false",
                description=(
                    "Publish a FUEL-style frontier tour. The inspection route is unchanged"
                ),
            ),
            DeclareLaunchArgument(
                "enable_online_replanning",
                default_value="false",
                description=(
                    "Experimental fused-map replanning; enables sensor gating and LOITER on failure"
                ),
            ),
            mavros_launch,
            mission,
            depth_bridge,
            lidar_bridge,
            mapper,
            Node(
                package="rviz2",
                executable="rviz2",
                name="inspection_rviz",
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_rviz")),
                arguments=[
                    "-d",
                    PathJoinSubstitution(
                        [
                            FindPackageShare("park_inspection"),
                            "rviz",
                            "inspection.rviz",
                        ]
                    ),
                ],
            ),
        ]
    )
