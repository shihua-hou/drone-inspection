from glob import glob
from setuptools import find_packages, setup

package_name = "park_inspection"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
        ("share/" + package_name + "/worlds/meshes", glob("worlds/meshes/*")),
        (
            "share/" + package_name + "/models/x500_depth_lidar",
            glob("models/x500_depth_lidar/*"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="fly",
    maintainer_email="fly@example.com",
    description="3D park inspection mission planning and PX4 SITL execution.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "inspection_mission = park_inspection.mission_node:main",
            "online_mapper = park_inspection.mapping_node:main",
        ],
    },
)
