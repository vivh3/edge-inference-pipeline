from setuptools import setup

package_name = "edge_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="vivh3",
    maintainer_email="vivsterhuang@gmail.com",
    description="Thin ROS 2 wrappers over the stdlib-only perception core.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "capture_node = edge_perception.capture_node:main",
            "inference_node = edge_perception.inference_node:main",
        ],
    },
)
