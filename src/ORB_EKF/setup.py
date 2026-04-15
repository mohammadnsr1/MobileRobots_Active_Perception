from glob import glob
from setuptools import find_packages, setup

package_name = "orb_ekf"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="vikas",
    maintainer_email="vikas@example.com",
    description="ORB-SLAM stereo VO and EKF fusion package.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "orb_vo_node = orb_ekf.orb_vo_node:main",
            "fused_output_node = orb_ekf.fused_output_node:main",
        ],
    },
)
