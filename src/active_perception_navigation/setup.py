from glob import glob
from setuptools import find_packages, setup

package_name = "active_perception_navigation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}", ["README.md"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="khaled",
    maintainer_email="khaled@example.com",
    description=(
        "Independent navigation pipeline contribution for active perception using "
        "topic-based confidence evaluation, next-best-view planning, and Nav2 orchestration."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ap_nav_confidence_evaluator = active_perception_navigation.confidence_evaluator:main",
            "ap_nav_nbv_planner = active_perception_navigation.nbv_planner:main",
            "ap_nav_orchestrator = active_perception_navigation.orchestrator:main",
            "ap_nav_safety_monitor = active_perception_navigation.safety_monitor:main",
        ],
    },
)
