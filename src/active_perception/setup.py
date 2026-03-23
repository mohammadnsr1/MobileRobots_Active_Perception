from setuptools import find_packages, setup

package_name = 'active_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohammadnsr1',
    maintainer_email='nasrmohammad661@gmail.com',
    description=(
        'ROS 2 active perception nodes for cylinder detection and target '
        'estimation from point clouds.'
    ),
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cylinder_finder = active_perception.cylinder_finder:main',
            'pose_estimator = active_perception.pose_estimator:main',
        ],
    },
)
