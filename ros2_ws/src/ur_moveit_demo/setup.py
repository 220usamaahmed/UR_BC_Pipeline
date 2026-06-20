from setuptools import find_packages, setup

package_name = 'ur_moveit_demo'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/record_motion.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Usama Ahmed Siddiquie',
    maintainer_email='220usamaahmed@gmail.com',
    description='Teaching demo: MoveIt path planning with obstacle avoidance for the UR3e.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'scene_builder          = ur_moveit_demo.scene_builder:main',
            'motion_planner         = ur_moveit_demo.motion_planner:main',
            'plan_and_execute       = ur_moveit_demo.plan_and_execute:main',
            'obstacle_markers       = ur_moveit_demo.obstacle_markers:main',
            'constrained_cartesian  = ur_moveit_demo.constrained_cartesian:main',
        ],
    },
)
