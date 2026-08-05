import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'legacy_bc_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Usama Ahmed Siddiquie',
    maintainer_email='220usamaahmed@gmail.com',
    description='Legacy behavior-cloning pipeline for MoveIt control.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'imitation_moveit_control = '
            'legacy_bc_pipeline.imitation_moveit_control:main',
            'trajectory_replay_control = '
            'legacy_bc_pipeline.trajectory_replay_control:main',
            'trajectory_control = '
            'legacy_bc_pipeline.trajectory_control:main',
            'dataset_recorder = '
            'legacy_bc_pipeline.dataset_recorder:main',
        ],
    },
)
