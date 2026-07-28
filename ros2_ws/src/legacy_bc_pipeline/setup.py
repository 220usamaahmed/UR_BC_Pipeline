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
        ],
    },
)
