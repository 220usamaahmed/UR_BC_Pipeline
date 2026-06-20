from glob import glob

from setuptools import find_packages, setup

package_name = 'bc_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Usama Ahmed Siddiquie',
    maintainer_email='220usamaahmed@gmail.com',
    description='Behaviour-cloning data pipeline: config-driven hard-coded '
                'controllers and bag recording for the UR3e.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sequence_runner = bc_pipeline.sequence_runner:main',
            'scene_publisher = bc_pipeline.scene_publisher:main',
        ],
    },
)
