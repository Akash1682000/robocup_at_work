import os
from glob import glob
from setuptools import setup

package_name = 'robocup_at_work'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.*')),
        (os.path.join('share', package_name, 'config', 'moveit'),
            glob('config/moveit/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        # install models/ recursively so model:// uris resolve
        *[
            (os.path.join('share', package_name, root),
             [os.path.join(root, f) for f in files])
            for root, _dirs, files in os.walk('models') if files
        ],
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akash',
    maintainer_email='a.sudheesh.akash@gmail.com',
    description='RoboCup@Work simulation in ROS 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'workstation_identifier ='
            ' robocup_at_work.workstation_identifier:main',
            'object_detector ='
            ' robocup_at_work.object_detector:main',
            'object_picker ='
            ' robocup_at_work.object_picker:main',
            'mission ='
            ' robocup_at_work.mission:main',
        ],
    },
)
