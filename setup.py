from setuptools import setup
import os
from glob import glob

package_name = 'ros2_aruco'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daichang',
    maintainer_email='18181985920@163.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_generate_marker = ros2_aruco.aruco_generate_marker:main',
            "tip_tracking_node = ros2_aruco.tip_tracking_node:main",
            "board_calibration_node = ros2_aruco.board_calibration_node:main",
            # 用于八棱柱专利实验
            # "board_calibration_node = ros2_aruco.new_calibration_node:main",   
            
            "keyboard_publisher = ros2_aruco.keyboard_publisher:main"
        ],
    },
)
