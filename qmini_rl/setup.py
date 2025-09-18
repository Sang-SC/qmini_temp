from setuptools import find_packages
from distutils.core import setup

setup(
    name='qmini_rl',
    version='1.0.0',
    author='Unitree Robotics',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='support@unitree.com',
    description='Qmini RL GYM',
    install_requires=['isaacgym',
                      'rsl-rl',
                      'matplotlib',
                      'numpy==1.23.0',
                      'tensorboard',
                      'setuptools==59.5.0',]
)