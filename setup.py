from setuptools import setup, find_packages
"""Setup module for project."""

setup(
    name='imu-prior-learning',
    version='0.1',
    description='Code for semester project at RPG: improve inertial odometry using deep learning',

    author='Guillem Torrente i Marti',
    author_email='tguillem@student.ethz.ch',

    packages=find_packages(exclude=[]),
    python_requires='>=3.5',
    install_requires=[
        'tensorflow==2.0a',
        'numpy',
        'scipy',
        'sklearn',
        'pyquaternion',
        'requests',
        'matplotlib',
        'python-gflags',
        'joblib',
        'PyYAML',
        'pandas'
    ],
)

