from setuptools import setup, find_packages

setup(
    name='eyetracker',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'matplotlib',
        'Pillow',
        'scipy',
        'mpl-point-clicker',
    ],
    entry_points={
        'console_scripts': [
            'eyetracker-run = scripts.run_analysis:main',
        ],
    },
)
