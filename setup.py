from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='eyetracker',
    version='0.1.0',
    author='lisa Bauer',
    description='Zebrafish eye tracking package',
    packages=find_packages(),
    install_requires=requirements
)
