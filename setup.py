from setuptools import find_packages
from setuptools import setup

setup(
    author='Max Halford',
    author_email='maxhalford25@gmail.com',
    description='Data science utilities library',
    license='MIT',
    name='xam',
    install_requires=open('requirements/base.txt').read().splitlines(),
    packages=find_packages(exclude=['examples']),
    url='https://github.com/MaxHalford/xam',
    version='0.0.1dev',
)
